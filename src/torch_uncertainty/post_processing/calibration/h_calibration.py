import logging
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import PostProcessing

from .utils import _determine_dimensionality, _extract_data

Weighting = Literal["cluaceweighted", "aceweighted", "eceweighted", "unweighted"]


class _PiecewiseLinear(nn.Module):
    r"""Learnable-slope monotonic piecewise-linear map over log-probabilities.

    The input domain :math:`(-\infty, 0]` (log-probabilities) is mirrored to
    :math:`[0, +\infty)` and split into ``num_segments`` equal-width segments
    over ``[0, value_range]`` (plus linear extrapolation beyond). Each segment
    carries a learnable non-negative slope, and the map is the integral of
    those slopes, so it is continuous and non-decreasing by construction. Unit
    slopes yield the identity map.

    Port (MIT, © 2025 Wenjian Huang) of ``models/monotonic_pwlinear.py`` from
    the reference implementation of `h-calibration
    <https://github.com/WenjianHuang93/h-Calibration>`__. The reference's
    masking-based interpolation is reformulated as the exact integrated-slope
    form, which keeps the same parameterization (equal-width segments,
    :math:`|\cdot|`-constrained slopes, identity at initialization) while
    guaranteeing continuity and monotonicity.

    Args:
        num_segments: Number of linear segments of the map. Defaults to ``20``.
        value_range: Width of the segment grid, in negative log-probability.
            Defaults to ``100.0``.

    References:
        [1] `Huang, W., Cao, G., Xia, J., Chen, J., Wang, H., & Zhang, J.
        (2025). h-calibration: Rethinking Classifier Recalibration with
        Probabilistic Error-Bounded Objective. IEEE TPAMI 2025
        <https://arxiv.org/abs/2506.17968>`_.
    """

    def __init__(self, num_segments: int = 20, value_range: float = 100.0) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.value_range = value_range
        self.register_buffer("steps", torch.linspace(0, value_range, num_segments + 1))
        self.slopes = nn.Parameter(torch.ones(num_segments))

    def forward(self, log_probs: Tensor) -> Tensor:
        """Map log-probabilities through the monotonic piecewise-linear map.

        Args:
            log_probs: Tensor of non-positive values (e.g. log-softmax outputs)
                of arbitrary shape.

        Returns:
            Tensor: The mapped values, of the same shape, with ranking
                preserved (the map is non-decreasing).
        """
        slopes = self.slopes.abs()
        width = self.value_range / self.num_segments
        # Value of the map at each segment start: integral of the slopes.
        starts = torch.cat((torch.zeros(1), torch.cumsum(slopes * width, dim=0)[:-1]))

        positions = -log_probs.flatten()
        clipped = positions.clamp(max=self.value_range)
        # Index of the segment each position falls into.
        seg_idx = (clipped.unsqueeze(-1) >= self.steps[1:]).sum(-1).clamp(max=self.num_segments - 1)
        within = clipped - self.steps[seg_idx]
        # Linear extrapolation with the last slope beyond the grid.
        tail = (positions - self.value_range).clamp(min=0) * slopes[-1]
        values = starts[seg_idx] + slopes[seg_idx] * within + tail
        return -values.view_as(log_probs)


class _MonotonicLogitCalibrator(nn.Module):
    r"""Monotonic recalibrator applying a piecewise-linear map to logits.

    Logits are normalized with log-softmax (making the map invariant to the
    scale and shift of the input logits) before being mapped element-wise by
    the monotonic :class:`_PiecewiseLinear` map. Since the map is shared
    across classes and non-decreasing, it preserves the ranking of the input
    logits, hence the model's predictions.

    Args:
        num_segments: Number of linear segments of the map. Defaults to ``20``.
        value_range: Width of the segment grid. Defaults to ``100.0``.
    """

    def __init__(self, num_segments: int = 20, value_range: float = 100.0) -> None:
        super().__init__()
        self.piecewise_linear = _PiecewiseLinear(num_segments, value_range)

    def forward(self, logits: Tensor) -> Tensor:
        """Recalibrate a batch of logits.

        Args:
            logits: Tensor of logits of shape ``(..., num_classes)``.

        Returns:
            Tensor: The recalibrated logits, of the same shape.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        return self.piecewise_linear(log_probs)


def _kmeans1d(values: Tensor, num_clusters: int, num_iterations: int = 25) -> Tensor:
    """Cluster 1-D values with a deterministic k-means (Lloyd's algorithm).

    Replacement for the ``faiss.Kmeans`` call of the reference implementation:
    centroids are initialized at evenly spaced quantiles (deterministic) and
    empty clusters keep their previous centroid.

    Args:
        values: The 1-D values to cluster.
        num_clusters: The number of clusters.
        num_iterations: The number of Lloyd's iterations. Defaults to ``25``.

    Returns:
        Tensor: The integer cluster index of each value.
    """
    values = values.detach().flatten()
    num_clusters = min(num_clusters, values.numel())
    quantiles = torch.linspace(0, 1, num_clusters, device=values.device)
    centroids = torch.quantile(values, quantiles)
    assignments = torch.zeros_like(values, dtype=torch.long)
    for _ in range(num_iterations):
        assignments = (values.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        counts = torch.bincount(assignments, minlength=num_clusters)
        sums = torch.zeros_like(centroids).index_add_(0, assignments, values)
        centroids = torch.where(counts > 0, sums / counts.clamp(min=1), centroids)
    return assignments


def _hcalib_error(
    probs: Tensor, onehot_targets: Tensor, epsilon: float, window_length: int
) -> Tensor:
    """Compute the event-wise error-bounded miscalibration of a probability set.

    All class-event probabilities are sorted by calibrated log-probability and
    smoothed with a moving average over ``window_length + 1`` neighboring
    events. Within each window, the gap between the mean probability of
    non-occurring events and the mean complementary probability of occurring
    events estimates the local miscalibration; only the excess over the
    tolerance ``epsilon`` is penalized.

    Args:
        probs: Calibrated probabilities of shape ``(num_samples, num_classes)``.
        onehot_targets: One-hot encoded targets of the same shape.
        epsilon: Tolerance of the error-bounded objective.
        window_length: Length of the smoothing window.

    Returns:
        Tensor: The event-wise hinge miscalibration values.
    """
    prob_evt = probs.flatten()
    occurmark = onehot_targets.flatten()
    _, order = torch.sort(torch.log(prob_evt + 1e-20))
    prob_evt = prob_evt.gather(0, order)
    occurmark = occurmark.gather(0, order)

    pad = window_length // 2
    window = torch.ones(1, 1, window_length + 1, device=prob_evt.device) / (window_length + 1)
    prob_not_occurred = (prob_evt * (1.0 - occurmark)).reshape(1, 1, -1)
    inv_prob_occurred = ((1.0 - prob_evt) * occurmark).reshape(1, 1, -1)
    prob_not_occurred = F.pad(prob_not_occurred, (pad, pad)).float()
    inv_prob_occurred = F.pad(inv_prob_occurred, (pad, pad)).float()
    ma_not_occurred = F.conv1d(prob_not_occurred, window).flatten()
    ma_occurred = F.conv1d(inv_prob_occurred, window).flatten()

    # Local miscalibration gap, penalized only beyond the tolerance epsilon.
    gap = ma_not_occurred - ma_occurred
    gap = F.relu(gap) + F.relu(-gap)
    return F.relu(gap - epsilon)


def _equal_width_indicators(probs: Tensor, num_bins: int) -> Tensor:
    """Assign each probability to the equal-width bins covering ``[0, 1]``.

    Args:
        probs: The flattened probabilities to bin.
        num_bins: The number of bins.

    Returns:
        Tensor: Boolean indicators of shape ``(num_bins, num_samples)``.
    """
    edges = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    indicators = (probs[None, :] >= edges[:-1, None]) & (probs[None, :] <= edges[1:, None])
    return indicators.to(probs.dtype)


def _bin_losses(error: Tensor, indicators: Tensor) -> Tensor:
    """Average the event-wise error within each bin.

    Args:
        error: The event-wise hinge miscalibration values.
        indicators: Boolean bin indicators of shape ``(num_bins, num_events)``.

    Returns:
        Tensor: The mean error of each bin.
    """
    return (error[None, :] * indicators).sum(dim=1) / (indicators.sum(dim=1) + 1e-5)


def _hcalib_subloss(
    probs: Tensor,
    onehot_targets: Tensor,
    weighting: Weighting,
    num_bins: int,
    epsilon: float,
    window_length: int,
) -> Tensor:
    """Aggregate the error-bounded miscalibration with a weighting scheme.

    Args:
        probs: Calibrated probabilities of shape ``(num_samples, num_classes)``.
        onehot_targets: One-hot encoded targets of the same shape.
        weighting: The weighting scheme (see :class:`HCalibrationScaler`).
        num_bins: The number of bins (or clusters) used by the binned schemes.
        epsilon: Tolerance of the error-bounded objective.
        window_length: Length of the smoothing window.

    Returns:
        Tensor: The scalar sub-loss.
    """
    error = _hcalib_error(probs, onehot_targets, epsilon, window_length)
    prob_evt = probs.flatten()

    if weighting == "unweighted":
        return error.mean()

    if weighting == "cluaceweighted":
        # Cluster-averaged (ACE-style) weighting: k-means bins on the
        # calibrated probabilities, then uniform averaging of the bin losses.
        assignments = _kmeans1d(prob_evt, num_bins)
        indicators = F.one_hot(assignments, num_classes=num_bins).t().to(probs.dtype)
        return _bin_losses(error, indicators).mean()

    indicators = _equal_width_indicators(prob_evt, num_bins)
    bin_loss = _bin_losses(error, indicators)
    if weighting == "aceweighted":
        return bin_loss.mean()

    # eceweighted: weight each equal-width bin by its top-1 frequency.
    top1_probs = probs.max(dim=-1).values
    counts = _equal_width_indicators(top1_probs, num_bins).sum(dim=1)
    return (bin_loss * counts / counts.sum()).sum()


def _hcalibration_loss(
    probs: Tensor,
    labels: Tensor,
    num_classes: int,
    weighting: Weighting,
    num_bins: int,
    epsilon: float,
    window_length: int,
    leave_one_class_out: bool,
) -> Tensor:
    """Compute the h-calibration loss (error-bounded ECE surrogate).

    Following the reference implementation, the loss is averaged over
    leave-one-class-out subsets: for each class, the samples whose target is
    that class are left out and the miscalibration of the remaining samples is
    measured. This covers both top-label and class-wise calibration.

    Args:
        probs: Calibrated probabilities of shape ``(num_samples, num_classes)``.
        labels: Integer targets of shape ``(num_samples,)``.
        num_classes: The number of classes.
        weighting: The weighting scheme (see :class:`HCalibrationScaler`).
        num_bins: The number of bins (or clusters) used by the binned schemes.
        epsilon: Tolerance of the error-bounded objective.
        window_length: Length of the smoothing window.
        leave_one_class_out: Whether to average the loss over
            leave-one-class-out subsets.

    Returns:
        Tensor: The scalar h-calibration loss.
    """
    onehot_targets = F.one_hot(labels, num_classes).to(probs.dtype)
    sublosses = []
    if leave_one_class_out:
        for evtid in range(num_classes):
            keep = labels != evtid
            if keep.any():
                sublosses.append(
                    _hcalib_subloss(
                        probs[keep],
                        onehot_targets[keep],
                        weighting,
                        num_bins,
                        epsilon,
                        window_length,
                    )
                )
    if not sublosses:  # coverage: ignore
        sublosses.append(
            _hcalib_subloss(probs, onehot_targets, weighting, num_bins, epsilon, window_length)
        )
    return torch.stack(sublosses).mean()


class HCalibrationScaler(PostProcessing):
    r"""h-calibration post-processing for calibrated probabilities
    (Huang et al., 2025).

    Unlike the linear logit scalers (temperature, vector, matrix, Dirichlet)
    and the binning methods (histogram binning, BBQ), h-calibration fits a
    *parametric monotonic map* on a differentiable error-bounded surrogate of
    the expected calibration error. A learnable-slope piecewise-linear
    monotonic recalibrator is optimized on the calibration logits so that the
    local gap between predicted probabilities and observed frequencies stays
    within a tolerance :math:`\epsilon`:

    .. math::
        \mathcal{L} = \frac{1}{C}\sum_{k=1}^{C} \frac{1}{M}\sum_{m=1}^{M}
        \left[\,|g^{(k)}_m| - \epsilon\,\right]_{+},

    where :math:`g^{(k)}_m` is the moving-average miscalibration gap of the
    :math:`m`-th sorted event window of the :math:`k`-th leave-one-class-out
    subset, and :math:`M` depends on the chosen bin weighting. Because the
    recalibrator is monotonic, the ranking of the logits — hence the model's
    predictions — is preserved.

    For binary classification (single-logit outputs), the logits are
    internally augmented to the two-class event space
    :math:`(0, \text{logit})` and the calibrated positive-class logit is
    returned.

    Args:
        model: Model to calibrate. Defaults to ``None``.
        num_segments: Number of linear segments of the monotonic map.
            Defaults to ``20``.
        weighting: Weighting of the error-bounded objective: ``"cluaceweighted"``
            (k-means bins, uniformly averaged — the reference default),
            ``"aceweighted"`` (equal-width bins, uniformly averaged),
            ``"eceweighted"`` (equal-width bins weighted by top-1 frequency)
            or ``"unweighted``. Defaults to ``"cluaceweighted"``.
        num_bins: Number of bins (or clusters) of the binned weighting schemes.
            Defaults to ``15``.
        epsilon: Tolerance of the error-bounded objective. Defaults to
            ``1e-20``.
        window_length: Length of the event smoothing window (rounded up to an
            even value). Defaults to ``200``.
        lr: Learning rate of the optimizer. Defaults to ``1e-2``.
        num_epochs: Number of optimization epochs on the calibration set.
            Defaults to ``100``.
        loss_weight: Scaling factor of the loss, following the reference
            implementation. Defaults to ``1e5``.
        leave_one_class_out: Whether to average the loss over
            leave-one-class-out subsets. Defaults to ``True``.
        eps: Small value for stability when converting probs back to logits.
            Defaults to ``1e-6``.
        device: Device to use for tensor operations. Defaults to ``None``.

    Attributes:
        calibrator: The fitted monotonic recalibrator.
        num_classes: The number of classes detected at fit time (``1`` for
            binary single-logit outputs).

    References:
        [1] `Huang, W., Cao, G., Xia, J., Chen, J., Wang, H., & Zhang, J.
        (2025). h-calibration: Rethinking Classifier Recalibration with
        Probabilistic Error-Bounded Objective. IEEE TPAMI 2025
        <https://arxiv.org/abs/2506.17968>`_.

    Note:
        This scaler is a port of the MIT-licensed reference implementation
        (`github.com/WenjianHuang93/h-Calibration <https://github.com/WenjianHuang93/h-Calibration>`__,
        © 2025 Wenjian Huang), adapted to TorchUncertainty. The
        piecewise-linear recalibrator variant is ported (rather than the
        nonlinear UMNN one) as it is dependency-free. The reference's
        ``faiss`` k-means is replaced by an equivalent deterministic pure-torch
        1-D k-means, and only the core weighting branches are kept: the
        faiss/KDEpy dependencies and the mutual-information binning of the
        reference are not ported.

    Warning:
        The h-calibration loss loops over the classes, so fitting is slower
        for large numbers of classes. As for every post-hoc calibrator, the
        calibration set must be disjoint from the training and test sets.
    """

    num_classes: int

    def __init__(
        self,
        model: nn.Module | None = None,
        num_segments: int = 20,
        weighting: Weighting = "cluaceweighted",
        num_bins: int = 15,
        epsilon: float = 1e-20,
        window_length: int = 200,
        lr: float = 1e-2,
        num_epochs: int = 100,
        loss_weight: float = 1e5,
        leave_one_class_out: bool = True,
        eps: float = 1e-6,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        if num_segments < 1:
            raise ValueError(
                f"The number of segments must be strictly positive. Got {num_segments}."
            )
        if num_bins < 2:
            raise ValueError(f"The number of bins must be at least 2. Got {num_bins}.")
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative. Got {epsilon}.")
        if window_length < 2:
            raise ValueError(f"The window length must be at least 2. Got {window_length}.")
        if lr <= 0:
            raise ValueError(f"The learning rate must be strictly positive. Got {lr}.")
        if num_epochs < 1:
            raise ValueError(f"The number of epochs must be strictly positive. Got {num_epochs}.")
        super().__init__(model)
        self.num_segments = num_segments
        self.weighting = weighting
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.window_length = window_length + (window_length % 2)
        self.lr = lr
        self.num_epochs = num_epochs
        self.loss_weight = loss_weight
        self.leave_one_class_out = leave_one_class_out
        self.eps = eps
        self.device = device
        self.calibrator = _MonotonicLogitCalibrator(num_segments)

    @staticmethod
    def _expand_binary_logits(logits: Tensor) -> Tensor:
        """Augment single-logit outputs to the two-class event space."""
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        return torch.cat((torch.zeros_like(logits), logits), dim=-1)

    def fit(
        self,
        dataloader: DataLoader,
        progress: bool = True,
    ) -> None:
        """Fit the monotonic recalibrator on the calibration data.

        The calibration logits and targets are extracted once, then the
        piecewise-linear monotonic map is optimized on the error-bounded
        h-calibration loss for ``num_epochs`` full-batch Adam steps.

        Args:
            dataloader: Dataloader providing the calibration data (logits and
                targets).
            progress: Whether to show a progress bar during data extraction.
                Defaults to ``True``.
        """
        if self.model is None or isinstance(self.model, nn.Identity):  # coverage: ignore
            logging.warning(
                "model is None. Fitting post_processing method on the dataloader's data directly."
            )
            self.model = nn.Identity()

        all_logits, all_labels = _extract_data(
            dataloader=dataloader, model=self.model, device=self.device, progress=progress
        )
        num_classes, _, _ = _determine_dimensionality(all_logits, all_labels)
        labels = all_labels.long()
        logits = self._expand_binary_logits(all_logits) if num_classes == 1 else all_logits
        # Binary single-logit outputs are calibrated over the two-class event space.
        event_classes = 2 if num_classes == 1 else num_classes
        self.num_classes = num_classes

        self.calibrator.to(all_logits.device)
        optimizer = torch.optim.Adam(self.calibrator.parameters(), lr=self.lr)
        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            probs = torch.softmax(self.calibrator(logits), dim=-1)
            loss = self.loss_weight * _hcalibration_loss(
                probs,
                labels,
                event_classes,
                self.weighting,
                self.num_bins,
                self.epsilon,
                self.window_length,
                self.leave_one_class_out,
            )
            loss.backward()
            optimizer.step()
        self.trained = True

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the fitted recalibrator and return calibrated logits.

        The forward pass transforms the model logits with the fitted monotonic
        map. The calibrated probabilities are converted back into the logit
        space for compatibility with downstream loss functions or metrics.

        Args:
            inputs: Input logits to be calibrated.

        Returns:
            Tensor: Calibrated logits.
        """
        if self.model is None:  # coverage: ignore
            raise ValueError("Provide a model before calling forward.")
        if not self.trained:
            logging.warning("Scaler not trained. Returning raw predictions.")
            return self.model(inputs)

        logits = self.model(inputs)
        if self.num_classes == 1:
            logits = self._expand_binary_logits(logits)
        calibrated = self.calibrator(logits)
        probs = torch.softmax(calibrated, dim=-1)

        if self.num_classes > 1:
            return torch.log(probs.clamp(self.eps, 1 - self.eps))
        # Binary case: return the calibrated logit of the positive class.
        positive_probs = probs[..., 1].clamp(self.eps, 1 - self.eps)
        return torch.logit(positive_probs, eps=self.eps).reshape(inputs.shape)
