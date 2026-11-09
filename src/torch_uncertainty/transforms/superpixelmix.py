"""SuperpixelMix data augmentation for semantic segmentation.

Code adapted from the official Superpixel-Mix implementation
(https://github.com/giannifranchi/deeplabv3-superpixelmix), distributed
under the MIT license.
"""

from importlib import util

import numpy as np
import torch
from torch import Tensor, nn

if util.find_spec("skimage"):
    from skimage.segmentation import slic

    skimage_installed = True
else:  # coverage: ignore
    skimage_installed = False


class SuperpixelMix(nn.Module):
    def __init__(
        self,
        n_segments: int = 200,
        compactness: float = 10.0,
        prop: float = 0.3,
    ) -> None:
        """SuperpixelMix augmentation for semantic segmentation.

        For each image of the batch, a binary superpixel mask is built by
        setting to zero a random proportion :attr:`prop` of the superpixels
        computed with SLIC. Each image is then mixed with the next image of
        the batch (in a circular manner) along this mask, and the label maps
        are mixed accordingly with a boolean selection so that they remain
        valid integer segmentation maps.

        Args:
            n_segments: Approximate number of superpixels computed by SLIC.
                Defaults to ``200``.
            compactness: Balance between color similarity and spatial
                proximity of the superpixels. Defaults to ``10.0``.
            prop: Proportion of superpixels set to zero in the mask.
                Defaults to ``0.3``.

        Reference:
            "Robust Semantic Segmentation with Superpixel-Mix" (BMVC 2021)
            https://arxiv.org/abs/2108.00968.
        """
        super().__init__()
        if not skimage_installed:  # coverage: ignore
            raise ImportError(
                "The scikit-image library is not installed. Please install "
                "torch_uncertainty with the image option: "
                """pip install -U "torch_uncertainty[image]"."""
            )
        if n_segments <= 0:
            raise ValueError(f"n_segments must be strictly positive. Got {n_segments}.")
        if compactness <= 0:
            raise ValueError(f"compactness must be strictly positive. Got {compactness}.")
        if not 0 < prop < 1:
            raise ValueError(f"prop must be in the (0, 1) interval. Got {prop}.")
        self.n_segments = n_segments
        self.compactness = compactness
        self.prop = prop

    def _superpixel_mask(self, img: Tensor) -> Tensor:
        """Compute a random binary superpixel mask for a single image.

        Args:
            img: Image of shape [C, H, W].

        Returns:
            Tensor: Binary mask of shape [H, W] whose value is 0 on the
                randomly selected superpixels and 1 elsewhere.
        """
        segments = slic(
            img.detach().permute(1, 2, 0).cpu().numpy(),
            n_segments=self.n_segments,
            compactness=self.compactness,
            channel_axis=-1,
        )
        labels = np.unique(segments)
        # Select at least one superpixel so the augmentation is never a silent
        # no-op: int() alone floors to zero whenever SLIC returns few regions
        # (e.g. a single superpixel on low-structure inputs).
        n_selected = max(1, int(labels.size * self.prop))
        selected = np.random.default_rng().permutation(labels)[:n_selected]
        mask = np.ones(segments.shape, dtype=np.float32)
        mask[np.isin(segments, selected)] = 0
        return torch.from_numpy(mask).to(img.device)

    def forward(self, imgs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Mix images and label maps along random superpixel masks.

        Each sample i is mixed with sample (i + 1) % B of the batch, where B
        is the batch size.

        Args:
            imgs: Batch of images of shape [B, C, H, W].
            targets: Batch of integer label maps of shape [B, H, W].

        Returns:
            tuple[Tensor, Tensor]: The mixed images and the mixed label maps,
                with the same dtype as ``targets``.
        """
        masks = torch.stack([self._superpixel_mask(img) for img in imgs])
        next_imgs = torch.roll(imgs, shifts=-1, dims=0)
        next_targets = torch.roll(targets, shifts=-1, dims=0)
        mixed_imgs = masks.unsqueeze(1) * imgs + (1 - masks).unsqueeze(1) * next_imgs
        mixed_targets = torch.where(masks.bool(), targets, next_targets)
        return mixed_imgs, mixed_targets
