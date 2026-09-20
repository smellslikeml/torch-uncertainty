# Validation — h-calibration post-processing scaler

## What is validated here (CPU, no model training)

`tests/post_processing/test_scalers.py::TestHCalibrationScaler` mirrors the sibling
scaler tests and covers:

- **Registry/import resolution**: `HCalibrationScaler` is importable from
  `torch_uncertainty.post_processing` (both `__init__` re-exports).
- **Valid probabilities**: after `fit` on small synthetic dataloaders (binary 1-D
  logits and 3-class logits, reusing the existing fixtures), `forward` returns
  finite calibrated logits whose softmax/sigmoid probabilities are non-negative and
  sum to one (multiclass) or lie in `[0, 1]` (binary).
- **Monotonicity**: the fitted piecewise-linear map is non-decreasing over the
  log-probability domain, and ranking (hence argmax predictions) is preserved
  end-to-end.
- **Calibration improvement**: on a synthetic overconfident 3-class problem
  (seeded), the expected calibration error computed with the package metric
  `torch_uncertainty.metrics.CalibrationError` strictly drops after fitting
  (e.g. ECE 0.43 → 0.20 with 60 epochs, → 0.07 with 100 epochs).
- **Weighting variants**: the faiss-free weighting branches (`cluaceweighted`,
  `aceweighted`, `eceweighted`, `unweighted`) each fit and produce finite outputs.
- **Constructor validation**: argument errors raise `ValueError` like the sibling
  scalers.

Run with `uv run pytest tests/post_processing/test_scalers.py` (~3 s on CPU).

## Deferred validation (NOT gated here)

The paper's full claim — state-of-the-art post-hoc calibration across standard
benchmarks (CIFAR-10/100, ImageNet, ...) — requires real pre-trained model logits,
which this repository's CI does not ship. To reproduce it:

1. Download the reference logits/labels (Hugging Face dataset `WJHuang/calibration`,
   or re-extract logits from TorchUncertainty-trained classifiers on a held-out
   calibration split).
2. Fit `HCalibrationScaler` on the calibration split of each model and evaluate
   ECE / CWECE / NLL (TorchUncertainty metrics) on the test split against the
   other scalers (temperature, vector, matrix, Dirichlet, histogram binning, BBQ,
   isotonic regression).
3. Compare with the reference implementation's reported results (Tables of
   arXiv:2506.17968); expect differences of the order of the port deviations
   listed below.

## Self-review (port notes)

- **Call site**: new method file
  `src/torch_uncertainty/post_processing/calibration/h_calibration.py`, exported in
  `post_processing/calibration/__init__.py` and `post_processing/__init__.py`,
  documented in `docs/source/api.rst` + `docs/source/references.rst`, tested in
  `tests/post_processing/test_scalers.py` — matching the touch set of the merged
  scaler/DEUP contributions.
- **Structure**: `HCalibrationScaler` subclasses `PostProcessing` directly and
  mirrors `IsotonicRegressionScaler` (`fit(dataloader, progress)` extracting
  logits+labels via the shared `utils`, `forward` returning calibrated logits in
  the sibling scalers' log-space convention). It does **not** subclass the linear
  `Scaler` base, which is specific to LBFGS logit-scaling.
- **Recalibrator variant**: the piecewise-linear monotonic map
  (`models/monotonic_pwlinear.py` of the reference) is ported — one variant, as
  specified. It was preferred over the nonlinear UMNN variant because it is
  dependency-free, has a single hyperparameter (`num_segments`), and is exactly
  monotonic by construction.
- **Loss**: the core error-bounded ECE surrogate of `loss/hCalib.py` is ported:
  sorted-event moving-average gap, hinge beyond `epsilon`, leave-one-class-out
  averaging, `lossweight` scaling. Only the core weighting branches are kept
  (`cluaceweighted` — the reference default, `aceweighted`, `eceweighted`,
  `unweighted`).
- **Trimmed from the reference**: `faiss` (k-means replaced by a deterministic
  pure-torch 1-D Lloyd's algorithm with quantile init), `KDEpy` (evaluation-metric
  only), `loss/mutuinfo_binning.py` (`mib*` variants), the density/distance
  weighting branches (`denweighted`, `dstweighted`, `soft/harddsteceweighted`),
  the stochastic dropout smoothing branch, and the `ThreadPoolExecutor` loss pool.
  No new dependencies: pure PyTorch (scikit-learn not even needed, unlike
  `IsotonicRegressionScaler`).
- **Deviations from the reference** (each documented in the module docstrings):
  1. The reference's masking-based interpolation of the piecewise-linear map is
     reformulated as the exact integrated-slope form: same parameterization
     (equal-width segments, `abs`-constrained slopes, identity at init) but
     provably continuous and monotonic (the reference's form can jump at segment
     boundaries once slopes deviate from 1).
  2. `faiss.Kmeans` → deterministic torch 1-D k-means (quantile init, 25 Lloyd
     iterations — faiss' default).
  3. The reference's batch-level optimization loop is replaced by full-batch Adam
     epochs over the extracted calibration logits (the objective is a set-level
     statistic).
  4. The reference's `eceweighted` returns a mean where `clueceweighted` returns a
     sum; the frequency-weighted combination is normalized to a sum here (constant
     factor, absorbed by `loss_weight`).
- **Out of scope** (intentionally): the reference's training-time model selection
  (dECE/CWECE_a/NLL selectors, early stopping, checkpointing) and its evaluation
  suite; the UMNN recalibrator variant; multi-benchmark evaluation (deferred as
  described above).

## Attribution

Ported with attribution from the MIT-licensed reference implementation
`github.com/WenjianHuang93/h-Calibration` (© 2025 Wenjian Huang). Paper:
Huang *et al.*, "h-calibration: Rethinking Classifier Recalibration with
Probabilistic Error-Bounded Objective", IEEE TPAMI 47(10):9023–9042, 2025
([arXiv:2506.17968](https://arxiv.org/abs/2506.17968)).
