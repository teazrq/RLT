# RLT development

* Report insufficient positive variable-sampling support as a catchable error
  instead of looping. Skip zero-weight SIR slices and use the established
  fallback when the response direction cannot be estimated. Correct the
  unweighted supremum-logrank variance in linear-combination searches.
* Preserve training label codes for factors, ordered factors, and character
  data-frame columns in prediction and kernels. Valid numerical training codes
  remain supported. Legacy fits need their original mapping or refitting before
  accepting label inputs.
* Accumulate survival hazards on the full training grid before selecting
  prediction times, and return populated per-tree hazards with `keep.all`.
* Correct IJ count moments to use floating-point arithmetic in all model families.
* Fix linear combination training-kernel arguments and singleton OOB counts.
* Handle one-point band dimensions, feasible smoothing bases, and an explicit
  naive fallback for one or two times. Document the CHF-based band reference
  separately from the linear combination mean tree product-limit estimate.
* Reject reduced training-time grids and event-free survival outcomes before
  native fitting. Use `band.grid.size` in `predict()` for reduced evaluation.
* Treat survival importance without comparable OOB pairs as unavailable; do not
  fabricate zero importance or matched importance variance from missing evidence.
  Contain fitting exceptions across nested parallel forests.
* Validate finite positive-total weights and report zero-mass sampled trees,
  including embedded trees, as catchable errors. Preserve individual zero weights
  and reject splits with undefined weighted children.

# RLT 6.1.1

* CRAN submission cleanup: fixed vignette pre-building, DESCRIPTION fields,
  man page examples, and C++ compiler warning.

# RLT 6.1.0

* Added OOB (out-of-bag) self-kernel to `forest.kernel()` via `oob = TRUE`.
  Computes kernel co-occurrence only from trees where both observations are OOB,
  eliminating response-contamination bias for unbiased degrees-of-freedom estimation.
  Returns `Kernel` (normalized), `N` (OOB count), and `C` (leaf-sharing count).

# RLT 6.0.2

* Improved regression distributed variable importance for multiple test samples.
* Fixed small issues in model validation, prediction, and variance-related code paths.

# RLT 6.0.1

* Added survival forests with logrank, sup-logrank, and Cox-gradient splitting.
* Added survival linear-combination splitting, observation weights, and survival confidence bands.

# RLT 6.0.0

* Rewrote the modeling backend in C++ with Rcpp/RcppArmadillo.
* Added deterministic parallel forests with reproducible random seeds across core counts.

# RLT 5.8.0

* Added `var.prob` for weighted variable sampling across model types.
* Added regression variable-importance variance estimation.

# RLT 5.7.0

* Added classification forest support with classification-specific linear-combination splitting.
* Added classification probability prediction and related categorical predictor handling.

# RLT 4.2.5

* Added regression linear-combination splitting and reinforcement learning splitting modes.

# RLT 4.1.3

* Improved reproducibility, model support, and speed.

# RLT 4.0.0

* Migrated the package to Rcpp and RcppArmadillo.

# RLT 3.2.3

* Fixed header compatibility issues.

# RLT 3.2.2

* Updated classification variable-weight handling.

# RLT 3.2.1

* Fixed Solaris compilation issues.

# RLT 3.2.0

* Fixed `omp.h` compilation issues.

# RLT 3.1.1

* Fixed a survival split crash.

# RLT 3.1.0

* Initial release of the RLT modeling interface.
