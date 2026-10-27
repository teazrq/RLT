# Reinforcement Learning Trees

Fit random forest models for regression, classification and survival
analysis. By default, splits use one variable at a time. Set
`reinforcement = TRUE` to use embedded-model guidance for
single-variable splits, or `linear.comb > 1` for linear combination
splits. Linear combination splits do not require `reinforcement = TRUE`;
regression and classification combination splits still use embedded
screening at sufficiently large nodes. See `param.control` for details.

## Usage

``` r
RLT(
  x,
  y,
  censor = NULL,
  model = NULL,
  ntrees = if (reinforcement) 100 else 500,
  mtry = max(1, as.integer(ncol(x)/2)),
  nmin = 5,
  alpha = 0,
  nsplit = 0,
  resample.replace = TRUE,
  resample.prob = if (resample.replace) 1 else 0.8,
  resample.preset = NULL,
  obs.w = NULL,
  var.prob = NULL,
  importance = FALSE,
  reinforcement = FALSE,
  linear.comb = 1,
  linear.comb.method = "default",
  split.rule = "default",
  var.mode = "none",
  param.control = list(),
  ncores = 0,
  verbose = 0,
  seed = NULL,
  ...
)
```

## Arguments

- x:

  A numeric `matrix` or a `data.frame` of features. Unordered factors in
  a data frame use categorical splits. Ordered factors and character
  columns retain their numerical rank-code treatment. Training label
  dictionaries are saved for all three types, so prediction and kernels
  match labels to the training codes. Numeric inputs for these columns
  are interpreted as integer training codes in
  `1:length(training_labels)`, not as labels. Character matrices are not
  supported.

- y:

  Response vector: numeric for regression or observed survival times,
  and a factor for classification.

- censor:

  Survival event indicator: 1 for an observed event and 0 for a
  right-censored observation.

- model:

  The model type: `"regression"`, `"classification"`, or `"survival"`.
  Quantile forest is not yet implemented.

- ntrees:

  Number of trees, `ntrees = 100` if reinforcement is used and
  `ntrees = 500` otherwise.

- mtry:

  Number of randomly selected variables used at each internal node.
  Default: max(1, floor(p/2)).

- nmin:

  Terminal node size. Splitting will stop when the internal node size is
  less equal to `nmin`. Default: 5.

- alpha:

  Minimum proportion of samples (of the parent node) enforced in each
  child node. Default: 0 (no constraint). Clamped to the range 0 to 0.5.

- nsplit:

  Number of random cutting points to compare for each variable at an
  internal node. Default: 0 (use all unique values, i.e., best split).
  When nsplit \> 0, random cutting points are generated.

- resample.replace:

  Whether the in-bag samples are obtained with replacement.

- resample.prob:

  Proportion of in-bag samples.

- resample.preset:

  A pre-specified matrix for in-bag data indicator/count matrix. It must
  be an \\n \times\\ `ntrees` matrix with integer entries. Positive
  number indicates the number of copies of that observation (row) in the
  corresponding tree (column); zero indicates out-of-bag; negative
  values indicates not being used in either. Extremely large counts
  should be avoided. The sum of each column should not exceed \\n\\.

- obs.w:

  Observation weights. The weights will be used for calculating the
  splitting scores, such as a weighted variance reduction or weighted
  gini index. But they will not be used for sampling observations. In
  that case, one can pre-specify `resample.preset` instead for balanced
  sampling, etc. For survival analysis, observation weights are
  supported in the `"logrank"`, `"suplogrank"`, and `"coxgrad"`
  splitting rules. Weighted logrank and suplogrank use a variance
  estimator that accounts for the observation weights. Weights must be
  finite and nonnegative with positive total mass; individual zero
  weights are allowed. A sampled outer or embedded tree with zero total
  observation weight produces a catchable error.

- var.prob:

  Variable probabilities for split variable selection. A numeric vector
  of length `p` (number of predictors) with finite nonnegative weights
  and positive total mass. The vector is normalized internally and
  guides sampling of split candidates. In particular, requesting the
  whole candidate pool returns all variables, and some
  regression/classification linear combination screening paths sample
  uniformly. Zero-probability exclusion is therefore not guaranteed
  across these paths. If `NULL` (default), uniform sampling is used.

- importance:

  Whether to calculate variable importance. Use `TRUE` or `"permute"` to
  measure the deterioration in out-of-bag (OOB) performance after
  permuting a predictor. With `"distribute"`, an OOB observation follows
  both children whenever a split uses the assessed predictor. Routing
  probabilities are proportional to child node weights, or sample counts
  for unweighted data. Regression and classification average
  terminal-node losses; survival evaluates concordance using averaged
  terminal-node risk scores. Both strategies are available for
  regression, classification, and survival. Survival importance uses OOB
  concordance and is unavailable when no usable OOB comparisons exist.

- reinforcement:

  Whether to use embedded-model guidance for single-variable splits.
  Default: `FALSE`. Linear combination splits are selected separately by
  `linear.comb`; see `param.control` for their model-specific screening
  behavior.

- linear.comb:

  Maximum number of variables in each linear combination split. Default:
  1 (single-variable splits). Values greater than 1 do not require
  `reinforcement = TRUE`. See `linear.comb.method` and `param.control`.

- linear.comb.method:

  Method for constructing linear combinations. `"default"` selects
  `"sir"` for regression, `"lda"` for classification, and `"coxph"` for
  survival. See `param.control` for all methods.

- split.rule:

  Splitting criterion. Default `"default"` selects the standard rule for
  each model. For survival: `"logrank"`, `"suplogrank"`, or `"coxgrad"`.
  See `param.control`.

- var.mode:

  Variance estimation mode: `"none"` (default), `"matched"`, `"IJ"`, or
  `"jack"`. `TRUE` is equivalent to `"matched"`. Matched sampling
  supports prediction and importance variance; IJ and jackknife modes
  support prediction variance. Active modes adjust resampling settings
  and retain observation tracking. See `param.control`.

- param.control:

  A list of additional forest and embedded-model settings. Reinforcement
  is available for regression, classification, and survival.
  Embedded-model defaults are provided automatically but may need
  tuning.

  Embedded-model settings:

  - `embed.ntrees`: number of embedded trees. Default: 50.

  - `embed.mtry`: candidate-variable proportion when less than 1, or
    count when at least 1. Default: 0.5.

  - `embed.nmin`: embedded terminal node size. Default: 5.

  - `embed.nsplit`: random cutting points; 0 uses the best split.
    Default: 3.

  - `embed.resample.replace`: sample with replacement. Default: TRUE.

  - `embed.resample.prob`: sample proportion within the current node.
    Default: 0.9.

  - `embed.mute`: variables to mute per split, as a count when at least
    1 or a proportion when less than 1. Default: 0 (no muting).

  - `embed.protect`: number of top variables protected from muting.
    Default: `ceiling(log(n))`.

  - `embed.threshold`: threshold relative to the best importance for
    inclusion in the protected set. Default: 0.25.

  Linear combination settings:

  - `linear.comb`: maximum variables per combination. Default: 1. Values
    greater than 1 select combination splits independently of
    `reinforcement`. Regression and classification use embedded
    importance screening when the node has at least `3 * embed.nmin`
    observations and direct screening at smaller nodes. Survival
    combination splits screen `mtry` candidates without an embedded
    forest.

  - `linear.comb.method`: regression accepts `"naive"`, `"lm"`, `"pca"`,
    and `"sir"` (default); classification accepts `"lda"` (default),
    `"naive"`, `"random"`, and `"logistic"`; survival accepts `"coxph"`
    (default) and `"naive"`.

  `split.rule` selects the splitting criterion:

  - Regression: `"var"` (variance reduction).

  - Classification: `"gini"` (Gini index).

  - Survival: `"logrank"` (default), `"suplogrank"`, or `"coxgrad"`.

  `resample.track` retains the in-bag count for each observation and
  tree. Default: FALSE. All active variance modes enable it
  automatically.

  `var.mode` prepares variance estimation during fitting:

  - `"none"` (default): no variance estimation.

  - `"matched"` (or `TRUE`): matched-sample U-statistic decomposition
    for prediction and variable importance variance. Uses paired
    half-samples without replacement, sets `resample.prob = 0.5`, and
    generates `resample.preset`, replacing any supplied preset. An odd
    `ntrees` is reduced to the preceding even number; at least 2 are
    required.

  - `"IJ"`: infinitesimal jackknife prediction variance.

  - `"jack"`: jackknife-after-bootstrap prediction variance.

  IJ and jackknife modes set `resample.replace = TRUE`, retain the
  requested `resample.prob`, and generate bootstrap counts instead of
  using a supplied `resample.preset`. They do not estimate variable
  importance variance. All active variance modes enable
  `importance = "distribute"` when importance was disabled; an
  explicitly requested importance method is retained. Stable variance
  estimates generally require many trees. IJ and jackknife modes
  recommend at least 2000 trees.

  `time.grid.size` is 0 by default and retains all distinct observed
  failure times for survival fitting. A positive integer must be at
  least the number of distinct failure times; it does not add times to
  the training grid. Requests that would reduce the training grid are
  rejected. Use `band.grid.size` in
  [`predict()`](https://rdrr.io/r/stats/predict.html) to reduce the
  evaluation grid instead.

- ncores:

  Number of CPU logical cores. Default is 0 (using all available cores
  when compiled with OpenMP). Builds without OpenMP run serially.

- verbose:

  Whether info should be printed.

- seed:

  Random seed number to replicate a previously fitted forest.
  Internally, the `xoshiro256++` generator is used. If not specified, a
  seed will be generated automatically and recorded.

- ...:

  Additional arguments.

## Value

A fitted object of class `RLT`. OOB predictions for each training
observation use only trees that did not sample that observation.

- Prediction:

  OOB numeric predictions for regression, factor labels for
  classification, or a matrix of hazard increments for survival.
  Survival rows correspond to observations and columns to `timepoints`.

- Error:

  OOB mean squared error for regression, misclassification rate for
  classification, or \\1-C\\ for survival. The survival concordance
  score uses the sum of the OOB cumulative hazards over the failure-time
  grid, not just the cumulative hazard at the final time. Lower error is
  better.

- VarImp:

  Variable importance when requested. Use
  [`importance`](https://teazrq.github.io/RLT/reference/importance.md)
  to obtain a named summary.

- FittedForest:

  Fitted tree structures.

- xlevels:

  Training label dictionaries for factor, ordered-factor, and character
  data-frame columns, with `NULL` entries for numerical features.

- ObsTrack:

  Provided if `resample.track = TRUE`, `var.mode != "none"`, or if
  `resample.preset` was supplied. This is an `n` \\\times\\ `ntrees`
  matrix that has the same meaning as `resample.preset`.

Classification forests also provide:

- nclass:

  Number of response classes.

- Prob:

  OOB class probabilities, with one row per observation and one column
  per class in the order given by `ylabels`.

Survival forests also provide:

- timepoints:

  ordered observed failure times

- NFail:

  The number of observed failure times

## References

- Zhu, R., Zeng, D., & Kosorok, M. R. (2015) "Reinforcement Learning
  Trees." Journal of the American Statistical Association. 110(512),
  1770-1784.

- Xu, T., Zhu, R., & Shao, X. (2024) "On variance estimation of random
  forests with Infinite-order U-statistics." Electronic Journal of
  Statistics. 18(1), 2135-2207.
  [doi:10.1214/24-EJS2247](https://doi.org/10.1214/24-EJS2247) .

- Formentini, S. E., Liang, W., & Zhu, R. (2022) "Confidence Band
  Estimation for Survival Random Forests." arXiv preprint
  arXiv:2204.12038. <https://arxiv.org/abs/2204.12038>.

## Examples

``` r
# \donttest{
  set.seed(42)
  x <- matrix(rnorm(300 * 5), ncol = 5)
  y <- rowSums(x[, 1:2]) + rnorm(300)
  fit <- RLT(x, y, ntrees = 100)
  print(fit)
#> -----------------------------------------
#> RLT Regression Forest
#> -----------------------------------------
#>               (N, P) = (300, 5)
#>           # of trees = 100
#>         (mtry, nmin) = (2, 5)
#>       split generate = Best
#>             sampling = 100% w/ replace
#>           importance = none
#>             OOB MSE = 1.385 (R2 = 0.5361)
#> -----------------------------------------
# }
```
