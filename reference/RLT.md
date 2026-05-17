# Reinforcement Learning Trees

          Fit models for regression, classification and survival

analysis using reinforced splitting rules. The model fits regular random
forest models by default unless the parameter `reinforcement` is set to
`"TRUE"`. Using `reinforcement = TRUE` activates embedded model for
splitting variable selection and allows linear combination split. To
specify parameters of embedded models, see definition of `param.control`
for details.

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

  A `matrix` or `data.frame` of features. If `x` is a data.frame, then
  all factors are treated as categorical variables, which will go
  through an exhaustive search of splitting criteria.

- y:

  Response variable. a `numeric`/`factor` vector.

- censor:

  Censoring indicator if survival model is used.

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
  estimator that accounts for the observation weights.

- var.prob:

  Variable probabilities for split variable selection. A numeric vector
  of length `p` (number of predictors) with non-negative weights. When
  supplied, `mtry` variables are sampled *without replacement* with
  probabilities proportional to these weights at each internal node.
  This effectively up-weights or down-weights individual predictors
  during tree construction. Works for all models (regression,
  classification, survival). The vector does not need to sum to 1; it is
  internally normalized. If `NULL` (default), uniform sampling is used.

- importance:

  Whether to calculate variable importance measures. When set to
  `"TRUE"` (or `"permute"`), the calculation follows Breiman's original
  permutation strategy. If set to `"distribute"`, then it sends the oob
  data to both child nodes with weights proportional to their sample
  sizes. Hence the final prediction is a weighted average of all
  possible terminal nodes that a perturbed observation could fall into.
  This feature is currently only available in regression and
  classification models.

- reinforcement:

  Should reinforcement splitting rule be used. Default is `"FALSE"`,
  i.e., regular random forests with marginal search of splitting
  variable. When it is activated, an embedded model is fitted to find
  the best splitting variable or a linear combination of them, if
  `linear.comb` \$\> 1\$. They can also be specified in `param.control`.

- linear.comb:

  Number of variables to combine in each linear combination split.
  Default is 1 (standard axis-aligned splits). See also
  `linear.comb.method` and `param.control`.

- linear.comb.method:

  Method for constructing linear combinations: `"default"`, `"coxph"`
  (Cox PH loading, survival only), or `"naive"` (covariance-based
  loading). See `param.control`.

- split.rule:

  Splitting criterion. Default `"default"` selects the standard rule for
  each model. For survival: `"logrank"`, `"suplogrank"`, or `"coxgrad"`.
  See `param.control`.

- var.mode:

  Variance estimation mode. Default is `"none"` (no variance
  estimation). Set to `"matched"` or `TRUE` to use matched-sample
  U-statistic decomposition for prediction variance and variable
  importance variance. When active, several resampling parameters are
  automatically adjusted. Equivalent to setting
  `param.control = list(var.mode = "matched")`. See `param.control` for
  full details.

- param.control:

  A list of additional parameters. This can be used to specify other
  features in a random forest or set embedded model parameters for
  reinforcement splitting rules. Using `reinforcement = TRUE` will
  automatically generate some default tuning for the embedded model.
  **Reinforcement is available for regression, classification, and
  survival models.** They are not necessarily optimized.

  - `embed.ntrees`: number of trees in the embedded model. Default: 50.

  - `embed.mtry`: proportion of variables for embedded splits. Default:
    0.5.

  - `embed.nmin`: terminal node size for embedded model. Default: 5.

  - `embed.nsplit`: number of random cutting points. Default: 3.

  - `embed.resample.replace`: whether to sample with replacement.
    Default: TRUE.

  - `embed.resample.prob`: proportion of samples (of the internal node)
    in the embedded model. Default: 0.9.

  - `embed.mute`: variables to mute per split. If \>= 1: exact count; if
    \< 1: proportion. Default: 0 (no muting).

  - `embed.protect`: number of top variables to protect from muting.
    Default: ceiling(log(n)).

  - `embed.threshold`: threshold, as a fraction of the best VI, for
    being included in the protected set at an internal node. Default:
    0.25.

  - `linear.comb`: number of variables to use in linear combination
    splits. Requires `reinforcement = TRUE`. Default: 1 (no linear
    combination).

  - `linear.comb.method`: method for constructing linear combination
    splits. Regression: `"naive"` (1), `"lm"` (2), `"pca"` (3), `"sir"`
    (4, default). Classification: `"lda"` (1, default), `"naive"` (2),
    `"random"` (3), `"logistic"` (4).

  - `time.grid.size`: number of unique time points for survival
    estimation (default 0 = all). See `time.grid.size` argument for
    details.

                         See \code{linear.comb} and \code{linear.comb.method} under
                         \code{param.control} documentation above.

                         \code{split.rule} specifies the splitting criterion for each model type.
                         \itemize{
                         \item \strong{Regression}: \code{"var"} (variance reduction, default and only option)
                         \item \strong{Classification}: \code{"gini"} (Gini index, default and only option)
                         \item \strong{Survival}: \code{"logrank"} (default), \code{"suplogrank"}, \code{"coxgrad"}
                         }
                         Internally mapped to integers: var=1, gini=1, logrank=1, suplogrank=2, coxgrad=3.

                         \code{resample.track} indicates whether to keep track
                         of which observations are used in each tree. This is
                         required for variance estimation (via \code{var.mode}).

                         \code{var.mode} specifies the variance estimation method
                         to prepare during model fitting. Currently available methods:
                         \itemize{
                         \item \code{"none"} (default): No variance estimation.
                         \item \code{"matched"}: Uses matched-sample U-statistic
                         decomposition (Xu, Zhu & Shao, 2023) for prediction
                         variance and variable importance variance. Also used for
                         confidence band in survival models (Formentini, Liang & Zhu, 2023).
                         }
                         Specifying \code{var.mode = TRUE} is equivalent to
                         \code{var.mode = "matched"}.
                         When \code{var.mode} is not \code{"none"}, the following
                         parameters are automatically adjusted if not already set:
                         \itemize{
                         \item \code{resample.preset} is constructed automatically
                         \item \code{resample.replace} is set to \code{FALSE}
                         \item \code{resample.prob} is set to 0.5
                         \item \code{resample.track} is set to \code{TRUE}
                         \item \code{importance} is set to \code{"distribute"}
                         }

                         It is recommended to use a very large \code{ntrees},
                         e.g, 10000 or larger. For \code{resample.prob} greater
                         than 0.5, one should consider the bootstrap
                         approach in Xu, Zhu & Shao (2023).

                         \\code{time.grid.size} specifies the number of unique
                         time points used for survival estimation. By default
                         (0), all observed failure times are used. Setting a
                         smaller number (e.g., 50) can speed up computation
                         for large datasets. The time points are selected at
                         evenly spaced quantiles of the observed failure times,
                         always including the minimum and maximum failure times.

- ncores:

  Number of CPU logical cores. Default is 0 (using all available cores).

- verbose:

  Whether info should be printed.

- seed:

  Random seed number to replicate a previously fitted forest.
  Internally, the `xoshiro256++` generator is used. If not specified, a
  seed will be generated automatically and recorded.

- ...:

  Additional arguments.

## Value

A `RLT` fitted object, constructed as a list consisting

- FittedForest:

  Fitted tree structures

- VarImp:

  Variable importance measures, if `importance = TRUE`

- Prediction:

  Out-of-bag prediction

- Error:

  Out-of-bag prediction error, adaptive to the model type

- ObsTrack:

  Provided if `resample.track = TRUE`, `var.mode != "none"`, or if
  `resample.preset` was supplied. This is an `n` \\\times\\ `ntrees`
  matrix that has the same meaning as `resample.preset`.

For classification forests, these items are further provided or will
replace the regression version

- NClass:

  The number of classes

- Prob:

  Out-of-bag predicted probability

For survival forests, these items are further provided or will replace
the regression version

- timepoints:

  ordered observed failure times

- NFail:

  The number of observed failure times

- Prediction:

  Out-of-bag prediction of hazard function

## References

- Zhu, R., Zeng, D., & Kosorok, M. R. (2015) "Reinforcement Learning
  Trees." Journal of the American Statistical Association. 110(512),
  1770-1784.

- Xu, T., Zhu, R., & Shao, X. (2023) "On Variance Estimation of Random
  Forests with Infinite-Order U-statistics." arXiv preprint
  arXiv:2202.09008.

- Formentini, S. E., Wei L., & Zhu, R. (2022) "Confidence Band
  Estimation for Survival Random Forests." arXiv preprint
  arXiv:2204.12038.

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
