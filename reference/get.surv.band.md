# Construct survival confidence bands

Construct two-sided survival limits by transforming a band around
predicted cumulative hazard. The reference curve is `exp(-x$CHF)`.

## Usage

``` r
get.surv.band(
  x,
  i = 0,
  alpha = 0.05,
  approach = "naive",
  nsim = 5000,
  k_rank = 10,
  k_mode = c("fixed", "proportion"),
  k_prop = 0.99,
  ...
)
```

## Arguments

- x:

  An RLT survival prediction object containing `CHF`, `Cov` and
  `timepoints`, obtained with `var.est = TRUE`. See
  [`predict.RLT`](https://teazrq.github.io/RLT/reference/predict.RLT.md)
  for the fitting and tracking requirements.

- i:

  Observation number in the prediction. The default `i = 0` calculates a
  separate band for every observation.

- alpha:

  Nominal noncoverage level for each subject's simultaneous band;
  `alpha = 0.05` requests a nominal 95 percent band.

- approach:

  Confidence band approach:

  - naive: marginal scales from the supplied covariance diagonal and a
    Monte Carlo critical value using the full covariance.

  - smoothed: marginal scales from a rank-K approximation to a
    covariance smoothed with generalized additive models (GAMs), plus an
    eigenvalue-ratio weighted residual correction. With one or two time
    points, uses the naive calculation with a warning.

- nsim:

  Number of Gaussian simulations for estimating the critical value.
  Increasing it improves Monte Carlo precision at the cost of runtime.

- k_rank:

  Nonnegative integer rank of the covariance approximation used to
  construct smoothed marginal scales. The effective rank is capped by
  the grid dimension. It also guides the GAM basis size, whose
  feasibility is bounded separately.

- k_mode:

  Rank selection mode: "fixed" (use k_rank) or "proportion" (auto-select
  by eigenvalue cumulative ratio).

- k_prop:

  Proportion threshold (0,1\] for cumulative eigenvalue ratio when
  k_mode = "proportion".

- ...:

  Further arguments (currently not used).

## Value

A list of class `c("RLT", "band", "surv")` with one named `Subject<i>`
entry per selected observation and a `timepoints` entry containing the
evaluation grid. Each subject entry contains `lower`, `upper`, and
`diagnostics`. With `i = 0`, all observations are included; otherwise
only the requested subject entry is included.

## Details

For linear combination forests, the returned `Survival` is the mean of
tree product-limit curves, which can differ from this reference curve.
These calculations do not directly estimate the sampling variation of
that returned mean product-limit estimator. Limits are evaluated on
`x$timepoints`; reducing the grid does not establish simultaneous
coverage between omitted time points. Both approaches simulate using the
supplied raw `Cov`; smoothing changes the marginal standardizing scales.
The intended target is the conditional survival curve at a subject's
predictor values. The nominal simultaneous coverage target is across the
returned evaluation times, separately for each subject. It is not joint
coverage across subjects or a prediction interval for an individual's
event time.

## Examples

``` r
# \donttest{
  set.seed(42)
  n <- 100
  x <- matrix(rnorm(n * 5), ncol = 5)
  y <- rexp(n, rate = exp(rowSums(x[, 1:2])))
  censor <- rbinom(n, 1, 0.7)
  fit <- RLT(x, y, censor = censor, model = "survival",
             ntrees = 200, var.mode = "matched")
#> var.mode ('matched') adjusted the following settings:
#>   resample.replace: TRUE -> FALSE
#>   resample.prob: 1 -> 0.5
#>   importance: FALSE -> distribute
  pred <- predict(fit, testx = x[1:3, ], var.est = TRUE)
  band <- get.surv.band(pred, i = 1, alpha = 0.05)
# }
```
