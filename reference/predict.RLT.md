# Predict with an RLT forest

Predict the outcome (regression, classification or survival) using a
fitted RLT object

## Usage

``` r
# S3 method for class 'RLT'
predict(
  object,
  testx = NULL,
  var.est = FALSE,
  var.mode = NULL,
  keep.all = FALSE,
  ncores = 1,
  verbose = 0,
  band.grid.size = 0,
  ...
)
```

## Arguments

- object:

  A fitted RLT object

- testx:

  New observations with the training feature structure. If omitted,
  returns the stored OOB `object$Prediction` directly. For supplied
  data, labels are matched using the saved training dictionaries;
  numerical values for encoded features mean integer training codes.
  Legacy fits require their original dictionaries or refitting before
  accepting labels.

- var.est:

  Whether to estimate prediction uncertainty. Fit with an active
  `var.mode`, or satisfy the requirements for an explicit
  prediction-time mode described below. Regression returns prediction
  variance; classification returns variance for each class probability.
  Survival returns cumulative-hazard covariance over the evaluation
  times.

- var.mode:

  Variance estimation mode for prediction. Can be `"none"`, `"matched"`,
  `"IJ"`, or `"jack"`. If `NULL` (default), uses the mode from the
  fitted object. Only used when `var.est = TRUE`. `"matched"` requires
  the forest to be fitted with `var.mode = "matched"`. `"IJ"` and
  `"jack"` require the forest to have been fitted with resample tracking
  (automatically enabled when using IJ or jack variance).

- keep.all:

  whether to keep the prediction from all trees. Warning: this can
  occupy a large storage space, especially in survival model

- ncores:

  number of cores

- verbose:

  print additional information

- band.grid.size:

  A nonnegative integer specifying the requested maximum number of
  evaluation time points. Default is 0, which uses all unique failure
  time points. If a positive integer is provided, a subset of time
  points will be selected using quantiles from the fifth percentile
  through the last time point. Quantile selection may return fewer
  distinct points. A request of 1 selects the fifth-percentile point.

- ...:

  ...

## Value

With `testx = NULL`, returns `object$Prediction` directly, without a
prediction list. Otherwise returns an `RLT` prediction list.

For regression and classification:

- Prediction:

  Regression: an \\N \times 1\\ numeric matrix. Classification: a factor
  of predicted labels using the training response levels. Here
  `N = nrow(testx)`.

- Prob:

  Classification only: an \\N \times K\\ matrix of class probabilities,
  with columns in the order of the fitted object's `ylabels`, where `K`
  is the number of classes.

- Variance:

  When variance estimation is requested, regression prediction variances
  or an \\N \times K\\ matrix of class-probability variances. Negative
  variance estimates are returned as `NA`.

**For Survival Forests**

`Hazard`, `CHF` and `Survival` each have one row per test observation
and one column per evaluation time. There is no `Prediction` component
in this list.

- Hazard:

  predicted pointwise hazard increments at the selected times

- CHF:

  predicted cumulative hazard, accumulated on the complete training grid
  before selecting evaluation times. On a reduced grid, accumulating
  only `Hazard` does not in general reproduce `CHF`.

- Survival:

  ordinary forests return `exp(-CHF)`. Linear combination forests return
  the mean of the tree product-limit survival curves. The
  [`get.surv.band()`](https://teazrq.github.io/RLT/reference/get.surv.band.md)
  interface uses `exp(-CHF)` as its reference curve.

- AllHazard:

  if `keep.all = TRUE`, the pointwise hazard increments for each tree,
  selected time, and observation, in that array order

- Cov:

  when variance estimation is requested, an array with one \\T\times T\\
  cumulative-hazard covariance matrix per test subject, where
  `T = length(timepoints)`. Its dimensions are time, time, and test
  observation, in that order.

- timepoints:

  ordered evaluation times selected from the training failure grid

## Examples

``` r
# \donttest{
  set.seed(42)
  x <- matrix(rnorm(300 * 5), ncol = 5)
  y <- rowSums(x[, 1:2]) + rnorm(300)
  fit <- RLT(x, y, ntrees = 100)
  pred <- predict(fit, testx = x[1:5, ])
  print(pred$Prediction)
#>             [,1]
#> [1,]  1.22292224
#> [2,]  0.28236394
#> [3,] -0.09172784
#> [4,]  1.41876752
#> [5,] -0.27677253
# }
```
