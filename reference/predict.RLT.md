# prediction using RLT

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

  The testing samples, must have the same structure as the training
  samples

- var.est:

  Whether to estimate the variance of each testing data. The original
  forest must be fitted with `var.mode != "none"`. For survival forests,
  calculates the covariance matrix over all observed time points and
  calculates critical value for the confidence band.

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

  An integer specifying the number of time points for confidence band
  calculation. Default is 0, which uses all unique failure time points.
  If a positive integer is provided, a subset of time points will be
  selected using quantiles, skipping the earliest 5% of time points to
  improve stability.

- ...:

  ...

## Value

A `RLT` prediction object, constructed as a list consisting

- Prediction:

  Prediction

- Variance:

  if `var.est = TRUE` and the fitted object is `var.mode != "none"`

**For Survival Forests**

- hazard:

  predicted hazard functions

- CumHazard:

  predicted cumulative hazard function

- Survival:

  predicted survival function

- Allhazard:

  if `keep.all = TRUE`, the predicted hazard function for each
  observation and each tree

- AllCHF:

  if `keep.all = TRUE`, the predicted cumulative hazard function for
  each observation and each tree

- Cov:

  if `var.est = TRUE` and the fitted object is `var.mode != "none"`. For
  each test subject, a matrix of size NFail\\\times\\NFail where NFail
  is the number of observed failure times in the training data

- Var:

  if `var.est = TRUE` and the fitted object is `var.mode != "none"`.
  Marginal variance for each subject

- timepoints:

  ordered observed failure times from the training data

- MarginalVar:

  if `var.est = TRUE` and the fitted object is `var.mode != "none"`.
  Marginal variance for each subject from the Cov matrix projected to
  the nearest positive definite matrix

- MarginalVarSmooth:

  if `var.est = TRUE` and the fitted object is `var.mode != "none"`.
  Marginal variance for each subject from the Cov matrix projected to
  the nearest positive definite matrix and then smoothed using Gaussian
  kernel smoothing

- CVproj:

  if `var.est = TRUE` and the fitted object is `var.mode != "none"`.
  Critical values to calculate confidence bands around cumulative hazard
  predictions at several confidence levels. Calculated using
  `MarginalVar`

- CVprojSmooth:

  if `var.est = TRUE` and the fitted object is `var.mode != "none"`.
  Critical values to calculate confidence bands around cumulative hazard
  predictions at several confidence levels. Calculated using
  `MarginalVarSmooth`

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
