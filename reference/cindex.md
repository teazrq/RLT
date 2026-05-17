# C-index

Calculate c-index for survival data

## Usage

``` r
cindex(y, censor, pred)
```

## Arguments

- y:

  survival time

- censor:

  The censoring indicator if survival model is used

- pred:

  the predicted value for each subject

## Value

c-index

## Examples

``` r
# \donttest{
  set.seed(42)
  n <- 100
  x <- matrix(rnorm(n * 5), ncol = 5)
  y <- rexp(n, rate = exp(rowSums(x[, 1:2])))
  censor <- rbinom(n, 1, 0.7)
  fit <- RLT(x, y, censor = censor, model = "survival", ntrees = 100)
  # Use cumulative hazard at last timepoint as risk score
  risk <- fit$Prediction[, ncol(fit$Prediction)]
  cindex(y, censor, risk)
#> [1] 0.3863981
# }
```
