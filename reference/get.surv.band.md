# get.surv.band

Calculate the survival function (two-sided) confidence band from a RLT
survival prediction.

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

  A RLT prediction object. Must be from a forest with var.mode !=
  "none".

- i:

  Observation number in the prediction. Default to calculate all (i =
  0).

- alpha:

  alpha level for interval (alpha/2, 1 - alpha/2).

- approach:

  Confidence band approach:

  - naive: marsd = sqrt(diag(Cov)), MC band using full covariance.

  - smoothed: GAM-smoothed rank-K covariance + eigenvalue-ratio weighted
    residual correction.

- nsim:

  Number of simulations for estimating the Monte Carlo critical value.

- k_rank:

  Rank truncation K used for the smooth low-rank covariance AND GAM
  basis size.

- k_mode:

  Rank selection mode: "fixed" (use k_rank) or "proportion" (auto-select
  by eigenvalue cumulative ratio).

- k_prop:

  Proportion threshold (0,1\] for cumulative eigenvalue ratio when
  k_mode = "proportion".

- ...:

  Further arguments (currently not used).

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
