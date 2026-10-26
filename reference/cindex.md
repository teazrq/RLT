# C-index

Calculate concordance between survival outcomes and a risk score. Larger
risk scores indicate earlier failure.

## Usage

``` r
cindex(y, censor, pred)
```

## Arguments

- y:

  Observed survival times.

- censor:

  Event indicator: 1 for an observed event, 0 for right censoring.

- pred:

  One risk score per observation.

## Value

The concordance index, or `NaN` when there are no comparable pairs.

## Details

The example uses final OOB cumulative hazard as its risk score. The
survival forest's stored `Error` uses a different score: the sum of OOB
cumulative hazards over all failure-time grid points. The example
therefore need not equal `1 - fit$Error`.

## Examples

``` r
# \donttest{
  set.seed(42)
  n <- 100
  x <- matrix(rnorm(n * 5), ncol = 5)
  y <- rexp(n, rate = exp(rowSums(x[, 1:2])))
  censor <- rbinom(n, 1, 0.7)
  fit <- RLT(x, y, censor = censor, model = "survival", ntrees = 100)
  # Sum OOB hazard increments to obtain final cumulative hazard.
  risk <- rowSums(fit$Prediction)
  cindex(y, censor, risk)
#> [1] 0.6734323
# }
```
