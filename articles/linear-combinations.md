# RLT with Linear Combinations

## Overview

Linear-combination splits allow each internal node to split on a
weighted combination of several variables rather than a single
predictor. This can help when the signal is carried by a direction such
as `X1 + X2` instead of one coordinate at a time.

The example below uses regression. The same idea can also be applied to
classification and survival forests by setting
`model = "classification"` or `model = "survival"` and choosing a
model-appropriate `linear.comb.method`.

## Regression Example

``` r

set.seed(3)

n <- 160
p <- 6
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("X", seq_len(p))

y <- 1 + X[, 1] + X[, 2] + 0.5 * X[, 3] + rnorm(n, sd = 0.5)

train_id <- 1:120
trainX <- X[train_id, ]
trainY <- y[train_id]
testX <- X[-train_id, ]
testY <- y[-train_id]
```

Fit a regression forest that considers three-variable linear
combinations at a split.

``` r

library(RLT)

fit_lc <- RLT(
  trainX, trainY,
  model = "regression",
  ntrees = 200,
  nmin = 5,
  mtry = 3,
  split.gen = "random",
  nsplit = 3,
  linear.comb = 3,
  linear.comb.method = "sir",
  ncores = 1,
  verbose = FALSE
)

print(fit_lc)
## ------------------------------------------
## RLT Regression Forest (Linear Combination)
## ------------------------------------------
##               (N, P) = (120, 6)
##           # of trees = 200
##         (mtry, nmin) = (3, 5)
##       split generate = Random, 3
## linear combination split = 3
##             sampling = 100% w/ replace
##           importance = none
##             OOB MSE = 0.4451 (R2 = 0.7977)
## ------------------------------------------
```

## Prediction

``` r

pred <- predict(fit_lc, testX, ncores = 1)
test_pred <- if (is.list(pred) && !is.null(pred$Prediction)) {
  pred$Prediction
} else {
  as.numeric(pred)
}

list(
  Test_MSE = round(mean((test_pred - testY)^2), 4),
  Linear_Combination_Size = fit_lc$parameters$linear.comb
)
## $Test_MSE
## [1] 0.7021
## 
## $Linear_Combination_Size
## [1] 3
```

## Other Model Types

For classification, use `model = "classification"` with methods such as
`"lda"`, `"naive"`, `"random"`, or `"logistic"`.

For survival, use `model = "survival"` with methods such as `"coxph"` or
`"naive"`.
