# Random Seed and Reproducibility - Tutorial (RLT)

## Overview

Use `RLT(seed = ...)` to repeat a forest fit on fixed data. Use base R’s
[`set.seed()`](https://rdrr.io/r/base/Random.html) to repeat random
steps in the wider analysis, such as data generation or a random
train/test split. Keeping these two sources of randomness separate helps
you check whether a change comes from fitting or from the data.

Follow [Get
Started](https://teazrq.github.io/RLT/articles/get-started.md) to
install RLT, then load the package:

``` r

library(RLT)
```

## Generate the data once

We use the same numeric and factor predictors, continuous response, and
held-out split as in the [Regression
Tutorial](https://teazrq.github.io/RLT/articles/regression-tutorial.md).
The response depends on predictors 2 through 6 and an additional
categorical effect for predictor 6, plus independent Gaussian noise.
`set.seed(1)` repeats this data generation; the data remain fixed for
all fits below.

``` r

set.seed(1)

trainn <- 80
testn <- 20
n <- trainn + testn
p <- 10
X1 <- matrix(rnorm(n * (p/2)), n, p/2)
X2 <- matrix(as.integer(runif(n * (p/2)) * 3), n, p/2)
X_numeric <- data.frame(X1, X2)
y <- 1 + rowSums(X_numeric[, 2:6]) +
  2 * (X_numeric[, p/2 + 1] %in% c(1, 2)) + rnorm(n)

X <- X_numeric
X[, (p/2 + 1):p] <- lapply(X[, (p/2 + 1):p], as.factor)

trainX <- X[1:trainn, ]
trainY <- y[1:trainn]
testX <- X[(trainn + 1):(trainn + testn), ]
testY <- y[(trainn + 1):(trainn + testn)]
```

## Repeat a fit with the same seed

Both calls use the same data, forest settings, and explicit fitting
seed. No data are regenerated between them.

``` r

fit_A <- RLT(
  trainX, trainY, model = "regression",
  ntrees = 200, mtry = p/2, nsplit = 0,
  seed = 1, ncores = 1, verbose = FALSE
)
fit_B <- RLT(
  trainX, trainY, model = "regression",
  ntrees = 200, mtry = p/2, nsplit = 0,
  seed = 1, ncores = 1, verbose = FALSE
)

pred_A <- predict(fit_A, testX)
pred_B <- predict(fit_B, testX)

list(
  Same_OOB_Predictions = isTRUE(all.equal(fit_A$Prediction, fit_B$Prediction)),
  Same_Test_Predictions = isTRUE(all.equal(pred_A$Prediction, pred_B$Prediction)),
  OOB_MSE = mean((fit_A$Prediction - trainY)^2),
  Test_MSE = mean((pred_A$Prediction - testY)^2)
)
## $Same_OOB_Predictions
## [1] TRUE
## 
## $Same_Test_Predictions
## [1] TRUE
## 
## $OOB_MSE
## [1] 3.501257
## 
## $Test_MSE
## [1] 4.124253
```

The two prediction comparisons should be `TRUE` in the same software
environment. OOB MSE uses each training observation’s out-of-bag
prediction, computed only from trees whose resample omitted it. Test MSE
uses the 20 held-out observations.

## Change only the fitting seed

The next fit changes `seed` while leaving the data and other settings
fixed. Resampling and split-candidate selection can therefore change, so
the resulting predictions may differ.

``` r

fit_C <- RLT(
  trainX, trainY, model = "regression",
  ntrees = 200, mtry = p/2, nsplit = 0,
  seed = 2, ncores = 1, verbose = FALSE
)
pred_C <- predict(fit_C, testX)

list(
  Same_Test_Predictions_As_A = isTRUE(all.equal(pred_C$Prediction, pred_A$Prediction)),
  OOB_MSE = mean((fit_C$Prediction - trainY)^2),
  Test_MSE = mean((pred_C$Prediction - testY)^2)
)
## $Same_Test_Predictions_As_A
## [1] FALSE
## 
## $OOB_MSE
## [1] 3.430922
## 
## $Test_MSE
## [1] 3.955942
```

Any difference here comes from fitting randomness on the same
observations.

## Repeat the whole analysis

Place [`set.seed()`](https://rdrr.io/r/base/Random.html) before the
first random step when you want to repeat data generation, a random data
split, and subsequent analysis together. If
[`RLT()`](https://teazrq.github.io/RLT/reference/RLT.md) has no explicit
`seed`, it draws one from R’s random number generator, so the preceding
random steps also determine that seed. Supplying `RLT(seed = ...)` makes
the fitting seed explicit even if earlier random steps change. The same
distinction applies to classification and survival models.

Record the data, code, package version, and source revision as well as
the seed. Fixed seeds do not establish identical numerical results
across platforms, compilers, or package revisions. See [matching the
documentation
revision](https://teazrq.github.io/RLT/articles/get-started.html#match-the-documentation-revision)
for source installation details.
