# RLT Regression Tutorial

## Overview

Use a regression forest to predict a continuous response from numeric
and categorical predictors. We simulate a known signal, fit on 80
observations, and evaluate predictions on 20 held-out observations.

## Prerequisites

Follow [Get
Started](https://teazrq.github.io/RLT/articles/get-started.md) to
install RLT, then load the package:

``` r

library(RLT)
```

## Data

The first five predictors are numeric and the last five are factors.
Factor columns tell RLT to use categorical splits. The numeric response
combines predictors 2 through 6, an additional effect for categories 1
and 2 of predictor 6, and independent Gaussian noise.

``` r

set.seed(1)

trainn <- 80
testn  <- 20
n <- trainn + testn
p <- 10

# Continuous + categorical predictors (last half as factors)
X1 <- matrix(rnorm(n * (p/2)), n, p/2)
X2 <- matrix(as.integer(runif(n * (p/2)) * 3), n, p/2)  # integers 0,1,2

X_numeric <- data.frame(X1, X2)

# Continuous outcome with a simple signal + noise
y <- 1 + rowSums(X_numeric[, 2:6]) + 2 * (X_numeric[, p/2 + 1] %in% c(1, 2)) + rnorm(n)

X <- X_numeric
X[, (p/2 + 1):p] <- lapply(X[, (p/2 + 1):p], as.factor)

# Train / test split
trainX <- X[1:trainn, ]
trainY <- y[1:trainn]
testX  <- X[(trainn + 1):(trainn + testn), ]
testY  <- y[(trainn + 1):(trainn + testn)]
```

## Fit

We use 200 trees and sample half of the predictors as split candidates
at each node with `mtry = p/2`. Setting `nsplit = 0` searches all
available cutpoints for numeric candidates. The explicit `seed` controls
fitting randomness; see [Random Seed and
Reproducibility](https://teazrq.github.io/RLT/articles/feature-seed.md)
for the distinction from
[`set.seed()`](https://rdrr.io/r/base/Random.html) used for data
generation above.

``` r

fit <- RLT(
  trainX, trainY, model = "regression",
  ntrees = 200, mtry = p/2, nsplit = 0,
  seed = 1, ncores = 1, verbose = FALSE
)
```

## Predict

``` r

pred <- predict(fit, testX)
head(data.frame(Observed = testY, Predicted = pred$Prediction))
##    Observed  Predicted
## 1  3.027775  3.7546121
## 2  6.341096  3.6998381
## 3  5.114996  4.2049286
## 4 -1.929980 -0.1486038
## 5  8.651209  5.6871107
## 6  7.395645  4.1123466
```

`pred$Prediction` contains one numeric prediction per held-out row.
These values estimate the conditional mean response at the supplied
predictor values.

## Evaluate

The fitted object’s `Prediction` component contains out-of-bag (OOB)
predictions. Each training observation is predicted using only trees
whose resample omitted it. The test predictions use the fitted forest on
observations reserved before fitting.

``` r

list(
  OOB_MSE = round(mean((fit$Prediction - trainY)^2), 4),
  Test_MSE = round(mean((pred$Prediction - testY)^2), 4)
)
## $OOB_MSE
## [1] 3.5013
## 
## $Test_MSE
## [1] 4.1243
```

Both mean squared errors measure prediction error, with smaller values
indicating more accurate predictions. OOB MSE uses the training
observations without including them in the corresponding trees; test MSE
uses the 20 held-out observations. Their difference reflects the
different evaluation samples and trees used, and the test estimate is
noisy with only 20 observations.

## Inspect

``` r

print(fit)
## -------------------------------------------------------------
## RLT Regression Forest
## -------------------------------------------------------------
##               (N, P) = (80, 10) [5 continuous, 5 categorical]
##           # of trees = 200
##         (mtry, nmin) = (5, 5)
##       split generate = Best
##             sampling = 100% w/ replace
##           importance = none
##             OOB MSE = 3.5013 (R2 = 0.5883)
## -------------------------------------------------------------
```

The printed summary records the forest settings and OOB error. To assess
which predictors the forest relies on, continue to [Variable
Importance](https://teazrq.github.io/RLT/articles/variable-importance.md).
