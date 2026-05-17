# Random Seed and Reproducibility - Tutorial (RLT)

## Overview

This page explains how to control randomness in your analyses with base
R’s [`set.seed()`](https://rdrr.io/r/base/Random.html).

## Where to set the seed

At the very beginning of your analysis (recommended). This covers
synthetic data generation and internal resampling randomness in
[`RLT()`](https://teazrq.github.io/RLT/reference/RLT.md). Or immediately
before model fitting if your data are fixed and only modeling randomness
matters.

**Prerequisites** — See [Get
Started](https://teazrq.github.io/RLT/articles/get-started.md).

## Demonstration — same seed, same results (regression)

We run the same pipeline twice with the same seed and compare outputs.

``` r

# ---------- Run A (seed = 1) ----------
set.seed(1)

# Small dataset (~100 obs)
trainn <- 80; testn <- 20; n <- trainn + testn; p <- 10
X1 <- matrix(rnorm(n * (p/2)), n, p/2)
X2 <- matrix(as.integer(runif(n * (p/2)) * 3), n, p/2)  # integers 0,1,2
X_numeric <- data.frame(X1, X2)

y <- 1 + rowSums(X_numeric[, 2:6]) +
  2 * (X_numeric[, p/2 + 1] %in% c(1, 2)) + rnorm(n)

X <- X_numeric
X[, (p/2 + 1):p] <- lapply(X[, (p/2 + 1):p], as.factor)

trainX <- X[1:trainn, ]; trainY <- y[1:trainn]
testX  <- X[(trainn + 1):(trainn + testn), ]; testY <- y[(trainn + 1):(trainn + testn)]

# Fit
library(RLT)
ntrees <- 200; ncores <- 1
nmin <- 5; mtry <- p/2; samplereplace <- TRUE; sampleprob <- 0.80
rule <- "best"; nsplit <- ifelse(rule == "best", 0, 3); importance <- TRUE

fit_A <- RLT(
  trainX, trainY, model = "regression",
  ntrees = ntrees, mtry = mtry, nmin = nmin,
  resample.prob = sampleprob, split.gen = rule,
  resample.replace = samplereplace,
  nsplit = nsplit, importance = importance,
  param.control = list(alpha = 0),
  ncores = ncores, verbose = FALSE
)
pred_A <- predict(fit_A, testX, ncores = ncores)

mse_train_A <- mean((fit_A$Prediction - trainY)^2)
mse_test_A  <- mean((pred_A$Prediction - testY)^2)

# ---------- Run B (same seed = 1) ----------
set.seed(1)

# Recreate the same data and pipeline
trainn <- 80; testn <- 20; n <- trainn + testn; p <- 10
X1 <- matrix(rnorm(n * (p/2)), n, p/2)
X2 <- matrix(as.integer(runif(n * (p/2)) * 3), n, p/2)
X_numeric <- data.frame(X1, X2)
y <- 1 + rowSums(X_numeric[, 2:6]) +
  2 * (X_numeric[, p/2 + 1] %in% c(1, 2)) + rnorm(n)

X <- X_numeric
X[, (p/2 + 1):p] <- lapply(X[, (p/2 + 1):p], as.factor)

trainX <- X[1:trainn, ]; trainY <- y[1:trainn]
testX  <- X[(trainn + 1):(trainn + testn), ]; testY <- y[(trainn + 1):(trainn + testn)]

fit_B <- RLT(
  trainX, trainY, model = "regression",
  ntrees = ntrees, mtry = mtry, nmin = nmin,
  resample.prob = sampleprob, split.gen = rule,
  resample.replace = samplereplace,
  nsplit = nsplit, importance = importance,
  param.control = list(alpha = 0),
  ncores = ncores, verbose = FALSE
)
pred_B <- predict(fit_B, testX, ncores = ncores)

mse_train_B <- mean((fit_B$Prediction - trainY)^2)
mse_test_B  <- mean((pred_B$Prediction - testY)^2)

# ---------- Summary for same-seed runs ----------
list(
  A_Train_MSE = round(mse_train_A, 6),
  A_Test_MSE  = round(mse_test_A, 6),
  B_Train_MSE = round(mse_train_B, 6),
  B_Test_MSE  = round(mse_test_B, 6),
  SameSeed_Predictions_Identical = isTRUE(all.equal(pred_A$Prediction, pred_B$Prediction))
)
## $A_Train_MSE
## [1] 3.456843
## 
## $A_Test_MSE
## [1] 4.086383
## 
## $B_Train_MSE
## [1] 3.456843
## 
## $B_Test_MSE
## [1] 4.086383
## 
## $SameSeed_Predictions_Identical
## [1] TRUE
```

## Demonstration — different seed, potentially different results

Now we change the seed and rerun the same pipeline once.

``` r

# ---------- Run C (seed = 2) ----------
set.seed(2)

trainn <- 80; testn <- 20; n <- trainn + testn; p <- 10
X1 <- matrix(rnorm(n * (p/2)), n, p/2)
X2 <- matrix(as.integer(runif(n * (p/2)) * 3), n, p/2)
X_numeric <- data.frame(X1, X2)
y <- 1 + rowSums(X_numeric[, 2:6]) +
  2 * (X_numeric[, p/2 + 1] %in% c(1, 2)) + rnorm(n)

X <- X_numeric
X[, (p/2 + 1):p] <- lapply(X[, (p/2 + 1):p], as.factor)

trainX <- X[1:trainn, ]; trainY <- y[1:trainn]
testX  <- X[(trainn + 1):(trainn + testn), ]; testY <- y[(trainn + 1):(trainn + testn)]

fit_C <- RLT(
  trainX, trainY, model = "regression",
  ntrees = ntrees, mtry = mtry, nmin = nmin,
  resample.prob = sampleprob, split.gen = rule,
  resample.replace = samplereplace,
  nsplit = nsplit, importance = importance,
  param.control = list(alpha = 0),
  ncores = ncores, verbose = FALSE
)
pred_C <- predict(fit_C, testX, ncores = ncores)

mse_train_C <- mean((fit_C$Prediction - trainY)^2)
mse_test_C  <- mean((pred_C$Prediction - testY)^2)

list(
  C_Train_MSE = round(mse_train_C, 6),
  C_Test_MSE  = round(mse_test_C, 6),
  DiffSeed_Predictions_EqualTo_RunA = isTRUE(all.equal(pred_C$Prediction, pred_A$Prediction))
)
## $C_Train_MSE
## [1] 3.323691
## 
## $C_Test_MSE
## [1] 2.306646
## 
## $DiffSeed_Predictions_EqualTo_RunA
## [1] FALSE
```

## Tips

- Choose any integer you like for the seed; the specific value doesn’t
  matter—consistency does.
- Keep one [`set.seed()`](https://rdrr.io/r/base/Random.html) near the
  top of your script to make the whole workflow reproducible.
- The same pattern works for classification and survival: place
  [`set.seed()`](https://rdrr.io/r/base/Random.html) before data
  simulation (if any) and before
  [`RLT()`](https://teazrq.github.io/RLT/reference/RLT.md).
