# RLT Regression Tutorial

## Overview

This page shows how to fit and predict a regression model with RLT.

## Prerequisites

Install RLT, then load it with
[`library(RLT)`](https://github.com/teazrq/RLT).

## Data

We generate continuous and categorical predictors with a continuous
outcome.

``` r

# (Optional) For reproducibility in this tutorial only.
# Detailed notes on random seeds are in the Seed & Reproducibility feature page.
set.seed(1)

# ---- Generate a small synthetic dataset ----
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

``` r

ntrees <- 200
ncores <- 1
nmin   <- 5
mtry   <- p/2
samplereplace <- TRUE
sampleprob    <- 0.80
rule    <- "best"
nsplit  <- ifelse(rule == "best", 0, 3)
importance <- TRUE

fit <- RLT(
  trainX, trainY, model = "regression",
  ntrees = ntrees, mtry = mtry, nmin = nmin,
  resample.prob = sampleprob, split.gen = rule,
  resample.replace = samplereplace,
  nsplit = nsplit, importance = importance,
  param.control = list(alpha = 0),
  ncores = ncores, verbose = FALSE
)
```

## Predict

``` r

pred <- predict(fit, testX, ncores = ncores)

# Helper in case predict() returns a list with $Prediction
get_pred <- function(obj) if (is.list(obj) && !is.null(obj$Prediction)) obj$Prediction else as.numeric(obj)

train_pred <- if (!is.null(fit$Prediction)) fit$Prediction else get_pred(predict(fit, trainX, ncores = ncores))
test_pred  <- get_pred(pred)
```

## Evaluate

``` r

mse_train <- mean((train_pred - trainY)^2)
mse_test  <- mean((test_pred  - testY)^2)

# A compact summary
list(
  Train_MSE = round(mse_train, 4),
  Test_MSE  = round(mse_test, 4)
)
## $Train_MSE
## [1] 3.4568
## 
## $Test_MSE
## [1] 4.0864
```

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
##             sampling = 80% w/ replace
##           importance = permutation
##             OOB MSE = 3.4568 (R2 = 0.5935)
## -------------------------------------------------------------
```
