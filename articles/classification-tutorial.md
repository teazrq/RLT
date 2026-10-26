# RLT Classification Tutorial

## Overview

Use a classification forest to predict class labels and probabilities
from numeric and categorical predictors. We simulate a binary response,
fit on 80 observations, and evaluate predictions on 20 held-out
observations.

## Prerequisites

Follow [Get
Started](https://teazrq.github.io/RLT/articles/get-started.md) to
install RLT, then load the package:

``` r

library(RLT)
```

## Data Preparation

The first five predictors are numeric and the last five are factors.
Factor columns tell RLT to use categorical splits. The probability of
class `1` increases with the first numeric predictor; the other
predictors have no effect on the simulated response. The response itself
must be a factor for classification.

``` r

set.seed(1)

trainn <- 80
testn  <- 20
n <- trainn + testn
p <- 10

# Continuous + categorical predictors (last half as factors)
X1 <- matrix(rnorm(n * (p/2)), n, p/2)
X2 <- matrix(as.integer(runif(n * (p/2)) * 10), n, p/2)  # integers 0-9

X <- data.frame(X1, X2)
X[, (p/2 + 1):p] <- lapply(X[, (p/2 + 1):p], as.factor)

# Binary outcome via a simple logistic signal
prob  <- plogis(-0.5 + 2 * X[, 1])
y     <- factor(rbinom(n, 1, prob = prob), levels = c(0, 1))

# Train / test split
trainX <- X[1:trainn, ]
trainY <- y[1:trainn]
testX  <- X[(trainn + 1):(trainn + testn), ]
testY  <- y[(trainn + 1):(trainn + testn)]
```

## Fit a classification model

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
  trainX, trainY, model = "classification",
  ntrees = 200, mtry = p/2, nsplit = 0,
  seed = 1, ncores = 1, verbose = FALSE
)
```

## Predict

``` r

pred <- predict(fit, testX)
head(data.frame(Observed = testY, Predicted = pred$Prediction))
##   Observed Predicted
## 1        0         0
## 2        1         0
## 3        0         1
## 4        0         0
## 5        1         1
## 6        0         1
head(pred$Prob)
##           [,1]      [,2]
## [1,] 0.6855000 0.3145000
## [2,] 0.5680833 0.4319167
## [3,] 0.3050000 0.6950000
## [4,] 0.7301667 0.2698333
## [5,] 0.1921667 0.8078333
## [6,] 0.3932500 0.6067500
```

`pred$Prediction` already contains factor class labels. `pred$Prob`
contains one row per held-out observation and one column per class, in
`levels(trainY)` order.

## Evaluate

The fitted object’s `Prediction` component contains out-of-bag (OOB)
class predictions. Each training observation is classified using only
trees whose resample omitted it. The test predictions use the fitted
forest on observations reserved before fitting.

``` r

list(
  OOB_Accuracy = round(mean(fit$Prediction == trainY), 4),
  Test_Accuracy = round(mean(pred$Prediction == testY), 4)
)
## $OOB_Accuracy
## [1] 0.725
## 
## $Test_Accuracy
## [1] 0.8
```

Accuracy is the fraction of correctly classified observations, so larger
values indicate better classification. OOB accuracy uses the training
observations without including them in the corresponding trees; test
accuracy uses the 20 held-out observations. With this small test set,
each additional correct classification changes test accuracy by 0.05.

## Inspect the fitted object

``` r

print(fit)
## -------------------------------------------------------------
## RLT Classification Forest
## -------------------------------------------------------------
##               (N, P) = (80, 10) [5 continuous, 5 categorical]
##           # of trees = 200
##         (mtry, nmin) = (5, 5)
##       split generate = Best
##             sampling = 100% w/ replace
##           importance = none
##       OOB misclass = 27.5%
## -------------------------------------------------------------
```

The printed summary records the forest settings and OOB
misclassification error, which is one minus OOB accuracy. To assess
which predictors the forest relies on, continue to [Variable
Importance](https://teazrq.github.io/RLT/articles/variable-importance.md).
