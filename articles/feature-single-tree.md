# Single Tree - Tutorial (RLT)

## Overview

This page shows how to inspect a single tree grown inside an RLT forest
using
[`get.one.tree()`](https://teazrq.github.io/RLT/reference/get.one.tree.md).

## Data

We generate continuous and categorical predictors with a continuous
outcome.

``` r

# (Optional) For reproducibility in this tutorial only.
set.seed(1)

# ---- Generate a small synthetic dataset ----
trainn <- 80
testn  <- 20
n <- trainn + testn
p <- 10

# Continuous + categorical predictors (last half as factors)
X1 <- matrix(rnorm(n * (p/2)), n, p/2)
X2 <- matrix(as.integer(runif(n * (p/2)) * 3), n, p/2)  # integers 0,1,2

# Continuous outcome with a simple signal + noise
X_numeric <- data.frame(X1, X2)
y <- 1 + rowSums(X_numeric[, 2:6]) +
  2 * (X_numeric[, p/2 + 1] %in% c(1, 2)) + rnorm(n)

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

# install.packages("devtools"); devtools::install_github("teazrq/RLT")
library(RLT)

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

## Inspect one tree

Use `get.one.tree(<fit>, <tree_id>)` to inspect a single tree. Choose
`tree_id` from 1 to `ntrees` (here we take the first tree).

``` r

# Preview the first few printed lines.
tree_output <- capture.output(get.one.tree(fit, 1))
cat(head(tree_output, 14), sep = "\n")
## Tree #1  [Regression]
## 
## Node  Depth  Split                                Value      n     NodeAve
## ------------------------------------------------------------------------------ 
##     1     0  X1.1(F)                             8.0000     64      0.0000
##     2     1  X1.1(F)                             4.0000     44      0.0000
##     3     1  X5                                  1.5463     20      0.0000
##     4     2  X2                                  0.8354     22      0.0000
##     5     2  X2                                 -0.6395     22      0.0000
##     6     3  X5                                  0.7657     18      0.0000
##     7     3  *                                        -      4      5.5438
##     8     4  X3.1(F)                             6.0000     17      0.0000
##     9     4  *                                        -      1      2.9351
##    10     5  X4.1(F)                             6.0000      7      0.0000
if (length(tree_output) > 14) {
  cat("\n... output truncated ...\n")
}
## 
## ... output truncated ...
```
