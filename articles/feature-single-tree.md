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

Use `get.one.tree(<fit>, <tree_id>)` to print a single tree. Choose
`tree_id` from 1 to `ntrees` (here we take the first tree).

``` r

# Peek into a single tree
get.one.tree(fit, 1)
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
##    11     5  X4                                  0.2242     10      0.0000
##    12     6  *                                        -      1     -1.9927
##    13     6  X4.1(F)                             2.0000      6      0.0000
##    14     7  *                                        -      2     -1.4353
##    15     7  *                                        -      4     -1.1561
##    16     6  *                                        -      3     -0.6253
##    17     6  X5.1(F)                             2.0000      7      0.0000
##    18     7  *                                        -      4      0.6492
##    19     7  *                                        -      3      1.3944
##    20     3  *                                        -      4     -0.3549
##    21     3  X4                                  0.8119     18      0.0000
##    22     4  X5                                 -1.4252     16      0.0000
##    23     4  *                                        -      2      7.1715
##    24     5  *                                        -      3      1.2651
##    25     5  X2.1(F)                            12.0000     13      0.0000
##    26     6  *                                        -      5      3.2027
##    27     6  X2                                 -0.4950      8      0.0000
##    28     7  *                                        -      1      2.1524
##    29     7  X5                                 -0.4019      7      0.0000
##    30     8  *                                        -      4      4.2790
##    31     8  *                                        -      3      5.8188
##    32     2  X5                                 -1.0821     17      0.0000
##    33     2  *                                        -      3      8.5867
##    34     3  *                                        -      2      2.8976
##    35     3  X4                                 -1.0991     15      0.0000
##    36     4  *                                        -      1      2.5067
##    37     4  X4                                 -0.6040     14      0.0000
##    38     5  *                                        -      3      6.7543
##    39     5  X3                                 -0.1644     11      0.0000
##    40     6  *                                        -      4      4.4821
##    41     6  X4                                  0.1635      7      0.0000
##    42     7  *                                        -      1      5.1204
##    43     7  X3                                  0.1327      6      0.0000
##    44     8  *                                        -      2      5.4671
##    45     8  *                                        -      4      5.8524
```
