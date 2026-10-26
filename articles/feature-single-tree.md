# Single Tree - Tutorial (RLT)

## Overview

Use
[`get.one.tree()`](https://teazrq.github.io/RLT/reference/get.one.tree.md)
to inspect the splits and terminal-node predictions in one tree of a
fitted forest. The example below fits an unweighted regression forest
with both continuous and categorical predictors.

## Data

We generate continuous and categorical predictors with a continuous
outcome.

``` r

set.seed(1)

# ---- Generate a small synthetic dataset ----
n <- 80
p <- 10

# Continuous + categorical predictors (last half as factors)
X1 <- matrix(rnorm(n * (p/2)), n, p/2)
X2 <- matrix(as.integer(runif(n * (p/2)) * 3), n, p/2)  # integers 0,1,2

# Continuous outcome with a simple signal + noise
X_numeric <- data.frame(X1, X2)
colnames(X_numeric) <- paste0("X", seq_len(p))
y <- 1 + rowSums(X_numeric[, 2:6]) +
  2 * (X_numeric[, p/2 + 1] %in% c(1, 2)) + rnorm(n)

X <- X_numeric
X[, (p/2 + 1):p] <- lapply(X[, (p/2 + 1):p], as.factor)

trainX <- X
trainY <- y
```

## Fit

``` r

library(RLT)

fit <- RLT(
  trainX, trainY, model = "regression",
  ntrees = 200, mtry = p/2, nmin = 5,
  resample.prob = 0.8, resample.replace = TRUE,
  nsplit = 0,
  ncores = 1, verbose = FALSE
)
```

Setting `nsplit = 0` searches all available cutpoints for each candidate
split variable.

## Inspect one tree

Choose a tree number from 1 to the number of fitted trees. This call
prints the first tree and invisibly returns a data frame.

``` r

get.one.tree(fit, tree = 1)
```

    ## Tree #1  [Regression]
    ## 
    ## Node  Depth  Split                                Value      n     NodeAve
    ## ------------------------------------------------------------------------------ 
    ##     1     0  X6(F)                              12.0000     64      0.0000
    ##     2     1  X1                                  0.2375     24      0.0000
    ##     3     1  X4                                  0.1177     40      0.0000
    ##     4     2  X4                                 -0.5699     17      0.0000
    ##     5     2  X2                                 -0.1728      7      0.0000
    ##     6     3  *                                        -      3     -0.8053
    ##     7     3  X5                                  1.2344     14      0.0000
    ##     8     4  *                                        -      4      1.2231
    ##     9     4  X4                                 -0.4458     10      0.0000
    ##    10     5  *                                        -      4      2.6146
    ## 
    ## ... output truncated ...

The preview shows the first few printed rows. `Node` identifies the
node, and `Depth` counts splits from the root, whose depth is 0. `Split`
names the predictor, with `(F)` marking a factor and `*` marking a
terminal node. For a continuous split, `Value` is the cutpoint; for a
factor, it encodes a partition of categories and should not be read as a
numeric threshold. Terminal nodes display `-` because they have no split
value.

Here `n` is the number of in-bag observations reaching a node, counting
repeated draws. At a terminal node, the printed `NodeAve` is the mean
response used as that leaf’s prediction. The values printed under
`NodeAve` for internal nodes are not node means. The returned data frame
contains `Node`, `Depth`, `Split`, `Value`, and `n`; `NodeAve` is part
of the printed regression display.
