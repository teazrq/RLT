# Compute a random forest kernel

Measure how often observations share terminal nodes in a fitted forest,
within one dataset or between two datasets. This is an experimental
feature; the available modes and their data requirements are described
below.

## Usage

``` r
forest.kernel(
  object,
  X1 = NULL,
  X2 = NULL,
  vs.train = FALSE,
  oob = FALSE,
  verbose = FALSE,
  ...
)
```

## Arguments

- object:

  A fitted RLT object.

- X1:

  The dataset for prediction. This calculates an \\n_1 \times n_1\\
  kernel matrix of `X1`.

- X2:

  The dataset for reference/training. If `X2` is supplied, then
  calculate an \\n_1 \times n_2\\ kernel matrix. If `vs.train` is used,
  then this must be the original training data.

- vs.train:

  Include each reference observation's in-bag multiplicity in its
  leaf-sharing count. `X2` must be the original training data in its
  original row order. To use this feature, you must specify
  `resample.track = TRUE` in `param.control` when fitting the forest

- oob:

  Logical. If `TRUE`, compute the OOB (out-of-bag) self-kernel, which
  counts co-occurrence only from trees where both observations are OOB.
  Each contributing tree was fitted without either observation in the
  pair. Requires `resample.track = TRUE` and `X1` to be the original
  training data. `X2` must be `NULL` (OOB kernel is defined for
  self-kernel only). The returned list contains three matrices: `Kernel`
  (normalized co-occurrence in \\\[0,1\]\\ where `N > 0`), `N` (number
  of trees where both are OOB), and `C` (number of trees where both are
  OOB and share a leaf). Entries with `N = 0` are returned as zero;
  inspect `N` to distinguish these unsupported pairs from observed zero
  leaf sharing.

- verbose:

  Whether fitting should be printed.

- ...:

  ... Additional arguments.

## Value

A list containing the kernel matrix. For `oob = TRUE`, the list also
contains `N` (OOB co-occurrence count) and `C` (OOB leaf-sharing count).

## Details

Ordinary kernel entries count trees in which two observations share a
terminal node. They are not normalized observation contributions to a
forest prediction. The training-data mode includes in-bag
multiplicities; the OOB mode instead normalizes by the number of trees
eligible for each pair.

## Examples

``` r
# \donttest{
  set.seed(42)
  x <- matrix(rnorm(200 * 5), ncol = 5)
  y <- rowSums(x[, 1:2]) + rnorm(200)
  fit <- RLT(x, y, ntrees = 100)
  K <- forest.kernel(fit, X1 = x[1:5, ])
  print(K$Kernel[1:3, 1:3])
#>      [,1] [,2] [,3]
#> [1,]  100    0    0
#> [2,]    0  100    0
#> [3,]    0    0  100
# }
```
