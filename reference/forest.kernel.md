# random forest kernel

    Get random forest induced kernel weight matrix of testing samples
                 or between any two sets of data. This is an experimental feature.
                 Use at your own risk.

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

  To calculate the kernel weights with respect to the training data.
  This is slightly different than supplying the training data to `X2`
  due to re-samplings of the training process. To use this feature, you
  must specify `resample.track = TRUE` in `param.control` when fitting
  the forest

- oob:

  Logical. If `TRUE`, compute the OOB (out-of-bag) self-kernel, which
  counts co-occurrence only from trees where both observations are OOB.
  This eliminates the self-contamination bias that arises when in-bag
  observations influence tree structure. Requires
  `resample.track = TRUE` and `X1` to be the original training data.
  `X2` must be `NULL` (OOB kernel is defined for self-kernel only). The
  returned list contains three matrices: `Kernel` (normalized
  co-occurrence in \\\[0,1\]\\), `N` (number of trees where both are
  OOB), and `C` (number of trees where both are OOB and share a leaf).

- verbose:

  Whether fitting should be printed.

- ...:

  ... Additional arguments.

## Value

A list containing the kernel matrix. For `oob = TRUE`, the list also
contains `N` (OOB co-occurrence count) and `C` (OOB leaf-sharing count).

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
