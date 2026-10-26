# Variable Importance Summary

Extract variable importance from a fitted RLT model. When importance
variance is supplied by `var.mode = "matched"`, estimated standard
errors, Z-scores, and significance codes are also reported. Unavailable
or negative variance estimates yield `NA` for SD and Z, and an empty
significance code.

## Usage

``` r
importance(object, ...)
```

## Arguments

- object:

  A fitted `RLT` object from
  [`RLT`](https://teazrq.github.io/RLT/reference/RLT.md).

- ...:

  Additional arguments (unused).

## Value

A `data.frame` with columns `Variable` and `VI`. If the fitted object
also contains importance variance estimates, the result includes `SD`,
`Z` and `Sig`:

- `Variable`: variable name

- `VI`: variable importance

- `SD`: estimated standard error of VI (`NA` for unavailable or negative
  variance)

- `Z`: Z-score (`VI / SD`, `NA` if SD is `NA`)

- `Sig`: significance code (`""` if not estimated or negative variance)

Significance codes: `***` \|Z\| \>= 2.58, `**` \|Z\| \>= 1.96, `*` \|Z\|
\>= 1.64.

## Examples

``` r
# \donttest{
set.seed(42)
x <- matrix(rnorm(100 * 5), ncol = 5)
colnames(x) <- paste0("X", seq_len(ncol(x)))
y <- x[, 1] - x[, 2] + rnorm(100)
fit <- RLT(x, y, ntrees = 100, importance = TRUE, seed = 42)
importance(fit)
#> Variable             VI
#> -------------------------- 
#> X1               1.0122
#> X2               1.5026
#> X3               0.0281
#> X4               0.0513
#> X5               0.0247
# }
```
