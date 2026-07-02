# Print Importance Summary

Print method for `importance.RLT` objects.

## Usage

``` r
# S3 method for class 'importance.RLT'
print(x, digits = 4, ...)
```

## Arguments

- x:

  An `importance.RLT` object.

- digits:

  Number of digits for formatting. Default: 4.

- ...:

  Additional arguments (unused).

## Examples

``` r
# \donttest{
set.seed(42)
x <- matrix(rnorm(100 * 5), ncol = 5)
y <- rowSums(x[, 1:2]) + rnorm(100)
fit <- RLT(x, y, ntrees = 50, importance = TRUE)
print(importance(fit))
#> Variable             VI
#> -------------------------- 
#> V1               1.1370
#> V2               0.7387
#> V3               0.1624
#> V4              -0.0617
#> V5              -0.1400
# }
```
