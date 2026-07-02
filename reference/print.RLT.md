# Print a RLT object

Print a RLT object

## Usage

``` r
# S3 method for class 'RLT'
print(x, ...)
```

## Arguments

- x:

  A fitted RLT object

- ...:

  ...

## Examples

``` r
# \donttest{
set.seed(42)
x <- matrix(rnorm(100 * 5), ncol = 5)
y <- rowSums(x[, 1:2]) + rnorm(100)
fit <- RLT(x, y, ntrees = 50)
print(fit)
#> ------------------------------------------
#> RLT Regression Forest
#> ------------------------------------------
#>               (N, P) = (100, 5)
#>           # of trees = 50
#>         (mtry, nmin) = (2, 5)
#>       split generate = Best
#>             sampling = 100% w/ replace
#>           importance = none
#>             OOB MSE = 1.4062 (R2 = 0.4623)
#> ------------------------------------------
# }
```
