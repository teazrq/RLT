# Variable Importance Summary

Extract variable importance from a fitted RLT model. When variance
estimation was enabled via `var.mode`, standard deviations, Z-scores,
and significance codes are also reported. Negative variance estimates
yield `NA` for SD, Z, and significance.

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

A `data.frame` with columns:

- `Variable`: variable name

- `VI`: variable importance

- `SD`: standard deviation of VI (`NA` if not estimated or negative
  variance)

- `Z`: Z-score (`VI / SD`, `NA` if SD is `NA`)

- `Sig`: significance code (`""` if not estimated or negative variance)

Significance codes: `***` \|Z\| \>= 2.58, `**` \|Z\| \>= 1.96, `*` \|Z\|
\>= 1.64.

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- RLT(x, y, model = "classification", importance = TRUE, var.mode = TRUE)
importance(fit)
} # }
```
