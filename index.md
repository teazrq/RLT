# RLT

**Reinforcement Learning Trees** is an R package for random forest
models with regression, classification, and survival analysis support.
The package uses a C++/Rcpp backend with OpenMP for parallel
computation.

[Website](https://teazrq.github.io/RLT/) \|
[GitHub](https://github.com/teazrq/RLT) \|
[Issues](https://github.com/teazrq/RLT/issues)

## Highlights

- Regression, classification, and survival forests
- Reinforcement learning splits with embedded model guidance
- Linear combination splits for multivariate split directions
- Variable importance and random forest kernel utilities
- Survival prediction, confidence bands, and tree inspection tools
- Reproducible parallel fitting through recorded random seeds

## Installation

Install the released version from CRAN when available:

``` r

install.packages("RLT")
```

Install the development version from GitHub:

``` r

# install.packages("remotes")
remotes::install_github("teazrq/RLT")
```

On Windows, source installation requires Rtools. On macOS and Linux,
source installation requires a working C++ toolchain and OpenMP support.

## Quick Example

``` r

library(RLT)

set.seed(1)
n <- 200
p <- 6
x <- matrix(rnorm(n * p), n, p)
y <- x[, 1] - x[, 2] + rnorm(n)

fit <- RLT(
  x, y,
  model = "regression",
  ntrees = 100,
  importance = TRUE,
  verbose = FALSE
)

pred <- predict(fit, x[1:5, ])
pred$Prediction

importance(fit)
```

## Documentation

The pkgdown site is built from the package vignettes and website-only
articles:

- [Get Started](https://teazrq.github.io/RLT/articles/get-started.html)
- [Regression
  Tutorial](https://teazrq.github.io/RLT/articles/regression-tutorial.html)
- [Classification
  Tutorial](https://teazrq.github.io/RLT/articles/classification-tutorial.html)
- [Survival
  Tutorial](https://teazrq.github.io/RLT/articles/survival-tutorial.html)
- [Reference](https://teazrq.github.io/RLT/reference/index.html)

## References

- Zhu, R., Zeng, D., and Kosorok, M. R. (2015). Reinforcement Learning
  Trees. *Journal of the American Statistical Association*, 110(512),
  1770-1784. <https://doi.org/10.1080/01621459.2015.1036994>
- Xu, T., Zhu, R., and Shao, X. (2024). On variance estimation of random
  forests with infinite-order U-statistics. *Electronic Journal of
  Statistics*, 18(1), 2135-2207.
- Formentini, S., Liang, K., and Zhu, R. (2022). Survival Function
  Confidence Band Estimation using Random Forests.
  <https://arxiv.org/abs/2204.12038>
