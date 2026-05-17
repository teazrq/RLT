# RLT Survival Analysis Tutorial

## Introduction

This vignette introduces survival analysis with **RLT** (Reinforcement
Learning Trees). RLT survival forests estimate individual survival,
hazard, and cumulative hazard functions via ensemble tree methods. Key
features include:

- **Three split rules**: logrank (default), suplogrank, and coxgrad.
- **Linear combination (LC) splits**: combine multiple variables into a
  single split direction.
- **Variance estimation**: matched-sample U-statistic, infinitesimal
  jackknife (IJ), and jackknife.
- **Confidence bands**:
  [`get.surv.band()`](https://teazrq.github.io/RLT/reference/get.surv.band.md)
  provides naive or smoothed simultaneous bands for survival curves.
- **Tree inspection**:
  [`get.one.tree()`](https://teazrq.github.io/RLT/reference/get.one.tree.md)
  inspects individual tree structures.

The examples below use small simulated datasets so that all code runs
quickly.

## Simulated data

We simulate data from a proportional hazards model with exponential
event times. The first two predictors carry signal; the rest are noise.
About 30% of observations are censored.

``` r

set.seed(42)
n <- 200
p <- 5
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("V", 1:p)

beta <- c(0.8, 0.5, 0, 0, 0)
hazard <- exp(X %*% beta)
surv_time <- rexp(n, rate = hazard)
censor_time <- runif(n, 0, 3)

y <- pmin(surv_time, censor_time)
censor <- as.numeric(surv_time <= censor_time)

table(censor)
#> censor
#>   0   1 
#>  76 124
```

## Basic usage

Fit a survival forest with `model = "survival"`. The third argument is
the censoring indicator (`1` = event observed, `0` = censored). By
default `split.rule = "logrank"`.

``` r

library(RLT)

fit <- RLT(X, y, censor,
           model = "survival",
           ntrees = 100,
           nmin = 5,
           verbose = FALSE)
fit
#> --------------------------------------
#> RLT Survival Forest
#> --------------------------------------
#>               (N, P) = (200, 5)
#>           # of trees = 100
#>         (mtry, nmin) = (2, 5)
#>       split generate = Best
#>             sampling = 100% w/ replace
#>           importance = none
#>           OOB error = 0.3283
#> --------------------------------------
```

Predict on new data (or the training data) to obtain survival curves,
hazards, and cumulative hazards:

``` r

pred <- predict(fit, X[1:5, ])

# Each component is an N x T matrix, where T is the number of unique failure times
str(pred$Survival)      # Survival function S(t)
#>  num [1:5, 1:124] 1 1 0.98 0.985 1 0.998 1 0.98 0.985 1 ...
str(pred$Hazard)        # Hazard function h(t)
#>  num [1:5, 1:124] 0 0 0.02 0.015 0 0.002 0 0 0 0 ...
str(pred$CHF)           # Cumulative hazard H(t)
#>  num [1:5, 1:124] 0 0 0.02 0.015 0 0.002 0 0.02 0.015 0 ...

# For survival forests, $Prediction is NULL
pred$Prediction
#> NULL
```

Plot the predicted survival curve for the first subject:

``` r

plot(pred$timepoints, pred$Survival[1, ], type = "s",
     xlab = "Time", ylab = "Survival Probability",
     main = "Predicted Survival Curve (Subject 1)")
```

![](survival-tutorial_files/figure-html/basic-plot-1.png)

## Split rules

RLT provides three splitting criteria for survival trees:

| Rule | Description | Best for |
|----|----|----|
| `logrank` | Standard log-rank test statistic (default) | General use, clear hazard differences |
| `suplogrank` | Supremum (maximum) of the standardized log-rank process over time | Non-proportional hazards, time-varying effects |
| `coxgrad` | Gradient of Cox partial likelihood | When a Cox-like direction is plausible; supports observation weights |

Fit the three rules on the same data and compare out-of-bag error
estimates:

``` r

fit_lr  <- RLT(X, y, censor, model = "survival", ntrees = 100,
               split.rule = "logrank",  verbose = FALSE)
fit_slr <- RLT(X, y, censor, model = "survival", ntrees = 100,
               split.rule = "suplogrank", verbose = FALSE)
fit_cg  <- RLT(X, y, censor, model = "survival", ntrees = 100,
               split.rule = "coxgrad", verbose = FALSE)

c(logrank = fit_lr$Error, suplogrank = fit_slr$Error, coxgrad = fit_cg$Error)
#>    logrank suplogrank    coxgrad 
#>  0.3341685  0.3252302  0.3718174
```

In practice, `logrank` is a safe default. `suplogrank` can be
advantageous when hazard ratios change over time. `coxgrad` is useful
when you want to incorporate observation weights (see below) or when the
data follow a Cox-like structure.

## Observation weights

Observation weights are passed via `obs.w`. For survival forests,
weights are **not** used by `logrank` or `suplogrank` (due to the
difficulty of weighted variance estimation for the test statistic), but
they **are** used by `coxgrad`.

``` r

w <- runif(n)
fit_w <- RLT(X, y, censor, model = "survival", ntrees = 100,
             split.rule = "coxgrad", obs.w = w, verbose = FALSE)
fit_w
#> --------------------------------------
#> RLT Survival Forest
#> --------------------------------------
#>               (N, P) = (200, 5)
#>           # of trees = 100
#>         (mtry, nmin) = (2, 5)
#>       split generate = Best
#>             sampling = 100% w/ replace
#>          obs weights = Yes
#>           importance = none
#>           OOB error = 0.3717
#> --------------------------------------
```

## Linear combination splits

When `linear.comb > 1`, each split uses a linear combination of
`linear.comb` variables instead of a single variable. For survival
forests, the available methods are:

- `"coxph"` (default): coefficients from a local Cox model fit.
- `"naive"`: simple correlation-based direction.

Specify these through `param.control`:

``` r

fit_lc <- RLT(X, y, censor,
              model = "survival",
              ntrees = 100,
              split.rule = "logrank",
              param.control = list(
                linear.comb = 3,
                linear.comb.method = "coxph"
              ),
              verbose = FALSE)
fit_lc
#> ----------------------------------------
#> RLT Survival Forest (Linear Combination)
#> ----------------------------------------
#>               (N, P) = (200, 5)
#>           # of trees = 100
#>         (mtry, nmin) = (2, 5)
#>       split generate = Best
#> linear combination split = 3
#>             sampling = 100% w/ replace
#>           importance = none
#>           OOB error = 0.3148
#> ----------------------------------------
```

Predictions from LC forests have the same structure as standard forests:

``` r

pred_lc <- predict(fit_lc, X[1:5, ])
str(pred_lc$Survival)
#>  num [1:5, 1:124] 0.985 0.998 0.993 0.995 1 ...
```

## Variable importance

Set `importance = TRUE` to compute variable importance. The importance
measure for survival forests is based on the decrease in the splitting
criterion (logrank, suplogrank, or coxgrad).

``` r

fit_imp <- RLT(X, y, censor,
               model = "survival",
               ntrees = 100,
               importance = TRUE,
               verbose = FALSE)

importance(fit_imp)
#> Variable             VI
#> -------------------------- 
#> V1               0.0763
#> V2               0.0283
#> V3               0.0038
#> V4              -0.0008
#> V5              -0.0007
```

When variance estimation is enabled (see next section),
[`importance()`](https://teazrq.github.io/RLT/reference/importance.md)
also reports standard errors, Z-scores, and significance codes.

## Variance estimation and confidence bands

RLT supports three variance estimation strategies for survival
predictions:

- **`"matched"`**: matched-sample U-statistic decomposition. Requires an
  even number of trees and subsampling without replacement at 50%
  (automatically adjusted).
- **`"IJ"`**: infinitesimal jackknife.
- **`"jack"`**: jackknife variance.

Enable variance estimation during fitting via `var.mode`, then request
covariance matrices at prediction time with `var.est = TRUE`.

The following example uses `eval = FALSE` because reliable variance
estimation typically requires many trees (e.g., 1,000+).

``` r

fit_var <- RLT(X, y, censor,
               model = "survival",
               ntrees = 1000,
               var.mode = "matched",
               verbose = FALSE)

# Predict with variance estimation
pred_var <- predict(fit_var, X[1:3, ], var.est = TRUE)

# pred_var$Cov is a T x T x N array: covariance of the cumulative hazard over time
str(pred_var$Cov)

# Marginal variances and critical values for bands
str(pred_var$MarginalVar)
str(pred_var$CVproj)
```

### Confidence bands with `get.surv.band()`

Given a prediction object with variance information,
[`get.surv.band()`](https://teazrq.github.io/RLT/reference/get.surv.band.md)
computes simultaneous confidence bands for the survival function. Two
approaches are available:

- **`"naive"`**: uses the full covariance matrix with a Monte Carlo
  critical value.
- **`"smoothed"`**: GAM-smoothed low-rank covariance plus
  eigenvalue-ratio weighted residual correction.

``` r

# Naive band for the first test subject
band_naive <- get.surv.band(pred_var, i = 1, alpha = 0.05,
                            approach = "naive", nsim = 5000)

# Smoothed band
band_smooth <- get.surv.band(pred_var, i = 1, alpha = 0.05,
                             approach = "smoothed",
                             nsim = 5000, k_rank = 10)

# Plot survival curve with naive band
t <- band_naive$timepoints
plot(t, pred_var$Survival[1, ], type = "s", ylim = c(0, 1),
     xlab = "Time", ylab = "Survival",
     main = "Survival Curve with 95% Confidence Band")
lines(t, band_naive$Subject1$lower, type = "s", col = "blue", lty = 2)
lines(t, band_naive$Subject1$upper, type = "s", col = "blue", lty = 2)
legend("topright", legend = c("Estimate", "95% Band"),
       col = c("black", "blue"), lty = c(1, 2))
```

You can also request all subjects at once with `i = 0` (the default).

### Reducing the time grid for bands

For large datasets, the full set of failure times can make covariance
matrices unwieldy. Use `band.grid.size` in
[`predict()`](https://rdrr.io/r/stats/predict.html) to evaluate variance
on a reduced quantile-based grid:

``` r

pred_reduced <- predict(fit_var, X[1:3, ], var.est = TRUE, band.grid.size = 50)
length(pred_reduced$timepoints)  # <= 50 time points
```

## Inspecting individual trees

Use
[`get.one.tree()`](https://teazrq.github.io/RLT/reference/get.one.tree.md)
to inspect the structure of any tree in the fitted forest. The preview
below shows the first few printed lines so the tutorial stays compact.

``` r

# Standard (single-variable) survival tree
tree_output <- capture.output(get.one.tree(fit, tree = 1))
cat(head(tree_output, 14), sep = "\n")
#> Tree #1  [Survival]
#> 
#> Node  Depth  Split                                Value      n
#> -------------------------------------------------------------- 
#>     1     0  V3                                 -1.9954    189
#>     2     1  V2                                  0.8901      4
#>     3     1  V1                                  1.7690      3
#>     4     2  V4                                 -0.5430      3
#>     5     2  *                                        -      4
#>     6     3  *                                        -      4
#>     7     3  *                                        -      3
#>     8     2  V2                                  1.0553     30
#>     9     2  *                                        -      3
#>    10     3  V1                                 -0.1216     84
if (length(tree_output) > 14) {
  cat("\n... output truncated ...\n")
}
#> 
#> ... output truncated ...
```

For LC forests,
[`get.one.tree()`](https://teazrq.github.io/RLT/reference/get.one.tree.md)
also shows the linear combination coefficients at each internal node:

``` r

tree_output <- capture.output(get.one.tree(fit_lc, tree = 1))
cat(head(tree_output, 14), sep = "\n")
#> Tree #1  [Survival, Linear Combination]
#> 
#> Node  Depth  Split                                Value      n
#> -------------------------------------------------------------- 
#>     1     0  0.453·V5 + 0.891·V3               2.0860    200
#>     2     1  0.395·V5 - 0.919·V4              -1.9186    193
#>     3     1  *                                        -      7
#>     4     2  *                                        -      2
#>     5     2  0.999·V1 + 0.036·V3               0.2900    191
#>     6     3  0.996·V1 - 0.088·V4              -0.0492    108
#>     7     3  0.975·V2 + 0.223·V1               1.0275     83
#>     8     4  0.989·V2 + 0.151·V4              -0.4532     80
#>     9     4  0.914·V5 + 0.405·V3               0.3363     28
#>    10     5  0.933·V5 + 0.360·V3               0.4243     16
if (length(tree_output) > 14) {
  cat("\n... output truncated ...\n")
}
#> 
#> ... output truncated ...
```

## Summary

- Fit a survival forest with
  `RLT(x, y, censor, model = "survival", ...)`.
- Predict with `predict(fit, testx)` to obtain `$Survival`, `$Hazard`,
  and `$CHF`.
- Choose `split.rule` among `"logrank"`, `"suplogrank"`, and
  `"coxgrad"`.
- Use `obs.w` with `split.rule = "coxgrad"` for weighted splits.
- Enable LC splits via
  `param.control = list(linear.comb = k, linear.comb.method = "coxph")`.
- Request variable importance with `importance = TRUE` and inspect via
  `importance(fit)`.
- Estimate prediction variance with `var.mode = "matched"` / `"IJ"` /
  `"jack"`, then call `predict(..., var.est = TRUE)`.
- Build confidence bands with
  `get.surv.band(pred, approach = "naive" or "smoothed")`.
- Inspect trees with `get.one.tree(fit, tree = k)`.
