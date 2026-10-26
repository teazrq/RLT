# RLT Survival Analysis Tutorial

## Introduction

This vignette introduces survival analysis with **RLT** (Reinforcement
Learning Trees). RLT survival forests estimate survival probabilities
and cumulative hazards for given predictor values, and return hazard
increments on an observed failure-time grid. Key features include:

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
event times and independent censoring. The first two predictors carry
signal; the rest are noise.

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

Predict at selected predictor values to obtain survival probabilities,
hazard increments, and cumulative hazards. Here we use five training
rows to illustrate the returned values; these predictions use all trees
and do not evaluate out-of-sample performance.

``` r

pred <- predict(fit, X[1:5, ])

# Each component is an N x T matrix: N subjects and T observed failure times
str(pred$Survival)      # Survival function S(t)
#>  num [1:5, 1:124] 1 1 0.98 0.985 1 ...
str(pred$Hazard)        # Hazard increments at the failure times
#>  num [1:5, 1:124] 0 0 0.02 0.015 0 0.002 0 0 0 0 ...
str(pred$CHF)           # Cumulative hazard H(t)
#>  num [1:5, 1:124] 0 0 0.02 0.015 0 0.002 0 0.02 0.015 0 ...

# For survival forests, $Prediction is NULL
pred$Prediction
#> NULL
```

The columns follow `pred$timepoints`. On this full grid, cumulative sums
of the hazard increments give the cumulative hazard, and an ordinary
survival forest returns `Survival = exp(-CHF)`.

Plot the predicted survival curve for the first subject:

``` r

plot(pred$timepoints, pred$Survival[1, ], type = "s",
     xlab = "Time", ylab = "Survival Probability",
     main = "Predicted Survival Curve (Subject 1)")
```

![Step curve for subject 1, with time on the horizontal axis and
predicted survival probability on the vertical axis. Survival decreases
in steps, with the steepest decline early in
follow-up.](survival-tutorial_files/figure-html/basic-plot-1.png)

Each downward step lowers the estimated probability that subject 1
survives beyond that time. Flat segments indicate no change in the
forest’s estimate between the displayed failure times. This is one
subject’s predicted curve, rather than a survival curve for the whole
sample.

## Split rules

RLT provides three splitting criteria for survival trees:

| Rule | How it scores a split |
|----|----|
| `logrank` | Standard log-rank test statistic (default) |
| `suplogrank` | Supremum (maximum) of the standardized log-rank process over time |
| `coxgrad` | Gradient of Cox partial likelihood |

The survival `$Error` is one minus the out-of-bag (OOB) concordance
index, so smaller values indicate better risk ordering. Each
observation’s OOB predictions use trees fitted without that observation.
The risk score sums its OOB cumulative hazards over the full
failure-time grid, with larger scores indicating higher risk. This sum
is neither the final cumulative hazard nor a time integral. Concordance
is undefined when no comparable event-time pairs exist.

Fit the three rules on the same data and inspect their OOB errors:

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

The default is `logrank`. This example compares OOB scores on one
simulated dataset; it does not establish which rule is best for other
data.

## Observation weights

Observation weights are passed via `obs.w`. All three survival split
rules use them; weighted `logrank` and `suplogrank` account for weights
in their split-score variance calculation. Weights must be finite and
nonnegative with positive total mass, and do not change how observations
are sampled for each tree. The arbitrary positive weights below
illustrate the syntax; they have no scientific interpretation in this
simulation.

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

Individual zero weights are allowed. See the observation-weight
restrictions in [`RLT()`](https://teazrq.github.io/RLT/reference/RLT.md)
for sampled trees with zero total weight.

## Linear combination splits

When `linear.comb > 1`, a split can combine up to `linear.comb`
variables in one direction. For survival forests, the available methods
are:

- `"coxph"` (default): coefficients from a local Cox model fit.
- `"naive"`: direction based on weighted survival score statistics.

Here `linear.comb = 3` allows up to three variables per split, with a
local Cox fit determining their coefficients:

``` r

fit_lc <- RLT(X, y, censor,
              model = "survival",
              ntrees = 100,
              split.rule = "logrank",
              linear.comb = 3,
              linear.comb.method = "coxph",
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

Predictions from LC forests have the same fields as standard forests.
Their `$Survival` is the mean of the trees’ product-limit survival
curves, which can differ from `exp(-CHF)`:

``` r

pred_lc <- predict(fit_lc, X[1:5, ])
str(pred_lc$Survival)
#>  num [1:5, 1:124] 0.985 0.998 0.993 0.995 1 ...
```

## Variable importance

Set `importance = TRUE` to compute variable importance. Survival
importance measures the decrease in OOB concordance after perturbing a
variable. OOB outcomes with no comparable pairs cannot supply a defined
importance contribution.

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

With matched variance estimation (see next section),
[`importance()`](https://teazrq.github.io/RLT/reference/importance.md)
also reports estimated standard errors, Z-scores, and significance
codes. Importance or its variance can be unavailable when required
comparisons are undefined. See the [variable importance
guide](https://teazrq.github.io/RLT/articles/variable-importance.md) for
interpretation of these quantities.

## Variance estimation and confidence bands

RLT supports three variance estimation strategies for survival
predictions:

- **`"matched"`**: matched-sample U-statistic decomposition for
  prediction and importance variance. Uses paired half-samples without
  replacement and an even number of trees.
- **`"IJ"`**: infinitesimal jackknife.
- **`"jack"`**: jackknife-after-bootstrap.

The example uses matched mode to estimate covariance of the predicted
cumulative hazard over time. Its 1,000 trees form 500 matched pairs,
with `resample.prob = 0.5` and `resample.replace = FALSE` stated
explicitly. Matched mode enforces these resampling settings and reduces
an odd tree count to the preceding even number; at least two trees are
required.

Set `var.mode` during fitting, then request the covariance at prediction
time with `var.est = TRUE`. Variance estimation generally needs more
trees than the small fitting examples above. The IJ and jackknife
alternatives use bootstrap sampling, retain the requested sampling
proportion, and estimate prediction variance only. All active variance
modes retain observation tracking and enable distributed importance when
importance was disabled. See
[`RLT()`](https://teazrq.github.io/RLT/reference/RLT.md) for the
resampling details.

``` r

fit_var <- RLT(X, y, censor,
               model = "survival",
               ntrees = 1000,
               resample.prob = 0.5,
               resample.replace = FALSE,
               var.mode = "matched",
               verbose = FALSE)

# Predict with variance estimation
pred_var <- predict(fit_var, X[1:3, ], var.est = TRUE)

# pred_var$Cov is a T x T x N array: covariance of the cumulative hazard over time
str(pred_var$Cov)

# Marginal standard deviations on the cumulative-hazard scale
sqrt(diag(pred_var$Cov[, , 1]))
```

### Confidence bands with `get.surv.band()`

For given predictor values, a survival confidence band describes
uncertainty about the conditional survival curve, not a prediction
interval for an individual’s event time. With `alpha = 0.05`,
[`get.surv.band()`](https://teazrq.github.io/RLT/reference/get.surv.band.md)
constructs a nominal 95% simultaneous band over the returned time grid,
separately for each requested subject. This does not provide joint
coverage across subjects or continuous-time coverage between grid
points.

The function transforms a cumulative-hazard band with computed reference
curve `exp(-pred_var$CHF)`. For ordinary forests this is also the
returned survival estimate. Linear combination forests instead return a
mean of tree product-limit curves; the CHF-based calculation does not
directly estimate the sampling variation of that alternative estimator.
Two approaches are available:

- **`"naive"`**: uses the full covariance matrix with a Monte Carlo
  critical value.
- **`"smoothed"`**: smooths the covariance and uses a low-rank
  approximation with a residual correction to form marginal
  standardizing scales.

Both approaches simulate with the original supplied covariance. For the
smoothed approach, `k_rank` controls the rank of the approximation used
for the marginal scales. In either approach, `nsim` controls the number
of simulations used to estimate the critical value; increasing it
reduces Monte Carlo variability at the cost of runtime. See the
[confidence-band
tutorial](https://teazrq.github.io/RLT/articles/confidence-interval.md)
and
[`get.surv.band()`](https://teazrq.github.io/RLT/reference/get.surv.band.md)
for rank selection and covariance details.

``` r

# Naive band for the first test subject
band_naive <- get.surv.band(pred_var, i = 1, alpha = 0.05,
                            approach = "naive", nsim = 5000)

# Smoothed band
band_smooth <- get.surv.band(pred_var, i = 1, alpha = 0.05,
                             approach = "smoothed",
                             nsim = 5000, k_rank = 10)

# Plot the reference curve for the CHF-based naive band
t <- band_naive$timepoints
plot(t, exp(-pred_var$CHF[1, ]), type = "s", ylim = c(0, 1),
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

The requested size is a maximum because selected quantiles can coincide.
Cumulative hazards still include increments at all earlier failure
times. Therefore, on a reduced grid, `cumsum(pred_reduced$Hazard[1, ])`
need not equal `pred_reduced$CHF[1, ]`. Bands describe the selected
grid, not simultaneous coverage between omitted times. See
[`predict()`](https://rdrr.io/r/stats/predict.html) for grid-selection
details and [`RLT()`](https://teazrq.github.io/RLT/reference/RLT.md) for
the full failure-time grid retained during fitting.

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
- Use `obs.w` for weighted fitting with any of the three split rules.
- Allow up to `k` variables per LC split with
  `linear.comb = k, linear.comb.method = "coxph"`.
- Request variable importance with `importance = TRUE` and inspect via
  `importance(fit)`.
- Estimate prediction variance with `var.mode = "matched"` / `"IJ"` /
  `"jack"`, then call `predict(..., var.est = TRUE)`.
- Build confidence bands with
  `get.surv.band(pred, approach = "naive" or "smoothed")`.
- Inspect trees with `get.one.tree(fit, tree = k)`.
