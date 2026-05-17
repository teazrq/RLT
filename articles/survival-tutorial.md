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
  prints individual tree structures.

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
to print the structure of any tree in the fitted forest. This is helpful
for understanding how splits are made and for debugging.

``` r

# Standard (single-variable) survival tree
get.one.tree(fit, tree = 1)
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
#>    11     3  V2                                  1.1210     26
#>    12     4  V1                                 -0.5965     32
#>    13     4  V3                                 -1.8568     79
#>    14     5  V4                                 -1.7762     36
#>    15     5  V1                                 -0.4547     19
#>    16     6  *                                        -      4
#>    17     6  V5                                  1.0575      6
#>    18     7  *                                        -     30
#>    19     7  *                                        -      6
#>    20     6  V3                                  0.4114      1
#>    21     6  V2                                  0.0950     10
#>    22     7  V5                                  0.8194      5
#>    23     7  *                                        -      1
#>    24     8  V1                                 -0.5387      3
#>    25     8  *                                        -      5
#>    26     9  *                                        -      4
#>    27     9  *                                        -      3
#>    28     7  *                                        -      9
#>    29     7  V5                                 -2.3802      9
#>    30     8  *                                        -      1
#>    31     8  V2                                  0.5056      7
#>    32     9  *                                        -      2
#>    33     9  V2                                  0.6554      5
#>    34    10  *                                        -      2
#>    35    10  *                                        -      5
#>    36     5  *                                        -      5
#>    37     5  V3                                 -1.7709     78
#>    38     6  *                                        -      1
#>    39     6  V4                                 -2.0794     77
#>    40     7  *                                        -      1
#>    41     7  V3                                  0.6777     20
#>    42     8  V2                                 -0.3421     31
#>    43     8  V1                                  0.6436      9
#>    44     9  V1                                 -0.0849     25
#>    45     9  V1                                  0.6699     11
#>    46    10  *                                        -      1
#>    47    10  V1                                  0.0867     21
#>    48    11  *                                        -      4
#>    49    11  V3                                  0.0843      3
#>    50    12  V2                                 -1.2221     12
#>    51    12  *                                        -      3
#>    52    13  V3                                 -0.4944      3
#>    53    13  V5                                 -1.0804     11
#>    54    14  *                                        -      3
#>    55    14  *                                        -      3
#>    56    14  *                                        -      1
#>    57    14  V2                                 -1.1489     10
#>    58    15  *                                        -      1
#>    59    15  V3                                 -0.8114      7
#>    60    16  *                                        -      3
#>    61    16  V1                                  1.4518      2
#>    62    17  *                                        -      5
#>    63    17  *                                        -      2
#>    64    10  V3                                 -0.0394      8
#>    65    10  V1                                  1.0314      5
#>    66    11  V4                                  1.1157      3
#>    67    11  V4                                  0.5953      1
#>    68    12  V3                                 -0.1546      1
#>    69    12  *                                        -      3
#>    70    13  V4                                 -0.7547      3
#>    71    13  *                                        -      1
#>    72    14  *                                        -      5
#>    73    14  *                                        -      3
#>    74    12  V1                                  0.1023      5
#>    75    12  *                                        -      1
#>    76    13  *                                        -      2
#>    77    13  *                                        -      5
#>    78    11  V2                                  0.7092      1
#>    79    11  *                                        -      5
#>    80    12  *                                        -      5
#>    81    12  *                                        -      1
#>    82     9  V2                                 -0.2810      7
#>    83     9  V1                                  1.0416      6
#>    84    10  *                                        -      4
#>    85    10  V3                                  1.0710      5
#>    86    11  *                                        -      2
#>    87    11  *                                        -      5
#>    88    10  *                                        -      3
#>    89    10  V1                                  1.3455      3
#>    90    11  *                                        -      3
#>    91    11  *                                        -      3
#>    92     4  *                                        -      4
#>    93     4  V1                                  0.0440     12
#>    94     5  V4                                 -0.1876     11
#>    95     5  V3                                 -0.8275      8
#>    96     6  *                                        -      3
#>    97     6  V4                                  0.2674      8
#>    98     7  *                                        -      3
#>    99     7  V4                                  1.1056      1
#>   100     8  V5                                  0.5966      3
#>   101     8  *                                        -      1
#>   102     9  *                                        -      4
#>   103     9  *                                        -      3
#>   104     6  *                                        -      4
#>   105     6  V3                                  0.0801      2
#>   106     7  V4                                  0.8718      2
#>   107     7  *                                        -      2
#>   108     8  *                                        -      4
#>   109     8  *                                        -      2
```

For LC forests,
[`get.one.tree()`](https://teazrq.github.io/RLT/reference/get.one.tree.md)
also shows the linear combination coefficients at each internal node:

``` r

get.one.tree(fit_lc, tree = 1)
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
#>    11     5  -0.858·V5 + 0.513·V4              0.5935     64
#>    12     6  *                                        -     11
#>    13     6  *                                        -      5
#>    14     6  -0.650·V2 + 0.760·V5              1.1324     48
#>    15     6  0.736·V2 + 0.677·V3               0.9499     16
#>    16     7  0.085·V2 + 0.996·V1              -0.2798     47
#>    17     7  *                                        -      1
#>    18     8  0.958·V5 - 0.287·V4               1.7619     43
#>    19     8  *                                        -      4
#>    20     9  -0.910·V2 + 0.415·V1              0.0870     42
#>    21     9  *                                        -      1
#>    22    10  0.888·V2 + 0.460·V4              -0.2977     37
#>    23    10  *                                        -      5
#>    24    11  *                                        -      5
#>    25    11  0.837·V3 + 0.547·V4              -0.8450     32
#>    26    12  *                                        -      5
#>    27    12  0.599·V4 + 0.800·V3               0.4179     27
#>    28    13  0.221·V4 + 0.975·V1              -0.8746     18
#>    29    13  *                                        -      9
#>    30    14  *                                        -      5
#>    31    14  -0.972·V1 + 0.236·V2              0.7973     13
#>    32    15  2.598·V3 + 6.316·V5               7.3082      8
#>    33    15  *                                        -      5
#>    34    16  2.479·V5 - 0.845·V1               2.1592      6
#>    35    16  *                                        -      2
#>    36    17  *                                        -      3
#>    37    17  *                                        -      3
#>    38     7  -0.975·V4 + 0.220·V3             -0.6725     15
#>    39     7  *                                        -      1
#>    40     8  0.087·V4 + 0.882·V1              -0.0437      9
#>    41     8  *                                        -      6
#>    42     9  1.136·V2 + 0.480·V4               0.4491      8
#>    43     9  *                                        -      1
#>    44    10  *                                        -      1
#>    45    10  -2.533·V5 - 0.352·V2              2.0573      7
#>    46    11  *                                        -      4
#>    47    11  *                                        -      3
#>    48     5  0.998·V3 - 0.063·V5               1.2621     19
#>    49     5  -9.219·V4 + 1.942·V5             -4.7201      9
#>    50     6  0.677·V2 - 0.736·V5               0.2501     18
#>    51     6  *                                        -      1
#>    52     7  *                                        -      9
#>    53     7  -0.045·V1 - 0.457·V5              0.2922      9
#>    54     8  *                                        -      2
#>    55     8  -0.404·V1 - 1.979·V5              1.5049      7
#>    56     9  *                                        -      2
#>    57     9  *                                        -      5
#>    58     6  *                                        -      3
#>    59     6  -1.975·V5 + 5.053·V4             -6.6797      6
#>    60     7  *                                        -      2
#>    61     7  *                                        -      4
#>    62     4  0.999·V2 - 0.051·V5              -0.7297     65
#>    63     4  0.538·V1 - 0.843·V3               0.9156     18
#>    64     5  0.560·V3 + 0.828·V1               1.6763     20
#>    65     5  -1.000·V3 + 0.018·V5              1.8963     45
#>    66     6  -0.994·V4 + 0.107·V3             -0.4150     18
#>    67     6  *                                        -      2
#>    68     7  *                                        -      6
#>    69     7  -0.999·V5 + 0.033·V3             -1.4621     12
#>    70     8  *                                        -      3
#>    71     8  -12.869·V3 - 1.775·V5           -11.7087      9
#>    72     9  *                                        -      2
#>    73     9  9.529·V5 + 1.216·V1               0.7138      7
#>    74    10  *                                        -      4
#>    75    10  *                                        -      3
#>    76     6  -1.000·V3 + 0.016·V5              0.1564     44
#>    77     6  *                                        -      1
#>    78     7  0.994·V4 + 0.108·V2              -0.6632     22
#>    79     7  1.000·V3 - 0.015·V5              -0.5938     22
#>    80     8  *                                        -      5
#>    81     8  -0.622·V1 + 0.783·V4              0.3575     17
#>    82     9  -0.767·V3 + 0.641·V4             -1.3116     16
#>    83     9  *                                        -      1
#>    84    10  *                                        -      1
#>    85    10  0.989·V5 + 0.150·V1               0.3125     15
#>    86    11  0.306·V2 + 0.952·V5              -0.1039     10
#>    87    11  *                                        -      5
#>    88    12  *                                        -      4
#>    89    12  1.350·V1 - 1.472·V5               1.2054      6
#>    90    13  *                                        -      3
#>    91    13  *                                        -      3
#>    92     8  0.032·V2 - 3.811·V5              -8.5675      6
#>    93     8  0.971·V3 + 0.239·V4              -0.4098     16
#>    94     9  *                                        -      1
#>    95     9  *                                        -      5
#>    96     9  *                                        -      5
#>    97     9  1.000·V3 - 0.005·V5              -0.2166     11
#>    98    10  -1.392·V4 + 0.154·V1             -0.5013      8
#>    99    10  *                                        -      3
#>   100    11  *                                        -      2
#>   101    11  -0.201·V3 - 3.591·V5             -7.2022      6
#>   102    12  *                                        -      2
#>   103    12  *                                        -      4
#>   104     5  0.578·V4 + 0.816·V2               1.9344     15
#>   105     5  *                                        -      3
#>   106     6  0.511·V4 + 0.859·V3               0.0407     14
#>   107     6  *                                        -      1
#>   108     7  0.629·V1 + 0.479·V2               1.0523      6
#>   109     7  -1.803·V1 - 1.835·V3             -0.9561      8
#>   110     8  *                                        -      1
#>   111     8  *                                        -      5
#>   112     8  1.857·V5 - 0.995·V2               0.4455      7
#>   113     8  *                                        -      1
#>   114     9  *                                        -      4
#>   115     9  *                                        -      3
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
