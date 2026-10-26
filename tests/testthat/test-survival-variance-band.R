# test-survival-variance-band.R

context("Survival - Variance, VarVI, keep.all, Validation, Bands")

test_that("matched variance: Cov is 3D array with correct dimensions", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "matched"))
  pred <- predict(fit, d$X, var.est = TRUE)
  expect_is(pred$Cov, "array")
  expect_equal(length(dim(pred$Cov)), 3)
})

test_that("IJ variance: Cov is 3D array", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "IJ"))
  pred <- predict(fit, d$X, var.est = TRUE)
  expect_is(pred$Cov, "array")
  expect_equal(length(dim(pred$Cov)), 3)
})

test_that("jack variance: Cov is 3D array with correct dimensions", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "jack"))
  pred <- predict(fit, d$X, var.est = TRUE)
  expect_is(pred$Cov, "array")
  expect_equal(length(dim(pred$Cov)), 3)
})

test_that("Covariance matrices are symmetric", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "matched"))
  pred <- predict(fit, d$X, var.est = TRUE)
  for (i in 1:min(5, dim(pred$Cov)[3])) {
    mat <- pred$Cov[, , i]
    expect_equal(mat[lower.tri(mat)], t(mat)[lower.tri(mat)])
  }
})

test_that("Different variance modes produce different Cov matrices", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit_m <- RLT(d$X, d$y, censor = d$censor, model = "survival",
               ntrees = 30, seed = 1, ncores = 2, verbose = FALSE,
               param.control = list(var.mode = "matched"))
  fit_j <- RLT(d$X, d$y, censor = d$censor, model = "survival",
               ntrees = 30, seed = 1, ncores = 2, verbose = FALSE,
               param.control = list(var.mode = "jack"))
  pred_m <- predict(fit_m, d$X, var.est = TRUE)
  pred_j <- predict(fit_j, d$X, var.est = TRUE)
  expect_false(identical(as.vector(pred_m$Cov), as.vector(pred_j$Cov)))
})

test_that("var.est=FALSE returns no Cov", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "matched"))
  pred <- predict(fit, d$X, var.est = FALSE)
  expect_null(pred$Cov)
})

test_that("band.grid.size reduces timepoints length", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit_full <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "matched"))
  pred_full <- predict(fit_full, d$X, var.est = TRUE)
  full_tp <- ncol(pred_full$Survival)
  
  pred_sml <- predict(fit_full, d$X, var.est = TRUE, band.grid.size = 5)
  expect_lt(ncol(pred_sml$Survival), full_tp)
  expect_lte(ncol(pred_sml$Survival), 5)
  ids <- match(pred_sml$timepoints, pred_full$timepoints)
  expect_equal(pred_sml$CHF, pred_full$CHF[, ids, drop = FALSE])
})

test_that("VarVI returned for survival matched mode", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, importance = TRUE, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "matched"))
  expect_true("VarVI" %in% names(fit))
  expect_length(fit$VarVI, d$p)
})

test_that("importance() shows SD/Z/Sig for survival matched", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, importance = TRUE, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "matched"))
  imp <- importance(fit)
  expect_true("SD" %in% colnames(imp))
  expect_true("Z" %in% colnames(imp))
  expect_true("Sig" %in% colnames(imp))
})

test_that("VarVI not returned for IJ mode in survival", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, importance = TRUE, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "IJ"))
  expect_null(fit$VarVI)
})

test_that("VarVI not returned for jack mode in survival", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, importance = TRUE, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "jack"))
  expect_null(fit$VarVI)
})

test_that("keep.all is accepted without error", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  expect_error(
    {
      fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
                 ntrees = 30, ncores = 2, verbose = FALSE,
                 param.control = list(var.mode = "matched"))
      predict(fit, d$X[1:2, , drop = FALSE], keep.all = TRUE)
    },
    NA
  )
})

test_that("keep.all stores per-tree hazards in the prediction", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "matched"))
  pred <- predict(fit, d$X[1:2, , drop = FALSE], keep.all = TRUE)
  expect_equal(dim(pred$AllHazard), c(30, length(pred$timepoints), 2))
  expect_equal(t(apply(pred$AllHazard, c(2, 3), mean)), pred$Hazard)
})

test_that("NA in y produces error", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  d$y[5] <- NA
  expect_error(
    RLT(d$X, d$y, censor = d$censor, model = "survival",
        ntrees = 30, ncores = 2, verbose = FALSE,
        param.control = list(var.mode = "matched"))
  )
})

test_that("NA in censor produces error", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  censor_na <- d$censor
  censor_na[5] <- NA
  expect_error(
    RLT(d$X, d$y, censor = censor_na, model = "survival",
        ntrees = 30, ncores = 2, verbose = FALSE,
        param.control = list(var.mode = "matched"))
  )
})

test_that("Mismatched dimensions produce error", {
  set.seed(42)
  X <- matrix(rnorm(80 * 5), 80, 5)
  y <- rexp(50)
  censor <- rep(1, 50)
  expect_error(
    RLT(X, y, censor = censor, model = "survival",
        ntrees = 30, ncores = 2, verbose = FALSE)
  )
})

test_that("Invalid censor values produce error", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  censor_bad <- sample(c(0, 1, 2), d$n, replace = TRUE)
  expect_error(
    RLT(d$X, d$y, censor = censor_bad, model = "survival",
        ntrees = 30, ncores = 2, verbose = FALSE)
  )
})

test_that("get.surv.band returns valid structure", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "matched"))
  pred <- predict(fit, d$X, var.est = TRUE)
  
  band <- get.surv.band(pred, i = 1, alpha = 0.1, nsim = 50)
  expect_s3_class(band, "RLT")
  expect_true("Subject1" %in% names(band))
  expect_true("timepoints" %in% names(band))
  expect_true("lower" %in% names(band$Subject1))
  expect_true("upper" %in% names(band$Subject1))
})

test_that("Confidence band contains survival curve", {
  d <- generate_survival_data(n = 80, p = 10, seed = 1)
  fit <- RLT(d$X, d$y, censor = d$censor, model = "survival",
             ntrees = 30, ncores = 2, verbose = FALSE,
             param.control = list(var.mode = "matched"))
  pred <- predict(fit, d$X, var.est = TRUE)
  
  subject_idx <- 1
  band <- get.surv.band(pred, i = subject_idx, alpha = 0.1, nsim = 50)
  s1 <- band$Subject1
  n_grid <- length(s1$lower)
  surv <- pred$Survival[subject_idx, 1:n_grid]
  within_band <- sum(surv >= s1$lower & surv <= s1$upper)
  expect_gt(within_band, 0)
})
