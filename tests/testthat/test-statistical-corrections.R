# Deterministic forest representations isolate prediction/variance algebra from
# splitting and sampling. They are numerical fixtures, not coverage experiments.
stat_root_fixture <- function(model = "survival", linear.comb = 1, nfail = 12) {
  n <- if (model == "survival") 2L * nfail else 12L
  x <- cbind(x1 = seq_len(n), x2 = rev(seq_len(n)))
  y <- switch(model,
    survival = rep(seq_len(nfail), each = 2L),
    regression = seq_len(n),
    classification = factor(rep(c("a", "b"), length.out = n)))
  args <- list(x = x, y = y, model = model, ntrees = 8L, nmin = n + 1L,
    linear.comb = linear.comb, resample.preset = matrix(1L, n, 8L),
    resample.replace = FALSE, resample.prob = 1,
    param.control = list(resample.track = TRUE), seed = 51, ncores = 1)
  if (model == "survival") args$censor <- rep(1, n)
  list(fit = do.call(RLT, args), x = x[1:2, , drop = FALSE])
}

stat_count_fixture <- function(n) {
  counts <- rbind(c(0, 0, 0, 0, 2, 2, 2, 2),
                  c(0, 0, 1, 1, 0, 0, 1, 1),
                  c(0, 1, 0, 1, 0, 1, 0, 2),
                  matrix(1L, n - 3L, 8L))
  storage.mode(counts) <- "integer"
  counts
}

stat_ij_reference <- function(pred, counts) {
  counts <- pmax(counts, 0) * 1.0
  b <- nrow(pred)
  pc <- sweep(pred, 2L, colMeans(pred))
  nc <- sweep(counts, 1L, rowMeans(counts))
  influence <- nc %*% pc / b
  raw <- crossprod(influence)
  count_var <- sum(rowMeans(counts^2) - rowMeans(counts)^2)
  bias <- count_var * crossprod(pc) / b^2
  list(corrected = raw - bias, bias = bias)
}

stat_cov_reference <- function(pred, counts, mode) {
  b <- nrow(pred)
  pc <- sweep(pred, 2L, colMeans(pred))
  if (mode == 1L) {
    half <- b %/% 2L
    delta <- pred[seq_len(half), , drop = FALSE] -
      pred[half + seq_len(half), , drop = FALSE]
    raw <- crossprod(delta) / b - crossprod(pc) / b
  } else if (mode == 2L) {
    raw <- stat_ij_reference(pred, counts)$corrected
  } else {
    oob <- counts == 0L
    valid <- which(rowSums(oob) > 0L)
    jack <- do.call(rbind, lapply(valid, function(i)
      colMeans(pred[oob[i, ], , drop = FALSE])))
    delta <- sweep(jack, 2L, colMeans(pred))
    nv <- length(valid)
    raw <- (nv - 1) / nv * crossprod(delta) -
      (exp(1) - 1) * nv / b^2 * crossprod(pc)
  }
  eg <- eigen(raw, symmetric = TRUE)
  projected <- eg$vectors %*% diag(pmax(eg$values, 1e-6), nrow(raw)) %*%
    t(eg$vectors)
  (projected + t(projected)) / 2
}

test_that("survival accumulation precedes grid restriction", {
  for (lc in c(1L, 2L)) {
    d <- stat_root_fixture(linear.comb = lc, nfail = 6L)
    full <- predict(d$fit, d$x, keep.all = TRUE)
    hazard <- 1 / (6:1)
    expect_equal(as.numeric(full$Hazard[1, ]), hazard, tolerance = 1e-12)
    expect_equal(as.numeric(full$CHF[1, ]), cumsum(hazard), tolerance = 1e-12)
    expect_equal(full$CHF[1, 6], 2.45, tolerance = 1e-12)
    expected_surv <- if (lc == 1L) exp(-cumsum(hazard)) else cumprod(1 - hazard)
    expect_equal(as.numeric(full$Survival[1, ]), expected_surv, tolerance = 1e-12)
    for (grid in c(1L, 2L, 8L)) {
      reduced <- predict(d$fit, d$x, band.grid.size = grid, keep.all = TRUE)
      ix <- match(reduced$timepoints, full$timepoints)
      expect_equal(reduced$CHF, full$CHF[, ix, drop = FALSE], tolerance = 1e-12)
      expect_equal(reduced$Survival, full$Survival[, ix, drop = FALSE], tolerance = 1e-12)
      expect_equal(reduced$Hazard, full$Hazard[, ix, drop = FALSE], tolerance = 1e-12)
      expect_equal(reduced$AllHazard, full$AllHazard[, ix, , drop = FALSE], tolerance = 1e-12)
    }
  }
})

test_that("all survival covariance modes use selected full-grid tree CHF", {
  for (lc in c(1L, 2L)) {
    d <- stat_root_fixture(linear.comb = lc)
    counts <- stat_count_fixture(d$fit$parameters$n)
    hazards <- outer(c(.02, .03, .04, .05, .15, .16, .17, .18), seq_len(12) / 12) +
      outer(c(.01, .02, .01, .02, .01, .02, .01, .03), rev(seq_len(12)) / 12)
    for (b in seq_len(8L))
      d$fit$FittedForest$NodeHaz[[b]][[1]] <- c(0, hazards[b, ])
    tree_chf <- t(apply(hazards, 1L, cumsum))
    tree_surv <- t(apply(1 - hazards, 1L, cumprod))
    d$fit$ObsTrack <- counts
    for (mode in 1:3) {
      d$fit$parameters$var.mode <- as.integer(mode)
      for (grid in c(0L, 1L, 2L, 8L)) {
        pr <- predict(d$fit, d$x, var.est = TRUE, keep.all = TRUE,
                      band.grid.size = grid)
        plain <- predict(d$fit, d$x, band.grid.size = grid)
        ix <- match(pr$timepoints, d$fit$timepoints)
        selected <- tree_chf[, ix, drop = FALSE]
        reference <- stat_cov_reference(selected, counts, mode)
        expect_equal(pr$CHF, plain$CHF, tolerance = 1e-12)
        expect_equal(pr$Hazard, plain$Hazard, tolerance = 1e-12)
        expect_equal(pr$Survival, plain$Survival, tolerance = 1e-12)
        expect_null(plain$AllHazard)
        expect_equal(dim(pr$AllHazard), c(8L, length(ix), 2L))
        for (subject in 1:2) {
          expect_equal(matrix(pr$AllHazard[, , subject], 8L, length(ix)),
                       hazards[, ix, drop = FALSE], tolerance = 1e-12)
          expect_equal(as.numeric(pr$Hazard[subject, ]),
                       colMeans(hazards[, ix, drop = FALSE]), tolerance = 1e-12)
          expect_equal(as.numeric(pr$CHF[subject, ]), colMeans(selected), tolerance = 1e-12)
          expected_surv <- if (lc == 1L) exp(-colMeans(selected)) else
            colMeans(tree_surv[, ix, drop = FALSE])
          expect_equal(as.numeric(pr$Survival[subject, ]), expected_surv, tolerance = 1e-12)
          expect_equal(matrix(pr$Cov[, , subject], length(ix), length(ix)),
                       reference, tolerance = 1e-10)
        }
        if (mode == 2L && length(ix) > 1L) {
          ij <- stat_ij_reference(selected, counts)
          expect_gt(max(diag(ij$bias)), 0)
          expect_gt(max(diag(ij$corrected)), 0)
          expect_gt(max(abs(ij$corrected[row(ij$corrected) != col(ij$corrected)])), 0)
        }
      }
    }
  }
})

test_that("regression and classification IJ use fractional count moments", {
  values <- c(.1, .1, .2, .2, .8, .8, .9, .9)
  for (model in c("regression", "classification")) {
    for (lc in c(1L, 2L)) {
      d <- stat_root_fixture(model, lc)
      counts <- stat_count_fixture(d$fit$parameters$n)
      d$fit$ObsTrack <- counts
      d$fit$parameters$var.mode <- 2L
      for (b in seq_len(8L)) {
        if (model == "regression") d$fit$FittedForest$NodeAve[[b]][1] <- values[b]
        else d$fit$FittedForest$NodeProb[[b]][1, ] <- c(values[b], 1 - values[b])
      }
      tree_pred <- if (model == "regression") matrix(values, ncol = 1L) else
        cbind(values, 1 - values)
      ij <- stat_ij_reference(tree_pred, counts)
      expect_true(any(rowMeans(counts) != floor(rowMeans(counts))))
      expect_gt(max(diag(ij$bias)), 0)
      expect_gt(min(diag(ij$corrected)), 0)
      pr <- predict(d$fit, d$x, var.est = TRUE, keep.all = TRUE)
      plain <- predict(d$fit, d$x)
      if (model == "regression") {
        expect_equal(as.numeric(pr$Variance), rep(ij$corrected[1, 1], 2L), tolerance = 1e-12)
        expect_equal(pr$Prediction, plain$Prediction)
      } else {
        expect_equal(pr$Variance, matrix(rep(diag(ij$corrected), each = 2L), 2L), tolerance = 1e-12)
        expect_equal(pr$Prob, plain$Prob)
      }
      for (b in seq_len(8L)) {
        if (model == "regression") d$fit$FittedForest$NodeAve[[b]][1] <- .25
        else d$fit$FittedForest$NodeProb[[b]][1, ] <- c(.25, .75)
      }
      constant <- predict(d$fit, d$x, var.est = TRUE)
      expect_true(all(constant$Variance == 0))
    }
  }
})

test_that("constant survival tree predictions retain the covariance eigenvalue floor", {
  for (lc in c(1L, 2L)) {
    d <- stat_root_fixture(linear.comb = lc)
    d$fit$ObsTrack <- stat_count_fixture(d$fit$parameters$n)
    for (mode in 1:3) {
      d$fit$parameters$var.mode <- as.integer(mode)
      for (grid in c(1L, 8L)) {
        pr <- predict(d$fit, d$x, var.est = TRUE, band.grid.size = grid)
        p <- length(pr$timepoints)
        for (subject in 1:2)
          expect_equal(matrix(pr$Cov[, , subject], p, p), diag(1e-6, p), tolerance = 1e-12)
      }
    }
  }
})

stat_band_fixture <- function(n, subjects = 1L, zero = FALSE) {
  sigma <- if (zero) matrix(0, n, n) else outer(seq_len(n), seq_len(n),
    function(i, j) .02 * exp(-abs(i - j) / 3))
  obj <- list(Cov = array(rep(sigma, subjects), c(n, n, subjects)),
    CHF = matrix(rep(seq(.1, 1, length.out = n), each = subjects), subjects),
    timepoints = seq_len(n))
  class(obj) <- c("RLT", "pred", "surv")
  obj
}

test_that("one/two-time smoothing falls back to the identical naive calculation", {
  for (n in 1:2) for (subjects in c(1L, 3L)) {
    obj <- stat_band_fixture(n, subjects)
    set.seed(79)
    naive <- get.surv.band(obj, nsim = 100L)
    for (rank_mode in c("fixed", "proportion")) {
      set.seed(79)
      expect_warning(smoothed <- get.surv.band(obj, approach = "smoothed",
        k_mode = rank_mode, nsim = 100L), "using approach = 'naive'")
      expect_identical(smoothed, naive)
    }
    for (subject in seq_len(subjects)) {
      expect_equal(dim(naive[[subject]]$lower), c(n, 1L))
      expect_true(all(is.finite(naive[[subject]]$lower)))
      expect_equal((log(naive[[subject]]$lower) + log(naive[[subject]]$upper)) / 2,
                   matrix(-obj$CHF[subject, ], n, 1L), tolerance = 1e-12)
    }
    zero <- get.surv.band(stat_band_fixture(n, subjects, zero = TRUE), nsim = 20L)
    for (subject in seq_len(subjects))
      expect_identical(zero[[subject]]$diagnostics$var_explained$var_explained, NA_real_)
  }
})

test_that("smoothing has feasible bases and explicit zero-variance diagnostics", {
  skip_if_not_installed("mgcv")
  for (n in c(3L, 8L, 12L)) for (subjects in c(1L, 3L)) {
    obj <- stat_band_fixture(n, subjects)
    for (rank_mode in c("fixed", "proportion")) {
      set.seed(17)
      band <- get.surv.band(obj, approach = "smoothed", k_mode = rank_mode,
                           k_rank = n + 5L, nsim = 100L)
      for (subject in seq_len(subjects)) {
        expect_equal(dim(band[[subject]]$lower), c(n, 1L))
        expect_true(all(is.finite(c(band[[subject]]$lower, band[[subject]]$upper))))
        expect_true(all(band[[subject]]$lower <= band[[subject]]$upper))
        expect_lte(band[[subject]]$diagnostics$var_explained$k_rank, n)
      }
    }
    zero <- get.surv.band(stat_band_fixture(n, subjects, zero = TRUE),
      approach = "smoothed", k_mode = "proportion", nsim = 20L)
    for (subject in seq_len(subjects)) {
      expect_identical(zero[[subject]]$diagnostics$var_explained$var_explained, NA_real_)
      expect_identical(zero[[subject]]$diagnostics$eig$k_used, 0)
      expect_identical(zero[[subject]]$diagnostics$eig$var_explained, NA_real_)
      expect_true(all(is.finite(c(zero[[subject]]$lower, zero[[subject]]$upper))))
    }
  }
  obj <- stat_band_fixture(3L)
  for (bad in list(-1, .5, NA_real_, Inf, c(1, 2)))
    expect_error(get.surv.band(obj, approach = "smoothed", k_rank = bad), "k_rank")
  for (bad in list(0, 1.1, NA_real_, Inf, c(.5, .8)))
    expect_error(get.surv.band(obj, approach = "smoothed", k_mode = "proportion",
                             k_prop = bad), "k_prop")
})

test_that("smoothing preserves the raw covariance Gaussian simulation", {
  skip_if_not_installed("mgcv")
  obj <- stat_band_fixture(3L)
  observed <- NULL
  band_function <- get.surv.band
  environment(band_function) <- new.env(parent = environment(get.surv.band))
  environment(band_function)$mc_band <- function(mar_sd, S, alpha, N) {
    observed <<- S
    matrix(0, length(mar_sd), length(alpha))
  }
  result <- band_function(obj, approach = "smoothed", k_rank = 2L)
  expect_equal(observed, obj$Cov[, , 1L])
  expect_equal(result[[1L]]$lower, matrix(exp(-obj$CHF[1L, ]), 3L))
  expect_equal(result[[1L]]$upper, result[[1L]]$lower)
})
