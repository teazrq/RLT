single_time_logrank <- function(x, event, cut) {
  left <- x <= cut
  n <- length(event)
  failures <- sum(event)
  left_n <- sum(left)
  expected <- failures * left_n / n
  variance <- expected * (1 - failures / n) * (n - left_n) / (n - 1)
  (sum(event[left]) - expected)^2 / variance
}

single_time_best_logrank <- function(x, event) {
  values <- sort(unique(x))
  cuts <- (head(values, -1L) + tail(values, -1L)) / 2
  max(vapply(cuts, function(cut) single_time_logrank(x, event, cut), numeric(1)))
}

test_that("LC survival preserves failures before censoring within tied times", {
  k <- 0:31
  x <- cbind(k, k %% 7L + .01 * k)
  event <- c(1L, 1L, rep(0L, 30L))
  # Independent Cox score at beta=0 and marginal logrank screening scores.
  score <- colSums(x[event == 1L, , drop = FALSE]) - sum(event) * colMeans(x)
  screening <- vapply(seq_len(ncol(x)), function(j) single_time_best_logrank(x[, j], event), numeric(1))
  expected_load <- score * sqrt(screening)
  for (cores in 1:2) {
    fit <- RLT(x, rep(1, 32L), censor = event, model = "survival",
      ntrees = 2L, mtry = 2L, nmin = 31L, linear.comb = 2L,
      linear.comb.method = "naive", resample.replace = FALSE,
      resample.prob = 1, ncores = cores, seed = 7L)
    forest <- fit$FittedForest
    for (tree in seq_along(forest$SplitVar)) {
      variables <- forest$SplitVar[[tree]][1L, ] + 1L
      load <- forest$SplitLoad[[tree]][1L, ]
      expect_equal(nrow(forest$SplitVar[[tree]]), 3L)
      expect_setequal(as.integer(variables), 1:2)
      expect_equal(sum(load != 0), 2L)
      expect_equal(as.vector(load), as.vector(expected_load[variables]), tolerance = 1e-12)
      projected <- as.vector(x[, variables, drop = FALSE] %*% load)
      chosen <- single_time_logrank(projected, event, forest$SplitValue[[tree]][1L])
      expect_equal(chosen, single_time_best_logrank(projected, event), tolerance = 1e-12)
      expect_true(all(is.finite(unlist(forest$NodeHaz[[tree]]))))
    }
  }
})

test_that("weighted Gini ties retain the reference split order", {
  k <- 0:31
  x <- cbind(k %% 3L, k %% 4L)
  y <- factor(rep("a", 32L), levels = c("a", "b"))
  for (cores in 1:2) {
    # A pure node has equivalent predictions under either partition, but
    # unstable index sorting changes weighted accumulation and the tie winner.
    fit <- suppressWarnings(RLT(x, y, model = "classification", ntrees = 2L,
      mtry = 2L, nmin = 31L, obs.w = 1 + k %% 5L, resample.replace = FALSE,
      resample.prob = 1, ncores = cores, seed = 7L))
    # Fix native input doubles independently of public weight normalization.
    weights <- as.numeric(1 + k %% 5L) / 93
    forest <- RLT:::ClaUniForestFit(x, rep(0L, 32L), rep(1L, 2L), 2L,
      weights, numeric(), matrix(integer(), 0L, 0L), fit$parameters)$FittedForest
    for (tree in seq_along(forest$SplitVar)) {
      expect_equal(as.vector(forest$SplitVar[[tree]]), c(0L, -1L, -1L))
      expect_equal(forest$SplitValue[[tree]][1L], 1.5)
      expect_equal(forest$NodeProb[[tree]], cbind(rep(1, 3L), rep(0, 3L)))
    }
  }
})
