kernel_root_fixture <- function(model, lc) {
  track <- rbind(c(0, 1, 2, 1), c(1, 0, 0, 1), c(2, 1, 0, 1), c(0, 1, 1, 1))
  x <- matrix(c(1:4, 11:14), 4, 2)
  args <- list(x = x, y = if (model == "classification") factor(c("A", "A", "B", "B")) else 1:4,
    model = model, ntrees = 4, nmin = 10, linear.comb = lc,
    resample.preset = track, seed = 4, ncores = 1)
  if (model == "survival") args$censor <- rep(1, 4)
  fit <- suppressWarnings(do.call(RLT, args))
  list(fit = fit, x = x, track = track)
}

test_that("training kernels count repeated in-bag observations in every native schema", {
  for (model in c("regression", "classification", "survival")) for (lc in c(1, 2)) {
    d <- kernel_root_fixture(model, lc)
    actual <- forest.kernel(d$fit, d$x[1:2, , drop = FALSE], d$x, vs.train = TRUE)
    expected <- matrix(c(4, 2, 4, 3), 2, 4, byrow = TRUE)
    expect_equal(actual$Kernel, expected)
  }
})

test_that("OOB kernels retain singleton diagonals and preserve the zero-denominator convention", {
  for (model in c("regression", "classification", "survival")) for (lc in c(1, 2)) {
    d <- kernel_root_fixture(model, lc)
    actual <- forest.kernel(d$fit, d$x, oob = TRUE)
    expected <- tcrossprod(d$track == 0)
    expect_equal(actual$N, expected)
    expect_equal(actual$C, expected)
    expect_equal(actual$Kernel, ifelse(expected > 0, 1, 0))
    expect_equal(diag(actual$C), diag(actual$N))
    expect_equal(actual$Kernel[2, 2], 1)
  }
})
