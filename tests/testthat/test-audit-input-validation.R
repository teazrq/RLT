test_that("survival grid validation precedes both native fitting interfaces", {
  x <- data.frame(a = seq_len(8), b = c(3, 1, 4, 7, 2, 8, 5, 6))
  y <- c(-2, 1, 1, 2, 4, 5, 6, 10)
  censor <- c(0, 1, 0, 1, 1, 0, 1, 0)
  seen <- NULL
  boundary <- function(X, Y, Censor, ...) {
    seen <<- list(Y = Y, Censor = Censor)
    stop("reached native boundary")
  }
  local_mocked_bindings(SurvUniForestFit = boundary,
                       SurvUniCombForestFit = boundary, .package = "RLT")

  for (lc in c(1, 2)) {
    fit <- function(grid, status = censor, times = y) {
      RLT(x, times, censor = status, model = "survival", ntrees = 2,
          ncores = 1, seed = 1, linear.comb = lc,
          param.control = list(time.grid.size = grid))
    }
    for (grid in c(0, 4, 20)) {
      seen <- NULL
      expect_error(fit(grid), "reached native boundary")
      expect_identical(seen$Y, c(0L, 1L, 1L, 2L, 3L, 3L, 4L, 4L))
      expect_true(all(seen$Y[censor == 1] %in% seq_len(4)))
    }
    seen <- NULL
    expect_error(fit(2), "cannot reduce.*band.grid.size")
    expect_null(seen)
    expect_error(fit(0, rep(0, 8)), "at least one observed event")
    expect_null(seen)
    for (grid in list(-1, 1.5, Inf, NA_real_, c(1, 2), numeric(), "2", 1 + 0i))
      expect_error(fit(grid), "single non-negative integer")
    expect_error(fit(0, times = replace(y, 1, Inf)), "finite survival times")
  }
})

test_that("weight normalization is finite, sparse, and scale invariant", {
  checks <- list(function(w) RLT:::check_obsw(w, 3),
                 function(w) RLT:::check_varprob(w, 3))
  for (check in checks) {
    expect_equal(check(c(0, 2, 6)), c(0, .25, .75))
    expect_equal(check(c(0, 2, 6) * 1e300), c(0, .25, .75))
    expect_equal(check(c(0, 2, 6) * 1e-300), c(0, .25, .75))
    expect_equal(check(c(1e308, 1e308, 0)), c(.5, .5, 0))
    for (w in list(rep(0, 3), c(1, Inf, 0), c(1, NA, 0),
                   c(1, NaN, 0), c(1, -1, 0)))
      expect_error(check(w))
    expect_equal(sum(check(c(1, 2, 3))), 1)
  }
})
