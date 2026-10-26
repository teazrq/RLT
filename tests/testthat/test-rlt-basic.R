# test-rlt-basic.R
# Tests for Reinforcement Learning Trees (embedded model) feature

context("RLT Basic Tests (Reinforcement Learning)")

test_that("RLT identifies x1 and x2 as most important in egg tray model", {
  set.seed(42)
  n <- 80
  p_noise <- 8
  
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  X_noise <- matrix(rnorm(n * p_noise), n, p_noise)
  X <- cbind(x1, x2, X_noise)
  colnames(X) <- c("x1", "x2", paste0("noise", 1:p_noise))
  y <- x1 * x2 + rnorm(n, sd = 0.5)
  
  fit <- RLT(X, y, ntrees = 30, ncores = 2, verbose = FALSE,
             reinforcement = TRUE, importance = TRUE,
             param.control = list(embed.ntrees = 20))
  
  top2 <- colnames(X)[order(fit$VarImp, decreasing = TRUE)[1:2]]
  expect_true("x1" %in% top2)
  expect_true("x2" %in% top2)
})

test_that("RLT muting reduces noise variable consideration", {
  set.seed(42)
  n <- 300
  p_noise <- 8
  
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  X_noise <- matrix(rnorm(n * p_noise), n, p_noise)
  X <- cbind(x1, x2, X_noise)
  colnames(X) <- c("x1", "x2", paste0("noise", 1:p_noise))
  y <- x1 * x2 + rnorm(n, sd = 0.5)
  
  fit <- RLT(X, y, ntrees = 100, ncores = 2, verbose = FALSE,
             reinforcement = TRUE, importance = TRUE,
             param.control = list(embed.ntrees = 50, embed.mtry = 2))
  
  # Signal VI should be higher than noise mean (with enough data)
  expect_true(all(!is.na(fit$VarImp)))
  signal_vi <- fit$VarImp[1:2]
  noise_vi <- fit$VarImp[3:10]
  expect_gt(mean(signal_vi), mean(noise_vi))
})
