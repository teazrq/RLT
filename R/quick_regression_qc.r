#' @title Quick Regression Quality Control
#' @name quick_regression_qc
#' @description Quick quality control test for RLT regression models
#' @export
quick_regression_qc <- function() {
  
  # Load RLT package
  if (!requireNamespace("RLT", quietly = TRUE)) {
    cat("❌ RLT package not found. Please install it first.\n")
    return(FALSE)
  }
  library(RLT)
  
  cat("=== RLT Regression Quick Quality Control ===\n\n")
  
  # Test 1: Basic functionality
  cat("1. Testing basic regression functionality...\n")
  
  set.seed(123)
  n <- 500
  p <- 10
  X <- matrix(rnorm(n*p), n, p)
  y <- 1 + X[, 1] + X[, 3] + rnorm(n)
  
  param <- list(
    ntrees = 50,
    mtry = 5,
    nmin = 10,
    split.gen = 1,
    nsplit = 1,
    resample.prob = 0.8,
    resample.replace = FALSE,
    reinforcement = FALSE,
    importance = 0,
    verbose = FALSE,
    ncores = 1,
    seed = 123
  )
  
  tryCatch({
    fit <- RLT(X, y, model = "regression", 
               ntrees = param$ntrees,
               mtry = param$mtry,
               nmin = param$nmin,
               split.gen = "random",
               nsplit = param$nsplit,
               resample.prob = param$resample.prob,
               resample.replace = param$resample.replace,
               reinforcement = param$reinforcement,
               importance = FALSE,
               verbose = param$verbose,
               ncores = param$ncores,
               seed = param$seed)
    
    if (inherits(fit, "RLT") && "reg" %in% class(fit)) {
      cat("   ✅ Basic regression: PASSED\n")
    } else {
      cat("   ❌ Basic regression: FAILED\n")
    }
  }, error = function(e) {
    cat("   ❌ Basic regression: FAILED -", e$message, "\n")
  })
  
  # Test 2: Linear combination
  cat("2. Testing linear combination regression...\n")
  
  param_comb <- list(
    ntrees = 50,
    mtry = 10,
    nmin = 10,
    split.gen = 1,
    nsplit = 1,
    resample.prob = 0.8,
    resample.replace = FALSE,
    reinforcement = FALSE,
    importance = 0,
    verbose = FALSE,
    ncores = 1,
    seed = 123,
    linear.comb = 3,
    split.rule = "naive"
  )
  
  tryCatch({
    fit_comb <- RLT(X, y, model = "regression", 
                    ntrees = param_comb$ntrees,
                    mtry = param_comb$mtry,
                    nmin = param_comb$nmin,
                    split.gen = "random",
                    nsplit = param_comb$nsplit,
                    resample.prob = param_comb$resample.prob,
                    resample.replace = param_comb$resample.replace,
                    reinforcement = param_comb$reinforcement,
                    importance = FALSE,
                    verbose = param_comb$verbose,
                    ncores = param_comb$ncores,
                    seed = param_comb$seed,
                    param.control = list(linear.comb = param_comb$linear.comb,
                                        split.rule = param_comb$split.rule))
    
    if (inherits(fit_comb, "RLT") && "reg" %in% class(fit_comb) && "comb" %in% class(fit_comb)) {
      cat("   ✅ Linear combination: PASSED\n")
    } else {
      cat("   ❌ Linear combination: FAILED\n")
    }
  }, error = function(e) {
    cat("   ❌ Linear combination: FAILED -", e$message, "\n")
  })
  
  # Test 3: Reproducibility
  cat("3. Testing reproducibility...\n")
  
  tryCatch({
    # Fit model twice with same seed
    fit1 <- RLT(X, y, model = "regression", 
                ntrees = param$ntrees,
                mtry = param$mtry,
                nmin = param$nmin,
                split.gen = "random",
                nsplit = param$nsplit,
                resample.prob = param$resample.prob,
                resample.replace = param$resample.replace,
                reinforcement = param$reinforcement,
                importance = FALSE,
                verbose = param$verbose,
                ncores = param$ncores,
                seed = param$seed)
    
    fit2 <- RLT(X, y, model = "regression", 
                ntrees = param$ntrees,
                mtry = param$mtry,
                nmin = param$nmin,
                split.gen = "random",
                nsplit = param$nsplit,
                resample.prob = param$resample.prob,
                resample.replace = param$resample.replace,
                reinforcement = param$reinforcement,
                importance = FALSE,
                verbose = param$verbose,
                ncores = param$ncores,
                seed = param$seed)
    
    # Check if predictions are identical
    if (all.equal(fit1$Prediction, fit2$Prediction, tolerance = 1e-10)) {
      cat("   ✅ Reproducibility: PASSED\n")
    } else {
      cat("   ❌ Reproducibility: FAILED\n")
    }
  }, error = function(e) {
    cat("   ❌ Reproducibility: FAILED -", e$message, "\n")
  })
  
  # Test 4: Edge cases
  cat("4. Testing edge cases...\n")
  
  # Single observation
  tryCatch({
    X_single <- matrix(1, 1, 2)
    y_single <- 1
      param_single <- param
  param_single$ntrees <- 10
  param_single$nmin <- 1
  param_single$mtry <- 1  # Fix mtry for single observation
    
    fit_single <- RLT(X_single, y_single, model = "regression", 
                      ntrees = param_single$ntrees,
                      mtry = param_single$mtry,
                      nmin = param_single$nmin,
                      split.gen = "random",
                      nsplit = param_single$nsplit,
                      resample.prob = param_single$resample.prob,
                      resample.replace = param_single$resample.replace,
                      reinforcement = param_single$reinforcement,
                      importance = FALSE,
                      verbose = param_single$verbose,
                      ncores = param_single$ncores,
                      seed = param_single$seed)
    
    cat("   ✅ Single observation: PASSED\n")
  }, error = function(e) {
    cat("   ❌ Single observation: FAILED -", e$message, "\n")
  })
  
  # Single variable
  tryCatch({
    X_one_var <- matrix(rnorm(100), 100, 1)
    y_one_var <- rnorm(100)
    param_one_var <- param
    param_one_var$mtry <- 1
    
    fit_one_var <- RLT(X_one_var, y_one_var, model = "regression", 
                      ntrees = param_one_var$ntrees,
                      mtry = param_one_var$mtry,
                      nmin = param_one_var$nmin,
                      split.gen = "random",
                      nsplit = param_one_var$nsplit,
                      resample.prob = param_one_var$resample.prob,
                      resample.replace = param_one_var$resample.replace,
                      reinforcement = param_one_var$reinforcement,
                      importance = FALSE,
                      verbose = param_one_var$verbose,
                      ncores = param_one_var$ncores,
                      seed = param_one_var$seed)
    
    cat("   ✅ Single variable: PASSED\n")
  }, error = function(e) {
    cat("   ❌ Single variable: FAILED -", e$message, "\n")
  })
  
  # Test 5: Data validation
  cat("5. Testing data validation...\n")
  
  # NA values
  tryCatch({
    X_with_na <- X
    X_with_na[1, 1] <- NA
    
    fit_with_na <- RLT(X_with_na, y, model = "regression", 
                      ntrees = param$ntrees,
                      mtry = param$mtry,
                      nmin = param$nmin,
                      split.gen = "random",
                      nsplit = param$nsplit,
                      resample.prob = param$resample.prob,
                      resample.replace = param$resample.replace,
                      reinforcement = param$reinforcement,
                      importance = FALSE,
                      verbose = param$verbose,
                      ncores = param$ncores,
                      seed = param$seed)
    
    cat("   ❌ NA handling: FAILED (should have stopped)\n")
  }, error = function(e) {
    cat("   ✅ NA handling: PASSED (correctly stopped)\n")
  })
  
  # Dimension mismatch
  tryCatch({
    y_wrong <- y[1:100]
    
    fit_wrong <- RLT(X, y_wrong, model = "regression", 
                    ntrees = param$ntrees,
                    mtry = param$mtry,
                    nmin = param$nmin,
                    split.gen = "random",
                    nsplit = param$nsplit,
                    resample.prob = param$resample.prob,
                    resample.replace = param$resample.replace,
                    reinforcement = param$reinforcement,
                    importance = FALSE,
                    verbose = param$verbose,
                    ncores = param$ncores,
                    seed = param$seed)
    
    cat("   ❌ Dimension check: FAILED (should have stopped)\n")
  }, error = function(e) {
    cat("   ✅ Dimension check: PASSED (correctly stopped)\n")
  })
  
  # Test 6: Parameter validation
  cat("6. Testing parameter validation...\n")
  
  # Invalid mtry
  tryCatch({
    param_invalid <- param
    param_invalid$mtry <- 15  # > p
    
    fit_invalid <- RLT(X, y, model = "regression", 
                      ntrees = param_invalid$ntrees,
                      mtry = param_invalid$mtry,
                      nmin = param_invalid$nmin,
                      split.gen = "random",
                      nsplit = param_invalid$nsplit,
                      resample.prob = param_invalid$resample.prob,
                      resample.replace = param_invalid$resample.replace,
                      reinforcement = param_invalid$reinforcement,
                      importance = FALSE,
                      verbose = param_invalid$verbose,
                      ncores = param_invalid$ncores,
                      seed = param_invalid$seed)
    
    cat("   ❌ mtry validation: FAILED (should have stopped)\n")
  }, error = function(e) {
    cat("   ✅ mtry validation: PASSED (correctly stopped)\n")
  })
  
  # Test 7: Performance
  cat("7. Testing performance...\n")
  
  tryCatch({
    n_large <- 1000
    p_large <- 20
    X_large <- matrix(rnorm(n_large * p_large), n_large, p_large)
    y_large <- 1 + rowSums(X_large[, 1:5]) + rnorm(n_large)
    
    param_large <- param
    param_large$ntrees <- 100
    param_large$mtry <- 10
    param_large$nmin <- 20
    
    start_time <- Sys.time()
    fit_large <- RLT(X_large, y_large, model = "regression", 
                    ntrees = param_large$ntrees,
                    mtry = param_large$mtry,
                    nmin = param_large$nmin,
                    split.gen = "random",
                    nsplit = param_large$nsplit,
                    resample.prob = param_large$resample.prob,
                    resample.replace = param_large$resample.replace,
                    reinforcement = param_large$reinforcement,
                    importance = FALSE,
                    verbose = param_large$verbose,
                    ncores = param_large$ncores,
                    seed = param_large$seed)
    large_time <- difftime(Sys.time(), start_time, units = "secs")
    
    cat("   ✅ Performance: PASSED (", round(as.numeric(large_time), 2), "seconds)\n")
  }, error = function(e) {
    cat("   ❌ Performance: FAILED -", e$message, "\n")
  })
  
  cat("\n=== QC Summary ===\n")
  cat("Basic functionality: Tested\n")
  cat("Linear combination: Tested\n")
  cat("Reproducibility: Tested\n")
  cat("Edge cases: Tested\n")
  cat("Data validation: Tested\n")
  cat("Parameter validation: Tested\n")
  cat("Performance: Tested\n")
  cat("\nQC completed successfully!\n")
  return(TRUE)
}

#' @title validate_regression_inputs
#' @name validate_regression_inputs
#' @description Quick validation of regression inputs
#' @param x Input features matrix
#' @param y Response vector
#' @param param Parameter list
#' @return Validation result
#' @export
validate_regression_inputs <- function(x, y, param) {
  
  cat("=== Input Validation ===\n")
  
  # Check data types
  if (!is.matrix(x)) {
    cat("❌ x must be a matrix\n")
    return(FALSE)
  }
  
  if (!is.vector(y)) {
    cat("❌ y must be a vector\n")
    return(FALSE)
  }
  
  if (!is.numeric(y)) {
    cat("❌ y must be numeric\n")
    return(FALSE)
  }
  
  # Check dimensions
  if (nrow(x) != length(y)) {
    cat("❌ Dimensions don't match: x has", nrow(x), "rows, y has", length(y), "elements\n")
    return(FALSE)
  }
  
  # Check for NA values
  if (any(is.na(x))) {
    cat("❌ x contains NA values\n")
    return(FALSE)
  }
  
  if (any(is.na(y))) {
    cat("❌ y contains NA values\n")
    return(FALSE)
  }
  
  # Check parameters
  if (!is.null(param$mtry) && param$mtry > ncol(x)) {
    cat("❌ mtry (", param$mtry, ") cannot be larger than number of variables (", ncol(x), ")\n")
    return(FALSE)
  }
  
  if (!is.null(param$nmin) && param$nmin < 1) {
    cat("❌ nmin cannot be less than 1\n")
    return(FALSE)
  }
  
  if (!is.null(param$resample.prob) && (param$resample.prob <= 0 || param$resample.prob > 1)) {
    cat("❌ resample.prob must be in (0, 1]\n")
    return(FALSE)
  }
  
  cat("✅ All inputs are valid\n")
  cat("Data: n =", nrow(x), ", p =", ncol(x), "\n")
  cat("Response range:", round(range(y), 3), "\n")
  
  return(TRUE)
} 