#' @title Print a RLT object
#' @description Print a RLT object
#' @param x A fitted RLT object
#' @param ... ...
#' 
#' @importFrom stats var
#' @export
#' @examples
#' \donttest{
#' set.seed(42)
#' x <- matrix(rnorm(100 * 5), ncol = 5)
#' y <- rowSums(x[, 1:2]) + rnorm(100)
#' fit <- RLT(x, y, ntrees = 50)
#' print(fit)
#' }
print.RLT <- function(x, ...)
{
  # detect object type
  obj_class <- class(x)
  
  if (obj_class[2] == "pred") {
    model_name <- switch(obj_class[3],
                         "reg" = "Regression",
                         "cla" = "Classification",
                         "surv" = "Survival",
                         "Quantile")
    cat(paste0("An RLT ", model_name, " prediction object \n"))
    return(invisible(x))
  }
  
  if (obj_class[2] == "band") {
    cat("An RLT survival confidence band object \n")
    return(invisible(x))
  }
  
  if (obj_class[2] == "kernel") {
    cat(paste0("An RLT ", obj_class[3], " kernel object.\n"))
    return(invisible(x))
  }
  
  # --- fitted object ---
  
  param <- x$parameters
  
  # model type
  model_name <- switch(obj_class[3],
                       "reg" = "Regression",
                       "cla" = "Classification",
                       "surv" = "Survival",
                       "Quantile")
  
  # header
  header <- paste0("RLT ", model_name, " Forest")
  
  # check linear combination
  ncomb <- param$linear.comb
  if (!is.null(ncomb) && ncomb > 1) {
    header <- paste0(header, " (Linear Combination)")
  }
  
  # collect lines (label, value)
  lines_fmt <- character(0)
  
  # (N, P) with continuous/categorical breakdown
  n_cont <- sum(x$ncat == 1)
  n_cat  <- sum(x$ncat > 1)
  np_str <- paste0("(", param$n, ", ", param$p, ")")
  if (n_cat > 0) {
    np_str <- paste0(np_str, " [", n_cont, " continuous, ", n_cat, " categorical]")
  }
  lines_fmt <- c(lines_fmt, paste0("              (N, P) = ", np_str))
  
  # ntrees
  lines_fmt <- c(lines_fmt, paste0("          # of trees = ", param$ntrees))
  
  # (mtry, nmin)
  lines_fmt <- c(lines_fmt, paste0("        (mtry, nmin) = (", param$mtry, ", ", param$nmin, ")"))
  
  # split generate
  split_str <- if (param$nsplit == 0) "Best" else paste0("Random, ", param$nsplit)
  lines_fmt <- c(lines_fmt, paste0("      split generate = ", split_str))
  
  # linear combination split (only if ncomb > 1)
  if (!is.null(ncomb) && ncomb > 1) {
    lines_fmt <- c(lines_fmt, paste0("linear combination split = ", ncomb))
  }
  
  # sampling: percentage format
  prob_pct <- round(param$resample.prob * 100, 1)
  # remove trailing .0 for integers
  if (prob_pct == floor(prob_pct)) prob_pct <- as.integer(prob_pct)
  replace_str <- if (param$resample.replace) "w/ replace" else "w/o replace"
  lines_fmt <- c(lines_fmt, paste0("            sampling = ", prob_pct, "% ", replace_str))
  
  # obs weights (only if used)
  if (!is.null(x$obs.w) && length(x$obs.w) > 0 && param$use.obs.w == 1) {
    lines_fmt <- c(lines_fmt, paste0("         obs weights = Yes"))
  }
  
  # var weights (only if used)
  if (!is.null(param$use.var.prob) && param$use.var.prob == 1) {
    lines_fmt <- c(lines_fmt, paste0("         var weights = Yes"))
  }
  
  # reinforcement
  if (!is.null(param$reinforcement) && param$reinforcement == 1) {
    lines_fmt <- c(lines_fmt, paste0("       reinforcement = Yes"))
  }
  
  # variance estimation
  var_mode <- param$var.mode
  if (!is.null(var_mode)) {
    is_active <- (is.numeric(var_mode) && var_mode != 0) ||
                 (is.character(var_mode) && var_mode != "none")
    if (is_active) {
      mode_str <- switch(as.character(var_mode),
                         "1" = "matched",
                         "2" = "IJ",
                         "3" = "jack",
                         var_mode)
      lines_fmt <- c(lines_fmt, paste0(" variance estimation = ", mode_str))
    }
  }
  
  # importance
  imp_str <- switch(as.character(param$importance),
                    "0" = "none",
                    "1" = "permutation",
                    "2" = "distribute",
                    paste0("mode ", param$importance))
  lines_fmt <- c(lines_fmt, paste0("          importance = ", imp_str))
  
  # OOB error (only if available)
  if (!is.null(x$Error) && length(x$Error) > 0) {
    if (obj_class[3] == "reg") {
      r2 <- 1 - x$Error / var(x$y)
      lines_fmt <- c(lines_fmt, paste0("            OOB MSE = ", round(x$Error, 4), " (R2 = ", round(r2, 4), ")"))
    } else if (obj_class[3] == "cla") {
      lines_fmt <- c(lines_fmt, paste0("      OOB misclass = ", round(x$Error * 100, 2), "%"))
    } else {
      lines_fmt <- c(lines_fmt, paste0("          OOB error = ", round(x$Error, 4)))
    }
  }
  
  # build output
  max_width <- max(nchar(header), max(nchar(lines_fmt)))
  hline <- paste(rep("-", max_width), collapse = "")
  
  cat(hline, "\n", sep = "")
  cat(header, "\n", sep = "")
  cat(hline, "\n", sep = "")
  cat(paste0(lines_fmt, collapse = "\n"))
  cat("\n", hline, "\n", sep = "")
  
  invisible(x)
}
