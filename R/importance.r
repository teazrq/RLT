#' @title Variable Importance Summary
#' @description   Extract variable importance from a fitted RLT model.
#'                When variance estimation was enabled via \code{var.mode},
#'                standard deviations, Z-scores, and significance codes are
#'                also reported. Negative variance estimates yield \code{NA}
#'                for SD, Z, and significance.
#'
#' @param object  A fitted \code{RLT} object from \code{\link{RLT}}.
#' @param ...     Additional arguments (unused).
#'
#' @return A \code{data.frame} with columns:
#' \itemize{
#'   \item \code{Variable}: variable name
#'   \item \code{VI}: variable importance
#'   \item \code{SD}: standard deviation of VI (\code{NA} if not estimated or negative variance)
#'   \item \code{Z}: Z-score (\code{VI / SD}, \code{NA} if SD is \code{NA})
#'   \item \code{Sig}: significance code (\code{""} if not estimated or negative variance)
#' }
#'
#' Significance codes: \code{***} |Z| >= 2.58, \code{**} |Z| >= 1.96, \code{*} |Z| >= 1.64.
#'
#' @examples
#' \dontrun{
#' fit <- RLT(x, y, model = "classification", importance = TRUE, var.mode = TRUE)
#' importance(fit)
#' }
#'
#' @export
importance <- function(object, ...)
{
  UseMethod("importance")
}

#' @export
importance.RLT <- function(object, ...)
{
  if (class(object)[[2]] != "fit")
    stop("importance() is only available for fitted RLT models.")

  if (is.null(object$VarImp))
    stop("No variable importance in this model. Fit with importance = TRUE.")

  vi <- as.numeric(object$VarImp)
  p <- length(vi)

  varnames <- if (!is.null(object$xnames)) object$xnames else paste0("V", 1:p)

  # Check if variance estimation was done
  has_varvi <- !is.null(object$VarVI) && length(object$VarVI) == p

  if (has_varvi)
  {
    vvi <- as.numeric(object$VarVI)
    sd_val <- sqrt(pmax(vvi, 0))
    sd_val[vvi < 0] <- NA

    z_val <- vi / sd_val

    sig_val <- character(p)
    sig_val[is.na(z_val)] <- ""
    sig_val[!is.na(z_val)] <- ifelse(abs(z_val[!is.na(z_val)]) >= 2.58, "***",
                              ifelse(abs(z_val[!is.na(z_val)]) >= 1.96, "**",
                              ifelse(abs(z_val[!is.na(z_val)]) >= 1.64, "*",  "")))

    result <- data.frame(Variable = varnames,
                         VI = vi,
                         SD = sd_val,
                         Z = z_val,
                         Sig = sig_val,
                         stringsAsFactors = FALSE,
                         check.names = FALSE)
  } else {
    result <- data.frame(Variable = varnames,
                         VI = vi,
                         stringsAsFactors = FALSE,
                         check.names = FALSE)
  }

  class(result) <- c("importance.RLT", "data.frame")
  attr(result, "has_variance") <- has_varvi
  return(result)
}

#' @title Print Importance Summary
#' @description Print method for \code{importance.RLT} objects.
#' @param x An \code{importance.RLT} object.
#' @param digits Number of digits for formatting. Default: 4.
#' @param ... Additional arguments (unused).
#'
#' @export
#' @examples
#' \donttest{
#' set.seed(42)
#' x <- matrix(rnorm(100 * 5), ncol = 5)
#' y <- rowSums(x[, 1:2]) + rnorm(100)
#' fit <- RLT(x, y, ntrees = 50, importance = TRUE)
#' print(importance(fit))
#' }
print.importance.RLT <- function(x, digits = 4, ...)
{
  has_variance <- attr(x, "has_variance")

  if (has_variance)
  {
    cat(sprintf("%-12s %10s %12s %10s  %s\n",
                "Variable", "VI", "SD", "Z", "Sig"))
    cat(strrep("-", 58), "\n")

    for (i in 1:nrow(x))
    {
      vi_str <- sprintf("%10.4f", x$VI[i])

      if (is.na(x$SD[i]))
      {
        sd_str <- sprintf("%12s", "NA")
        z_str  <- sprintf("%10s", "NA")
        sig_str <- ""
      } else {
        sd_str <- sprintf("%12.6f", x$SD[i])
        z_str  <- sprintf("%10.2f", x$Z[i])
        sig_str <- x$Sig[i]
      }

      cat(sprintf("%-12s %s %s %s  %s\n",
                  x$Variable[i], vi_str, sd_str, z_str, sig_str))
    }

    n_neg <- sum(is.na(x$SD))
    if (n_neg > 0)
      cat(sprintf("\nNote: %d variable(s) with negative variance estimate (SD, Z shown as NA)\n", n_neg))
  } else {
    cat(sprintf("%-12s %10s\n",
                "Variable", "VI"))
    cat(strrep("-", 26), "\n")

    for (i in 1:nrow(x))
    {
      vi_str <- sprintf("%10.4f", x$VI[i])
      cat(sprintf("%-12s %s\n", x$Variable[i], vi_str))
    }
  }

  invisible(x)
}
