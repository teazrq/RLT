#' @title C-index
#' @name cindex
#' @description Calculate concordance between survival outcomes and a risk score.
#' Larger risk scores indicate earlier failure.
#' @param y Observed survival times.
#' @param censor Event indicator: 1 for an observed event, 0 for right censoring.
#' @param pred One risk score per observation.
#' @details The example uses final OOB cumulative hazard as its risk score.
#' The survival forest's stored \code{Error} uses a different score: the sum
#' of OOB cumulative hazards over all failure-time grid points. The example
#' therefore need not equal \code{1 - fit$Error}.
#' @export
#' @return The concordance index, or \code{NaN} when there are no comparable pairs.
#' @examples
#' \donttest{
#'   set.seed(42)
#'   n <- 100
#'   x <- matrix(rnorm(n * 5), ncol = 5)
#'   y <- rexp(n, rate = exp(rowSums(x[, 1:2])))
#'   censor <- rbinom(n, 1, 0.7)
#'   fit <- RLT(x, y, censor = censor, model = "survival", ntrees = 100)
#'   # Sum OOB hazard increments to obtain final cumulative hazard.
#'   risk <- rowSums(fit$Prediction)
#'   cindex(y, censor, risk)
#' }

cindex <- function(y, censor, pred)
{
    if (length(y) != length(censor) | length(y) != length(pred))
        stop("arguments length differ")
    
    if ( any( ! (censor %in% c(0, 1)) ) )
        stop("censoring indicator must be 0 or 1")
        
	return( cindex_d(y, censor, pred) )
}
