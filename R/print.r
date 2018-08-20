#' @title Print a RLT object
#' @description Print a RLT object
#' @param x A fitted RLT object
#' @param ... ...
#' @examples
#' x = matrix(rnorm(100), ncol = 10)
#' y = rowMeans(x)
#' fit = RLT(x, y, ntrees = 5)
#' fit

print.RLT<- function(x, ...)
{
  cat("Reinforcement Learning Trees for", x$model, "model:\n")
  cat("         number of trees:", x$ntrees, "\n")
  cat("             sample size:", x$n, "\n")
  cat("     number of variables:", x$p, "\n")
  cat("                    nmin:", x$nmin, "\n")

  if (x$reinforcement == 0)
    cat("                    mtry:", x$mtry, "\n")

  cat("              resampling:", paste(round(x$resample.prob*100,2), "%", sep=""), c("without", "with")[x$replacement+1], "replacement \n")
  cat(" split generating method:", x$split.gen, "\n")
  cat("     use subject weights:", c("No", "Yes")[x$use.sub.weight+1], "\n")

  # if (x$oobMSE)
  # cat("      variance explained:", paste(round(x$PMSE*100,2), "%", sep=""), "(based on OOB cross-validation) \n")

  cat("  reinforcement learning:", c("No", "Yes")[x$reinforcement+1], "\n")

  if (x$reinforcement)
  {
    if (x$muting > 1)
      cat("        muting by count:", x$muting, "at each split \n")

    if (x$muting == -1)
      cat("                  muting:", paste(round(x$muting.percent*100,2), "%", sep=""), "at each split \n")
      cat("     protected variables:", x$protect, "\n")

    if (x$combsplit > 1)
      cat("     Linear combination:", x$combsplit, "with VI threshold", x$combsplit.th, "\n")
  }
}
