#' @title SurvForest
#' @name SurvForest
#' @description Internal function for fitting survival forest
#' @keywords internal

SurvForest <- function(x, y, censor,
                       ncat,
                       param,
                       RLT.control,
                       obs.w,
                       var.w,
                       ncores,
                       verbose,
                       ...)
{
  # prepare y
  storage.mode(y) <- "integer"

  # check splitting rule 
  all.split.rule = c("var")
    
  param$"split.rule" <- match.arg(param$"split.rule", all.split.rule)
  param$"split.rule" <- match(param$"split.rule", all.split.rule)
  
  # fit model
  fit = SurvForestUniFit(x, y, censor, ncat,
                         param, RLT.control,
                         obs.w,
                         var.w,
                         ncores,
                         verbose)

  fit[["ncat"]] = ncat
  fit[["parameters"]] = param
  fit[["RLT.control"]] = RLT.control
  fit[["obs.w"]] = obs.w
  fit[["var.w"]] = var.w
  fit[["y"]] = y
  fit[["censor"]] = censor
  
  class(fit) <- c("RLT", "fit", "surv")
  return(fit)
}
