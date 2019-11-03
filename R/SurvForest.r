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
  if ( any( ! (censor %in% c(0, 1)) ) )
      stop("censoring indicator must be 0 or 1")    
    
  # prepare y
  
  timepoints = sort(unique(y[censor == 1]))
  
  y.point = rep(NA, length(y))
  
  for (i in 1:length(y))
  {
    if (censor[i] == 1)
      y.point[i] = match(y[i], timepoints)
    else
      y.point[i] = sum(y[i] >= timepoints)
  }
  
  storage.mode(y.point) <- "integer"
  storage.mode(censor) <- "integer"
  
  param$'nfail' = length(timepoints)
  
  # check splitting rule 
  all.split.rule = c("var")
    
  param$"split.rule" <- match.arg(param$"split.rule", all.split.rule)
  param$"split.rule" <- match(param$"split.rule", all.split.rule)
  
  # fit model
  fit = SurvForestUniFit(x, y.point, censor, ncat,
                         param, RLT.control,
                         obs.w,
                         var.w,
                         ncores,
                         verbose)

  fit[["timepoints"]] = timepoints
  fit[["ncat"]] = ncat
  fit[["parameters"]] = param
  fit[["RLT.control"]] = RLT.control
  fit[["obs.w"]] = obs.w
  fit[["var.w"]] = var.w
  fit[["y"]] = y
  fit[["y.point"]] = y.point
  fit[["censor"]] = censor
  
  class(fit) <- c("RLT", "fit", "surv")
  return(fit)
}
