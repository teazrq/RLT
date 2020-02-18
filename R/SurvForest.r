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
                       ObsTrack,
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
  if (is.null(param$"split.rule"))
    param$"split.rule" = "logrank"
  
  if (param$"split.rule" == "penll" & param$use.var.w == 0)
    stop("must specify variable weights if penalized splitting rule is used.")
  
  all.split.rule = c("logrank", "suplogrank", "ll", "penll")

  param$"split.rule" <- match.arg(param$"split.rule", all.split.rule)
  param$"split.rule" <- match(param$"split.rule", all.split.rule)
  
  # fit model
  fit = SurvForestUniFit(x, y.point, censor, ncat,
                         param, RLT.control,
                         obs.w, var.w,
                         ncores, verbose,
                         ObsTrack)

  fit[["parameters"]] = param
  fit[["RLT.control"]] = RLT.control  
  
  fit[["timepoints"]] = timepoints
  fit[["ncat"]] = ncat
  fit[["obs.w"]] = obs.w
  fit[["var.w"]] = var.w
  fit[["y"]] = y
  fit[["y.point"]] = y.point
  fit[["censor"]] = censor
  
  class(fit) <- c("RLT", "fit", "surv")
  return(fit)
}
