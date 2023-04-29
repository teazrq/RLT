#' @title SurvForest
#' @name SurvForest
#' @description Internal function for fitting survival forest
#' @keywords internal

SurvForest <- function(x, y, censor, 
                       ncat, failcount,
                       obs.w, var.w,
                       resample.preset,
                       param,
                       ...)
{
  # prepare y
  timepoints = sort(unique(y[censor == 1]))

  # a smaller failcount
  if (failcount != 0 & failcount < length(timepoints))
  {
    timeloc = floor(stats::quantile(1:length(timepoints), 
                             probs = seq(0, 1, length.out = failcount)))
    
    # reduced set of timepoints
    timepoints = timepoints[timeloc]
  }

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
  
  if (param$linear.comb == 1)
  {
    if (param$verbose > 0)
      cat("Fitting Survival Forest... \n")    
      
    # check splitting rules
    if (is.null(param$"split.rule"))
      param$"split.rule" = "logrank"
    
    if (param$"split.rule" == "default")
      param$"split.rule" = "logrank"
    
    all.split.rule = c("logrank", "suplogrank", "coxgrad")
    
    #param$"split.rule" <- match.arg(param$"split.rule", all.split.rule)
    param$"split.rule" <- match(param$"split.rule", all.split.rule)
    if(is.na(param$"split.rule")){
      print("split.rule chosen not currently implemented: switching to logrank.")
      print(paste0("Implemented split rules: ", paste0(all.split.rule, collapse = ", ")))
      param$"split.rule" = 1
    }
    
    # check currently not implemented
    if (param$"split.rule" <= 2)
    {
      if (param$'use.obs.w' != 0)
        print("Observation weight is only implemented with coxgrad splitting rule")
          
      if (param$'use.var.w' != 0)
        print("Variable weight is only implemented with coxgrad splitting rule")
    }
    
    # fit single variable split model
    fit = SurvUniForestFit(x, y.point, censor, ncat,
                          obs.w, var.w,
                          resample.preset,
                          param)
  
    fit[["parameters"]] = param
    fit[["ncat"]] = ncat
    fit[["obs.w"]] = obs.w
    fit[["var.w"]] = var.w
    fit[["y"]] = y
    fit[["censor"]] = censor
    fit[["failcount"]] = failcount
    fit[["timepoints"]] = timepoints
    
    class(fit) <- c("RLT", "fit", "surv", "uni", "single")
  }else{
    cat("Linear combination fitting not implemented for survival random forests.")
  }

  return(fit)
}
