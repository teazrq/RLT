#' @title RegForest
#' @name RegForest
#' @description Internal function for fitting regression forest
#' @keywords internal

RegForest <- function(x, y, ncat,
                      obs.w, var.w,
                      resample.preset,
                      param,
                      ...)
{
  # prepare y
  storage.mode(y) <- "double"
  

  
  if (param$linear.comb == 1)
  {
    if (param$verbose > 0)
      cat("Fitting Regression Forest ... \n")    
      
    # check splitting rules
    all.split.rule = c("default")
    param$"split.rule" <- match.arg(param$"split.rule", all.split.rule)
    param$"split.rule" <- as.integer(match(param$"split.rule", all.split.rule))

    # fit single variable split model
    fit = RegUniForestFit(x, y, ncat,
                          obs.w, var.w,
                          resample.preset,
                          param)
  
    fit[["parameters"]] = param
    fit[["ncat"]] = ncat
    fit[["obs.w"]] = obs.w
    fit[["var.w"]] = var.w
    fit[["y"]] = y
    
    class(fit) <- c("RLT", "fit", "reg", "uni", "single")
  }else{
    
    if (param$verbose > 0)
      cat("Fitting Regression Forest with Linear Combination Splitting... \n") 
    
    # check splitting rules
    # default is sir
    all.split.rule = c("default", "save", "pca")
    param$"split.rule" <- match.arg(param$"split.rule", all.split.rule)
    param$"split.rule" <- as.integer(match(param$"split.rule", all.split.rule))
    
    # fit linear combination split model
    fit = RegUniCombForestFit(x, y, ncat,
							  obs.w, var.w,
							  resample.preset,
							  param)
    
    fit[["parameters"]] = param
    fit[["ncat"]] = ncat
    fit[["obs.w"]] = obs.w
    fit[["var.w"]] = var.w
    fit[["y"]] = y
    
    class(fit) <- c("RLT", "fit", "reg", "uni", "comb")
  }

  return(fit)
}
