#' @title RegForest
#' @name RegForest
#' @description Internal function for fitting regression forest
#' @keywords internal

RegForest <- function(x, y, 
                      ncat,
                      obs.w, var.w,
                      resample.preset,
                      param,
                      ...)
{
  # prepare data
  if (!is.vector(y)) stop("y must be a vector")
  if (any(is.na(x))) stop("NA not permitted in x")
  if (any(is.na(y))) stop("NA not permitted in y")
  if (nrow(x) != length(y)) stop("number of observations does not match: x & y")
  if (!is.numeric(y)) stop("y must be numerical for regression")
  
  storage.mode(y) <- "double"
  
  # fit model
  
  if (param$linear.comb == 1)
  {
    if (param$verbose > 0)
      cat("Regression Random Forest ... \n")
      
    # check splitting rules
    if (is.null(param$"split.rule"))
      param$"split.rule" <- "var"

    # existing splitting rule for regular regression
    all.split.rule = c("var")

    param$"split.rule" <- match(param$"split.rule", all.split.rule)
    
    if (param$"split.rule" == 0)
      warning("split.rule is not compatiable with regression, switching to default")
    
    param$"split.rule" = 1
      
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
      cat("Regression Forest with Linear Combination Splits ... \n") 
    
    # check splitting rules
    if (is.null(param$"split.rule"))
      param$"split.rule" <- "sir"
    
    all.split.rule = c("sir", "save", "pca", "lm")
    param$"split.rule" <- match(param$"split.rule", all.split.rule)

    if (param$"split.rule" == 0)
      warning("split.rule is not compatiable with linear combination regression; reset")    

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
