#' @title ClaForest
#' @name RegForest
#' @description Internal function for fitting classification forest
#' @keywords internal

ClaForest <- function(x, y, 
                      ncat,
                      obs.w, var.w,
                      resample.preset,
                      param,
                      ...)
{
  # prepare data
  # if (!is.vector(y)) stop("y must be a vector")
  if (any(is.na(x))) stop("NA not permitted in x")
  if (any(is.na(y))) stop("NA not permitted in y")
  if (nrow(x) != length(y)) stop("number of observations does not match: x & y")
  
  if (!is.factor(y)) stop("y must be a factor")
  if (length(unique(y)) != nlevels(y)) warning("y contain empty classes")
  
  nclass = nlevels(y)
  if (nclass < 2) step("y's are identical")
  
  ylabels = levels(y)
  y.integer = as.numeric(y) - 1
  
  storage.mode(y.integer) <- "integer"
  
  # fit model
  
  if (param$linear.comb == 1)
  {
    if (param$verbose > 0)
      cat("in ClaForest ... \n")
    
    # check splitting rules
    if (is.null(param$"split.rule"))
      param$"split.rule" <- "gini"
    
    # existing splitting rule for regular regression
    all.split.rule = c("gini")
    
    param$"split.rule" <- match(param$"split.rule", all.split.rule)
    
    if (param$"split.rule" == 0)
      warning("split.rule is not compatiable with classification, switching to default")
    
    param$"split.rule" = 1

    # fit single variable split model
    fit = ClaUniForestFit(x, y.integer, ncat, nclass,
                          obs.w, var.w,
                          resample.preset,
                          param)

    fit[["parameters"]] = param
    fit[["ncat"]] = ncat
    fit[["obs.w"]] = obs.w
    fit[["var.w"]] = var.w
    fit[["y"]] = y
    fit[["ylabels"]] = ylabels
    fit[["nclass"]] = nclass

    fit$Prediction = as.factor( c(1:nclass, fit$Prediction+1) )[-(1:fit$parameters$n)]
    levels(fit$Prediction) = ylabels
    
    fit$OOBPrediction = as.factor( c(1:nclass, fit$OOBPrediction+1) )[-(1:fit$parameters$n)]
    levels(fit$OOBPrediction) = ylabels

    class(fit) <- c("RLT", "fit", "cla", "uni")
  }else{
    
    if (param$verbose > 0)
      cat("Classification Forest with Linear Combination Splits ... \n")
    
    stop("Not avaliable yet")
    
  }
  
  return(fit)
}
