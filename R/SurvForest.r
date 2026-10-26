#' @title SurvForest
#' @name SurvForest
#' @description Internal function for fitting survival forest
#' @keywords internal

SurvForest <- function(x, y, censor, 
                       ncat, time.grid.size,
                       obs.w, var.prob,
                       resample.preset,
                       param,
                       ...)
{
  # prepare data
  if (!is.vector(y)) stop("y must be a vector")
  if (any(is.na(x))) stop("NA not permitted in x")
  if (any(is.na(y))) stop("NA not permitted in y")
  if (any(is.na(censor))) stop("NA not permitted in censor")
  if (nrow(x) != length(y)) stop("number of observations does not match: x & y")
  if (nrow(x) != length(censor)) stop("number of observations does not match: x & censor")
  if (!is.numeric(y)) stop("y must be numerical for survival")
  if (any(!is.finite(y))) stop("y must contain finite survival times")
  if (!is.numeric(censor)) stop("censor must be numeric")
  if (!all(censor %in% c(0, 1))) stop("censor must be 0 or 1")

  if (!is.numeric(time.grid.size) || is.complex(time.grid.size) ||
      length(time.grid.size) != 1L ||
      !is.finite(time.grid.size) || time.grid.size < 0 ||
      time.grid.size != floor(time.grid.size))
    stop("time.grid.size must be a single non-negative integer")

  # Get all unique failure time points
  timepoints = sort(unique(y[censor == 1]))
  if (length(timepoints) == 0L)
    stop("survival fitting requires at least one observed event")

  # Dropping failure times would leave observed events without valid indices.
  if (time.grid.size != 0 && time.grid.size < length(timepoints))
    stop("time.grid.size cannot reduce the training failure-time grid; use band.grid.size in predict() for reduced evaluation")

  # Map observed times to integer positions in the time grid
  y.point = rep(NA, length(y))
  
  for (i in seq_len(length(y)))
  {
    if (censor[i] == 1)
      y.point[i] = match(y[i], timepoints)
    else
      y.point[i] = sum(y[i] >= timepoints)
  }

  if (anyNA(y.point) || any(y.point < 0) ||
      any(y.point > length(timepoints)) || any(y.point[censor == 1] < 1))
    stop("invalid survival time indices before native fitting")
  
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
    
    # Handle both string and integer inputs
    if (is.character(param$"split.rule"))
    {
      all.split.rule = c("logrank", "suplogrank", "coxgrad")
      param$"split.rule" <- match(param$"split.rule", all.split.rule)
      
      if(is.na(param$"split.rule")){
        warning("split.rule not recognized. Use 'logrank', 'suplogrank', or 'coxgrad'. Switching to logrank.")
        param$"split.rule" = 1L
      }
    } else if (is.numeric(param$"split.rule"))
    {
      # Already an integer, just validate it's in range 1-3
      if (param$"split.rule" < 1 || param$"split.rule" > 3)
      {
        warning("split.rule integer must be 1-3. Resetting to 1 (logrank)")
        param$"split.rule" = 1L
      }
      param$"split.rule" = as.integer(param$"split.rule")
    } else
    {
      warning("split.rule must be character or integer. Resetting to logrank")
      param$"split.rule" = 1L
    }
    
    # fit single variable split model
    fit = SurvUniForestFit(x, y.point, censor, ncat,
                          obs.w, var.prob,
                          resample.preset,
                          param)
  
    fit[["parameters"]] = param
    fit[["ncat"]] = ncat
    fit[["obs.w"]] = obs.w
    fit[["var.prob"]] = var.prob
    fit[["y"]] = y
    fit[["censor"]] = censor
    fit[["time.grid.size"]] = time.grid.size
    fit[["timepoints"]] = timepoints
    
    class(fit) <- c("RLT", "fit", "surv", "uni")
  }else{
    
    if (param$verbose > 0)
      cat("Fitting Survival Forest with Linear Combination Splits ... \n")
    
    # check linear combination method
    if (is.null(param$"linear.comb.method"))
      param$"linear.comb.method" <- "coxph"
    
    # Handle both string and integer inputs
    if (is.character(param$"linear.comb.method"))
    {
      all.linear.comb.method = c("coxph", "naive")
      param$"linear.comb.method" <- match(param$"linear.comb.method", all.linear.comb.method, nomatch = 0)
      
      if (param$"linear.comb.method" == 0)
      {
        warning("linear.comb.method not recognized. Use 'coxph' or 'naive'. Resetting to coxph")
        param$"linear.comb.method" = 1L
      }
    } else if (is.numeric(param$"linear.comb.method"))
    {
      if (param$"linear.comb.method" < 1 || param$"linear.comb.method" > 2)
      {
        warning("linear.comb.method integer must be 1-2. Resetting to 1 (coxph)")
        param$"linear.comb.method" = 1L
      }
      param$"linear.comb.method" = as.integer(param$"linear.comb.method")
    } else
    {
      warning("linear.comb.method must be character or integer. Resetting to coxph")
      param$"linear.comb.method" = 1L
    }
    
    # check splitting rules (logrank for LC scoring)
    if (is.null(param$"split.rule"))
      param$"split.rule" = "logrank"
    
    if (param$"split.rule" == "default")
      param$"split.rule" = "logrank"
    
    if (is.character(param$"split.rule"))
    {
      all.split.rule = c("logrank", "suplogrank", "coxgrad")
      param$"split.rule" <- match(param$"split.rule", all.split.rule)
      
      if(is.na(param$"split.rule")){
        warning("split.rule not recognized. Use 'logrank', 'suplogrank', or 'coxgrad'. Switching to logrank.")
        param$"split.rule" = 1L
      }
    } else if (is.numeric(param$"split.rule"))
    {
      if (param$"split.rule" < 1 || param$"split.rule" > 3)
      {
        warning("split.rule integer must be 1-3. Resetting to 1 (logrank)")
        param$"split.rule" = 1L
      }
      param$"split.rule" = as.integer(param$"split.rule")
    } else
    {
      warning("split.rule must be character or integer. Resetting to logrank")
      param$"split.rule" = 1L
    }
    
    # fit linear combination split model
    fit = SurvUniCombForestFit(x, y.point, censor, ncat,
                                obs.w, var.prob,
                                resample.preset,
                                param)
    
    fit[["parameters"]] = param
    fit[["ncat"]] = ncat
    fit[["obs.w"]] = obs.w
    fit[["var.prob"]] = var.prob
    fit[["y"]] = y
    fit[["censor"]] = censor
    fit[["time.grid.size"]] = time.grid.size
    fit[["timepoints"]] = timepoints
    
    class(fit) <- c("RLT", "fit", "surv", "uni", "comb")
  }

  return(fit)
}
