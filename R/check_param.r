#' @title check_param
#' @name check_param
#' @description Check parameters
#' @keywords internal

check_param <- function(n, p,
                        ntrees, mtry, nmin,
                        alpha, split.gen, split.rule, nsplit,
                        replacement, resample.prob,
                        importance, reinforcement, kernel.ready)
{
  ntrees = max(ntrees, 1)
  storage.mode(ntrees) <- "integer"

  mtry = max(min(mtry, p), 1)
  storage.mode(mtry) <- "integer"  
  
  nmin = max(1, floor(nmin))
  storage.mode(nmin) <- "integer"
  
  alpha = max(0, min(alpha, 0.5))
  storage.mode(alpha) <- "double"
  
  split.gen = match(split.gen, c("random", "rank", "best"))
  storage.mode(split.gen) <- "integer"
  
  nsplit = max(1, nsplit)
  storage.mode(nsplit) <- "integer"
  
  replacement = (replacement != 0)
  storage.mode(replacement) <- "integer"
  
  resample.prob = max(0, min(resample.prob, 1))
  storage.mode(resample.prob) <- "double"
  
  importance = (importance != 0)
  storage.mode(importance) <- "integer"
  
  reinforcement = (reinforcement != 0)
  storage.mode(reinforcement) <- "integer"
  
  kernel.ready = (kernel.ready != 0)
  storage.mode(kernel.ready) <- "integer"
  
  return(param <- list("n" = n,
                       "p" = p,
                       "ntrees" = ntrees,
                       "mtry" = mtry,
                       "nmin" = nmin,
                       "alpha" = alpha,
                       "split.gen" = split.gen,
                       "split.rule" = split.rule, # this need to be checked within the R fitting function
                       "nsplit" = nsplit,
                       "replacement" = replacement,
                       "resample.prob" = resample.prob,
                       "importance" = importance,
                       "reinforcement" = reinforcement,
                       "kernel.ready" = kernel.ready,
                       "use.obs.w" = 0L,
                       "use.var.w" = 0L))
}



#' @title check_param
#' @name check_param
#' @description Check parameters for RLT method
#' @keywords internal

check_RLT_param <- function(RLT.control)
{
  
  return(RLT.control)

}
  

#' @title check_input
#' @name check_input
#' @description Check input arguments
#' @param x x
#' @param y x
#' @param censor censor
#' @param model model 
#' @keywords internal


check_input <- function(x, y, censor, model)
{
  if (!is.matrix(x) & !is.data.frame(x)) stop("x must be a matrix or a data.frame")
  if (!is.vector(y)) stop("y must be a vector")
  
  if (any(is.na(x))) stop("NA not permitted in x")
  if (any(is.na(y))) stop("NA not permitted in y")
  
  if (!is.null(y))
    if (nrow(x) != length(y)) stop("number of observations does not match: x & y")
  
  if (!is.numeric(y) & !is.factor(y))
    stop("y must be numeric or factor")
  
  if (!is.null(censor))
  {
    if (!is.vector(censor) | !is.numeric(censor)) stop("censor must be a numerical vector")
    if (length(y) != length(censor)) stop("number of observations does not match: y & censor")
    
    if ( any(sort(unique(censor)) != c(0, 1)) )
      stop("censoring indicator must be 0 (censored) or 1 (failed) and not identical")
  }
  
  # decide which model to fit 
  
  if (is.null(model))
  {
    model = "regression" 

    if (!is.null(censor))
      model = "survival"
    
    if (is.factor(y))
      model = "classification"    
    
    if (is.numeric(y) & length(unique(y)) < 5 )
      stop("Number of unique values in y is too small. Please check input data and/or change to classification.")
  }
  
  return(model)
}



