#' @title check_param
#' @name check_param
#' @description Check parameters
#' @keywords internal

check_param <- function(n, p, ntrees, 
                        mtry, nmin, alpha, 
                        split.gen, split.rule, nsplit, 
                        replacement, resample.prob,
                        importance, reinforcement, 
                        track.obs)
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
  
  track.obs = (track.obs != 0)
  storage.mode(track.obs) <- "integer"  
  
  return(param <- list("n" = n,
                       "p" = p,
                       "ntrees" = ntrees,
                       "mtry" = mtry,
                       "nmin" = nmin,
                       "alpha" = alpha,
                       "split.gen" = split.gen,
                       "split.rule" = split.rule,
                       "nsplit" = nsplit,
                       "replacement" = replacement,
                       "resample.prob" = resample.prob,
                       "importance" = importance,
                       "reinforcement" = reinforcement,
                       "track.obs" = track.obs))
}


#' @title check_RLT_param
#' @name check_RLT_param
#' @description Check parameters for RLT method
#' @keywords internal

check_RLT_param <- function(control)
{

  if (!is.list(control)) {
    stop("RLT.control must be a list of tuning parameters")
  }
  
  if (is.null(control$embed.ntrees)) {
    embed.ntrees <- 50
  } else embed.ntrees = max(control$embed.ntrees, 1)
  
  storage.mode(embed.ntrees) <- "integer"
  
  if (is.null(control$embed.resample.prob)) {
    embed.resample.prob <- 0.8
  } else embed.resample.prob = max(0, min(control$embed.resample.prob, 1))
  
  storage.mode(embed.resample.prob) <- "double"
  
  if (is.null(control$embed.mtry.prop)) { # for embedded model, mtry is proportion
    embed.mtry.prop <- 1/2
  } else embed.mtry.prop = max(min(control$embed.mtry.prop, 1), 0)
  
  storage.mode(embed.mtry.prop) <- "double"
  
  if (is.null(control$embed.nmin)) {
    embed.nmin <- 5
  } else embed.nmin = max(1, floor(control$embed.nmin))

  storage.mode(embed.nmin) <- "double"
  
  if (is.null(control$embed.split.gen)) {
    embed.split.gen <- 1
  } else embed.split.gen = match(control$embed.split.gen, c("random", "rank", "best"))
  
  storage.mode(embed.split.gen) <- "integer"
  
  if (is.null(control$embed.nsplit)) {
    embed.nsplit <- 1
  } else embed.nsplit = max(1, control$embed.nsplit)
  
  storage.mode(embed.nsplit) <- "integer"
  
  return(list("embed.ntrees" = embed.ntrees,
               "embed.resample.prob" = embed.resample.prob,
               "embed.mtry.prop" = embed.mtry.prop,
               "embed.nmin" = embed.nmin,
               "embed.split.gen" = embed.split.gen,
               "embed.nsplit" = embed.nsplit))
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
    
    if ( is.numeric(y) & length(unique(y)) < 5 )
      warning("Number of unique values in y is too small. Please check input data and/or consider changing to classification.")
  }
  
  return(model)
}



