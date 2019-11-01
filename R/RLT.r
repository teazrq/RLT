#' @title Reinforcement Learning Trees
#' @description Fit models for regression, classification and survival analysis using reinforced splitting rules. The model reduces to regular random forests if reinforcement is turned off
#' @param x A matrix or data.frame for features
#' @param y Response variable, a numeric/factor vector or a Surv object
#' @param censor The censoring indicator if survival model is used
#' @param model The model type: \code{regression}, \code{classification} and \code{survival}
#' @param reinforcement Should reinforcement splitting rule be used. Default is \cdoe{FALSE}, meaning regular random forests. Embedded model tuning parameters are automatically chosen. They can also be specified in \code{RLT.control}.
#' @param ntrees Number of trees, \code{ntrees = 100} if use reinforcement, \code{ntrees = 1000} otherwise
#' @param mtry Number of variables used at each internal node, only for \code{reinforcement = FALSE}
#' @param nmin Terminal node size. Splitting will stop when the internal node size is less than twice of \code{nmin}
#' @param alpha Minimum number of observations required for each child node as a portion of the parent node. Must be within \code{(0, 0.5]}. This is only effective when \code{split.gen} is \code{random} or \code{best}
#' @param split.gen How the cutting points are generated: \code{random}, \code{rank} and \code{best}. The \code{rank} and \code{best} will try to ensure terminal node size while \code{random} does not guarantee it. 
#' @param split.rule splitting rule for comparison
#' @param nsplit Number of random cutting points to compare for each variable at an internal node
#' @param replacement Whether the in-bag samples are sampled with replacement
#' @param resample.prob Proportion of in-bag samples
#' @param obs.w Observation weights
#' @param var.w Variable weights. When \code{split.rule} is not \code{penalized}, this performs weighted sample of \code{mtry} variables to select the splitting variable. When \code{split.rule = penalized}, this is treated as penalty.
#' @param importance Should importance measures be calculated
#' @param track.obs Track which terminal node the observation belongs to
#' @param RLT.control a list of tuning parameters for embedded model in reinforcement splitting rule
#' @param kernel.ready Should kernel information be saved for later use? This is mainly for obtainning bootstraped confidence intervels.
#' @param seed random seed using dqrng::xoshiro256plus generator 
#' @param verbose Whether fitting should be printed
#' @param ncores Number of cores
#' @param ... additional arguments
#' @return A \code{RLT} object; a list consisting of
#' \item{FittedForest}{Fitted tree structures}
#' \item{ObsTrack}{An indicator matrix for whether each observation is used in each fitted tree}
#' \item{VarImp}{Variable importance measures, if \code{importance = TRUE}}
#' \item{NodeRegi}{Observation id info in each terminal node, if \code{kernel.ready = TRUE}}
#' \item{OOBPred}{Out-of-bag prediction values}
#' @references Zhu, R., Zeng, D., & Kosorok, M. R. (2015) "Reinforcement Learning Trees." Journal of the American Statistical Association. 110(512), 1770-1784.
#' @references Zhu, R., & Kosorok, M. R. (2012). "Recursively Imputed Survival Trees." Journal of the American Statistical Association, 107(497), 331-340.

RLT <- function(x, y, censor = NULL, model = NULL, reinforcement = FALSE,
        				ntrees = if (reinforcement) 100 else 500,
        				mtry = max(1, as.integer(ncol(x)/3)),
        				nmin = max(1, as.integer(log(nrow(x)))),
        				alpha = 0,
        				split.gen = "random",
        				split.rule = NULL,
        				nsplit = 1,
        				replacement = TRUE,
        				resample.prob = if(replacement) 1 else 0.9,
        				obs.w = NULL,
        				var.w = NULL,
        				importance = FALSE,
        				track.obs = FALSE,
        				RLT.control = list("RLT"= FALSE),
        				kernel.ready = FALSE,
        				seed = NA,
        				ncores = 1,
        				verbose = 0,
        				...)
{
  # check inputs
  
  if (missing(x)) stop("x is missing")
  if (missing(y)) stop("y is missing")
  
  # check model type
  model = check_input(x, y, censor, model)

  p = ncol(x)
  n = nrow(x)

  # check RF parameters
  param <- check_param(n, p,
                       ntrees, mtry, nmin,
                       alpha, split.gen, split.rule, nsplit,
                       replacement, resample.prob,
                       importance, reinforcement, kernel.ready)
  
  # check RLT parameters
  if (reinforcement)
    RLT.control <- check_RLT_control(RLT.control)
  
  # check observation weights  
  if (is.null(obs.w))
  {
    param$"use.obs.w" = 0L
    obs.w = 0L
  }else{
    param$"use.obs.w" = 1L
    obs.w = as.numeric(as.vector(obs.w))
    
    if (any(obs.w <= 0))
      stop("observation weights cannot be negative or zero")
    
    storage.mode(obs.w) <- "double"    
    obs.w = obs.w/sum(obs.w)
    
    if (length(obs.w) != nrow(x))
      stop("length of observation weights must be n")
  }
  
  # check variable weights  
  if (is.null(var.w))
  {
    param$"use.var.w" = 0L
    var.w = 0L
  }else{
    param$"use.var.w" = 1L
    
    var.w = as.numeric(as.vector(var.w))
    var.w = pmax(0, var.w)
    
    if (sum(var.w) <= 0) stop("variable weights must contain positive values")
    
    storage.mode(var.w) <- "double"
    var.w = var.w/sum(var.w)
    
    if (length(var.w) != ncol(x)) stop("length of variable weights must be p")
  }
  
  # prepare x, continuous and categorical
  if (is.data.frame(x))
  {
    # data.frame, check for categorical variables
    xlevels <- lapply(x, function(x) if (is.factor(x)) levels(x) else 0)
    ncat <- sapply(xlevels, length)
    ## Treat ordered factors as numerics.
    ncat <- ifelse(sapply(x, is.ordered), 1, ncat)
  }else{
    # numerical matrix for x, all continuous
    ncat <- rep(1, p)
    names(ncat) <- colnames(x)
    xlevels <- as.list(rep(0, p))
  }
  
  storage.mode(ncat) <- "integer"
  
  if (max(ncat) > 53)
    stop("Cannot handle categorical predictors with more than 53 categories")
  
  xnames = colnames(x)
  x <- data.matrix(x)
  
  # other things
  
  storage.mode(verbose) <- "integer"
  ncores = as.integer(max(1, ncores))
  storage.mode(ncores) <- "integer"
  
  if (is.na(seed))
    seed = runif(1)*1e8

  param$"seed" = as.integer(seed)
    
	# fit model

	if (model == "regression")
	{
	  RLT.fit = RegForest(x, y,
	                      ncat,
	                      param,
	                      RLT.control,
	                      obs.w,
	                      var.w,
	                      ncores,
	                      verbose,
	                      ...)
	}
  
  if (model == "survival")
  {
    cat(" run survival forest ")
    
    RLT.fit = SurvForest(x, y, censor,
                         ncat,
                         param,
                         RLT.control,
                         obs.w,
                         var.w,
                         ncores,
                         verbose,
                         ...)
  }
  
  
  
  RLT.fit$"xnames" = xnames

	return(RLT.fit)
}
