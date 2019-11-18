#' @title                 Reinforcement Learning Trees
#' @description           Fit models for regression, classification and 
#'                        survival analysis using reinforced splitting rules.
#'                        The model reduces to regular random forests if 
#'                        reinforcement is turned off.
#' @param x               A `matrix` or `data.frame` of features
#' 
#' @param y               Response variable. a `numeric`/`factor` vector.
#'                        
#' @param censor          The censoring indicator if survival model is used.
#' 
#' @param model           The model type: `"regression"`, `"classification"` 
#'                        or `"survival"`.
#'                        
#' @param reinforcement   Should reinforcement splitting rule be used. Default
#'                        is `"FALSE"`, i.e., regular random forests. When it
#'                        is activated, embedded model tuning parameters are 
#'                        automatically chosen. They can also be specified in 
#'                        `RLT.control`.
#'                        
#' @param ntrees          Number of trees, `ntrees = 100` if reinforcement is
#'                        used and `ntrees = 1000` otherwise.
#'                        
#' @param mtry            Number of randomly selected variables used at each 
#'                        internal node.
#'                        
#' @param nmin            Terminal node size. Splitting will stop when the 
#'                        internal node size is less than twice of `nmin`. This
#'                        is equivalent to setting `nodesize` = 2*`nmin` in the
#'                        `randomForest` package.
#'                        
#' @param alpha           Minimum number of observations required for each 
#'                        child node as a portion of the parent node. Must be 
#'                        within `[0, 0.5)`. When `alpha` $> 0$ and `split.gen`
#'                        is `rank` or `best`, this will force each child node 
#'                        to contain at least \eqn{\max(\texttt{nmin}, \alpha \times N_A)}
#'                        number of number of observations, where $N_A$ is the 
#'                        sample size at the current internal node. This is 
#'                        mainly for theoritical concern. 
#'                        
#' @param split.gen       How the cutting points are generated: `"random"`, 
#'                        `"rank"` or `"best"`. `"random"` performs random 
#'                        cutting point and does not take `alpha` into 
#'                        consideration. `"rank"` could be more effective when 
#'                        there are a large number of ties. It can also be used 
#'                        to guarantee child node size if `alpha` > 0. `"best"` 
#'                        finds the best cutting point, and can be cominbed with 
#'                        `alpha` too.
#'                        
#' @param split.rule      Splitting rule for comparison:
#'                        \item{regression}{`"var"` for variance reduction}
#'                        \item{survival}{`"logrank"`, `"suplogrank"`, 
#'                                        `"LL"` and `"penLL"`}
#'                        \item{classification}{`"gini"`}
#' 
#' @param nsplit          Number of random cutting points to compare for each 
#'                        variable at an internal node.
#'                        
#' @param replacement     Whether the in-bag samples are sampled with 
#'                        replacement.
#'                        
#' @param resample.prob   Proportion of in-bag samples.
#' 
#' @param obs.w           Observation weights
#' 
#' @param var.w           Variable weights. When `"split.rule"` is not
#'                        `"penLL"`, this performs weighted sample of `"mtry"` 
#'                        variables to select the splitting variable. Otherwise
#'                        this is treated as the penalty.
#'                        
#' @param importance      Should importance measures be calculated
#' 
#' @param track.obs       Track which terminal node the observation belongs to.
#'                        The fitted object will return `"ObsTrack"` as the 
#'                        indicator/counts of in-bag data. 
#' 
#' @param ObsTrack        Pre-specified matrix for in-bag data indicator/count 
#'                        matrix. It will not be used if it contains any 
#'                        negative value. This is an experimental feature, try 
#'                        at your own risk. 
#'                        
#' @param RLT.control     A list of tuning parameters for embedded model in 
#'                        reinforcement splitting rule. See \code{RLT.control}.
#'                        
#' @param seed            Random seed using the `Xoshiro256+` generator.
#' 
#' @param ncores          Number of cores. Default is 1.
#' 
#' @param verbose         Whether fitting should be printed.
#' 
#' @param ...             Additional arguments.
#' 
#' @export
#' 
#' @return 
#' 
#' A \code{RLT} object, constructed as a list consisting
#' 
#' \item{FittedForest}{Fitted tree structures}
#' \item{VarImp}{Variable importance measures, if `importance = TRUE`}
#' \item{Prediction}{In-bag prediction values}
#' \item{OOBPrediction}{Out-of-bag prediction values}
#' \item{ObsTrack}{An indicator matrix for whether each observation is used in 
#'                 each fitted tree}
#'                 
#' @references Zhu, R., Zeng, D., & Kosorok, M. R. (2015) "Reinforcement Learning Trees." Journal of the American Statistical Association. 110(512), 1770-1784.
#' @references Zhu, R., & Kosorok, M. R. (2012). "Recursively Imputed Survival Trees." Journal of the American Statistical Association, 107(497), 331-340.

RLT <- function(x, y, censor = NULL, model = NULL, 
                reinforcement = FALSE,
        				ntrees = if (reinforcement) 100 else 500,
        				mtry = max(1, as.integer(ncol(x)/3)),
        				nmin = max(1, as.integer(log(nrow(x)))),
        				alpha = 0,
        				split.gen = "random",
        				split.rule = NULL,
        				nsplit = 1,
        				replacement = TRUE,
        				resample.prob = if(replacement) 1 else 0.85,
        				obs.w = NULL,
        				var.w = NULL,
        				importance = FALSE,
        				track.obs = FALSE,
        				ObsTrack = NaN,
        				RLT.control = list("RLT"= FALSE),
        				seed = NaN,
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
  param <- check_param(n, p, ntrees, 
                       mtry, nmin, alpha, 
                       split.gen, split.rule, nsplit, 
                       replacement, resample.prob,
                       importance, reinforcement, 
                       track.obs)
  
  # check RLT parameters
  if (reinforcement)
    RLT.control <- check_RLT_param(RLT.control)
  
  # check ObsTrack
  if ( !any(is.na(ObsTrack)) )
  {
      if (!is.matrix(ObsTrack))
          stop("ObsTrack must be a matrix")
      
      if (nrow(ObsTrack) != n | ncol(ObsTrack) != ntrees)
          stop("Number of rows in ObsTrack must be the same as x;\n Number of columns in ObsTrack must be the same as ntrees")      
  
      param$pre.obs.track = TRUE
  }else{
      ObsTrack = as.matrix(0);
      param$pre.obs.track = FALSE
  }
      
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
    seed = .Machine$integer.max * runif(1)

  param$"seed" = as.integer(seed)
    
	# fit model

	if (model == "regression")
	{
	  RLT.fit = RegForest(x, y, ncat,
	                      param, RLT.control,
	                      obs.w, var.w,
	                      ncores, verbose,
	                      ObsTrack,
	                      ...)
	}
  
  if (model == "survival")
  {
    cat(" run survival forest ")
    
    RLT.fit = SurvForest(x, y, censor, ncat,
                         param, RLT.control,
                         obs.w, var.w,
                         ncores, verbose,
                         ObsTrack,
                         ...)
  }

  RLT.fit$"xnames" = xnames

	return(RLT.fit)
}
