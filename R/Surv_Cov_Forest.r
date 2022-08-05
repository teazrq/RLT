#' @title                 Survival random forest with covariance estimation
#' @description           These trees will always be sampled without replacement.
#'                        The choices of tuning parameters will be limited.
#'                        Use at your own risk.
#'                        
#' @param x               A `matrix` or `data.frame` of features
#' 
#' @param y               Response variable. a `numeric`/`factor` vector.
#'                        
#' @param censor          The censoring indicator if survival model is used.
#'                        
#' @param testx           A `matrix` or `data.frame` of testing data
#'                        
#' @param ntrees          Number of trees. To obtain a stable and accurate 
#'                        estimation, the default is `ntrees = 10000`.
#'                        
#' @param mtry            Number of randomly selected variables used at each 
#'                        internal node.
#'                        
#' @param nmin            Terminal node size. Same as used in \code{RLT}.
#'                        
#' @param resample.prob   Proportion of in-bag samples. Can be larger than 0.5.
#'                        
#' @param split.gen       `"random"`, `"rank"` or `"best"`.
#' 
#' @param nsplit          Number of random cutting points to compare for each 
#'                        variable at an internal node.
#'                        
#' @param resample.prob   Proportion of in-bag samples.
#' 
#' @param param.control   A list of additional parameters. However, choices are
#'                        limited. For example, reinforcement splitting rules 
#'                        are not implemented. 
#'                        
#' @param ncores          Number of cores. Default is 0 (using all available cores).
#' 
#' @param verbose         Whether fitting info should be printed.
#' 
#' @param seed            Random seed number to replicate a previously fitted forest. 
#'                        Internally, the `xoshiro256++` generator is used. If not specified, 
#'                        a seed will be generated automatically. 
#'                        
#' @param ...             Additional arguments.
#' 
#' @return                Prediction and covariance estimation of the testing data
#' 
#' @export Surv_Cov_Forest
Surv_Cov_Forest <- function(x, y, censor, testx,
                  			   ntrees = 10000,
                  			   mtry = max(1, as.integer(ncol(x)/3)),
                  			   nmin = max(1, as.integer(log(nrow(x)))),
                  			   split.gen = "best",
                  			   nsplit = 0,
                  			   resample.prob = 0.5, 
                  			   param.control = list(),
                  			   ncores = 1,
                  			   verbose = 0,
                  			   seed = NULL,
                  			   ...)
{
  # check inputs
  if (missing(x)) stop("x is missing")
  if (missing(y)) stop("y is missing")
  
  if (!is.matrix(testx) & !is.data.frame(testx)) stop("testx must be a matrix or a data.frame")
  if (any(is.na(testx))) stop("NA not permitted in testx")

  # check model type
  model = check_input(x, y, censor = NULL, model = "survival")
  
  # parameters
  p = ncol(x)
  n = nrow(x)
  
  ntrees = max(ntrees, 1)
  storage.mode(ntrees) <- "integer"
  
  resample.prob = max(0, min(resample.prob, 1))
  storage.mode(resample.prob) <- "double"
  
  if (is.null(seed) | !is.numeric(seed))
    seed = runif(1) * .Machine$integer.max
  
  # construct resample.preset

  if (resample.prob <= 0.5)
  {
    # use the default RLT 
    RLT.fit <- RLT(x = x, y = y, censor=censor,
                   ntrees = ntrees,
                   mtry = mtry,
                   nmin = nmin,
                   split.gen = split.gen,
                   nsplit = nsplit,
                   resample.replace = FALSE,
                   resample.prob = resample.prob, 
                   var.ready = TRUE,
                   param.control = param.control,
                   ncores = ncores,
                   verbose = verbose,
                   seed = seed, ...)
    
    RLT.pred = predict(RLT.fit, testx, var.est = TRUE,
                       ncores = ncores)
    
    resultMat = list("Survival" = RLT.pred$Survival,
                     "CumHazard" = RLT.pred$CumHazard,
                     "hazard" = RLT.pred$hazard,
                     "Cov" = RLT.pred$Cov, 
                     "Var" = RLT.pred$Var, 
                     "Fit" = RLT.fit)

    class(resultMat) <- c("RLT", "Var", "surv")
    
    return(resultMat)
    
  }else{
    cat("Covariance estimation for survival random forests with resample.prob>0.5 not available.\n")    
    cat("Set resample.prob<=0.5.\n")    
    
  }
  
}
