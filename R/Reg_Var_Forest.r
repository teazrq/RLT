#' @title                 Regression random forest with variance estimation
#' @description           This function provides some experimental features 
#'                        to estimate the variance of random forests, especially
#'                        when the sub-sampling size is larger than \eqn{n/2}. 
#'                        These trees will always be sampled without replacement.
#'                        The choices of tuning parameters will be limited.
#'                        Use at your own risk.
#'                        
#' @param x               A `matrix` or `data.frame` of features
#' 
#' @param y               Response variable. a `numeric`/`factor` vector.
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
#' @return                Prediction and variance estimation of the testing data
#' 
#' @export Reg_Var_Forest
Reg_Var_Forest <- function(x, y, testx,
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
  model = check_input(x, y, censor = NULL, model = "regression")
  
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
    RLT.fit <- RLT(x = x, y = y,
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
    
    RLT.pred = predict(RLT.fit, testx, var.est = TRUE, ncores = ncores)
    
    resultMat = list("Prediction" = RLT.pred$Prediction,
                     "var" = RLT.pred$Variance, 
                     "Fit" = RLT.fit)
    
    class(resultMat) <- c("RLT", "Var", "reg")
    
    return(resultMat)
    
  }else{

    RLT.fit <- RLT(x = x, y = y,
                   ntrees = ntrees,
                   mtry = mtry,
                   nmin = nmin,
                   split.gen = split.gen,
                   nsplit = nsplit,
                   resample.replace = FALSE,
                   resample.prob = resample.prob,
                   param.control = param.control,
                   ncores = ncores,
                   verbose = verbose,
                   seed = seed, ...)

    BS.fit <- RLT(x = x, y = y,
                  ntrees = ntrees,
                  mtry = mtry,
                  nmin = nmin,
                  split.gen = split.gen,
                  nsplit = nsplit,
                  resample.replace = TRUE,
                  resample.prob = resample.prob,
                  param.control = param.control,
                  ncores = ncores,
                  verbose = verbose,
                  seed = seed + 1, ...)
    
    RLT.pred = predict(RLT.fit, testx, ncores = ncores, keep.all = TRUE)
    BS.pred = predict(BS.fit, testx, ncores = ncores, keep.all = TRUE)
    
    Var = (1 + 1/ntrees) * apply(BS.pred$PredictionAll, 1, var) - 
          (1 - 1/ntrees) * apply(RLT.pred$PredictionAll, 1, var)

    resultMat = list("Prediction" = RLT.pred$Prediction,
                     "var" = Var, 
                     "Vh" = apply(BS.pred$PredictionAll, 1, var),
                     "Vs" = apply(RLT.pred$PredictionAll, 1, var),
                     "Fit" = RLT.fit)
    
    class(resultMat) <- c("RLT", "Var", "reg")
    
    return(resultMat)
  }
  
}
