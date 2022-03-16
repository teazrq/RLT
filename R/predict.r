#' @title prediction using RLT
#' @description Predict the outcome (regression, classification or survival) 
#'              using a fitted RLT object
#' @param object   A fitted RLT object
#' @param testx    The testing samples, must have the same structure as the 
#'                 training samples
#' @param var.est  Whether to estimate the variance of each testing data. 
#'                 The original forest must be fitted with \code{var.ready = TRUE}.
#' @param keep.all whether to keep the prediction from all trees
#' @param ncores   number of cores
#' @param ... ...
#'
#' @return 
#' 
#' A \code{RLT} prediction object, constructed as a list consisting
#' 
#' \item{Prediction}{Prediction}
#' \item{Variance}{if \code{var.est = TRUE} and the fitted object is 
#'                 \code{var.ready = TRUE}}
#'
#' @export

predict.RLT<- function(object,
                       testx = NULL,
                       var.est = FALSE,
                       keep.all = FALSE,
                       ncores = 1,
                       verbose = 0,
                       ...)
{

  if (is.null(testx))
  {
    return(object$OOBPrediction)
  }
  
  if (!is.matrix(testx) & !is.data.frame(testx)) stop("testx must be a matrix or a data.frame")
  
  if( class(object)[2] == "fit" &  class(object)[3] == "reg" )
  {
    # check test data 

    if (is.null(colnames(testx)))
    {
      if (ncol(testx) != object$parameters$p) stop("test data dimension does not match training data, variable names are not supplied...")
    }else if (any(colnames(testx) != object$variablenames)){
      
      warning("test data variables names does not match training data...")
      varmatch = match(object$variablenames, colnames(testx))
      if (any(is.na(varmatch))) stop("test data missing some variables...")
      testx = testx[, varmatch]
    }

    testx <- data.matrix(testx)
    
    if (var.est & !object$parameters$var.ready)
      stop("The original forest is not fitted with `var.ready` Please check the conditions and build another forest.")

    pred <- RegUniForestPred(object$FittedForest$SplitVar,
                             object$FittedForest$SplitValue,
                             object$FittedForest$LeftNode,
                             object$FittedForest$RightNode,
                             object$FittedForest$NodeAve,
                             testx,
                             object$ncat,
                             var.est,
                             keep.all,
                             ncores,
                             verbose)
    
    class(pred) <- c("RLT", "pred", "reg")
    return(pred)
  }
  
  if( class(object)[2] == "fit" &  class(object)[3] == "surv" ) 
  {
    # check test data 
    
    if (is.null(colnames(testx)))
    {
      if (ncol(testx) != object$parameters$p) stop("test data dimension does not match training data, variable names are not supplied...")
    }else if (any(colnames(testx) != object$variablenames)){
      
      warning("test data variables names does not match training data...")
      
      varmatch = match(object$variablenames, colnames(testx))
      
      if (any(is.na(varmatch))) stop("test data missing some variables...")
      testx = testx[, varmatch]
    }
    
    testx <- data.matrix(testx)
    
    pred <- SurvUniForestPred(object$FittedForest$SplitVar,
                              object$FittedForest$SplitValue,
                              object$FittedForest$LeftNode,
                              object$FittedForest$RightNode,
                              object$FittedForest$NodeHaz,
                              testx,
                              object$ncat,
                              length(object$timepoints),
                              var.est,
                              keep.all,
                              ncores,
                              verbose)
    
    class(pred) <- c("RLT", "pred", "surv")
    return(pred)
  }
}
