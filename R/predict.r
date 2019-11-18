#' @title prediction using RLT
#' @description Predict the outcome (regression, classification or survival) using a fitted RLT object
#' @param object A fitted RLT object
#' @param testx the testing samples, must have the same structure as the training samples
#' @param keep.all whether to keep the prediction from all trees
#' @param ncores number of cores
#' @param ... ...
#' @export

predict.RLT<- function(object, 
                       testx = NULL, 
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
    
    pred <- RegForestUniPred(object$FittedForest$NodeType,
                             object$FittedForest$SplitVar,
                             object$FittedForest$SplitValue,
                             object$FittedForest$LeftNode,
                             object$FittedForest$RightNode,
                             object$FittedForest$NodeSize,                             
                             object$FittedForest$NodeAve,
                             testx,
                             object$ncat,
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
    
    pred <- SurvForestUniPred(object$FittedForest$NodeType,
                              object$FittedForest$SplitVar,
                              object$FittedForest$SplitValue,
                              object$FittedForest$LeftNode,
                              object$FittedForest$RightNode,
                              object$FittedForest$NodeSize,
                              object$FittedForest$NodeHaz,
                              testx,
                              object$ncat,
                              length(object$timepoints),
                              keep.all,
                              ncores,
                              verbose)
    
    class(pred) <- c("RLT", "pred", "surv")
    return(pred)
  }
}
