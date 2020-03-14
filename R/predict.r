#' @title prediction using RLT
#' @description Predict the outcome (regression, classification or survival) using a fitted RLT object
#' @param object A fitted RLT object
#' @param testx the testing samples, must have the same structure as the training samples
#' @param treeindex if only a subset of trees are used for prediction, specify the index. The index should start with 0. This is an experimental feature. 
#' @param keep.all whether to keep the prediction from all trees
#' @param ncores number of cores
#' @param ... ...
#' @export

predict.RLT<- function(object, 
                       testx = NULL, 
                       treeindex = NULL,                       
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
  
  if (is.null(treeindex)){
    treeindex = c(1:object$parameters$ntrees) - 1
  }else{
    if (any(treeindex < 0 | treeindex >= object$parameters$ntrees))
      stop("treeindex out of bound")
  }
  
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
                             treeindex,
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
                              treeindex,
                              keep.all,
                              ncores,
                              verbose)
    
    class(pred) <- c("RLT", "pred", "surv")
    return(pred)
  }
}
