#' @title Get kernel weight for a suject
#' @description Get the kernel weights induced from a random forests for 
#'              predicting test sujects
#' @param object A fitted RLT object
#' @param testx the testing data
#' @param ... ...
#' @examples

getKernelWeight <- function(object, testx, ncores = 1, verbose = FALSE, ...)
{
  
  if (is.null(testx))
  {
    cat("in sample kernel weights not implemented yet")
  }
  
  if (!is.matrix(testx) & !is.data.frame(testx)) stop("testx must be a matrix or a data.frame")
  
  
  if( class(object)[2] == "fit" )
  {
    # check test data 
    
    if (!RLTfit$parameters$kernel.ready)
      stop("the fitted object is not kernel ready")
    
    if (is.null(colnames(testx)))
    {
      if (ncol(testx) != object$parameters$p) stop("test data dimension does not match training data, variable names are not supplied...")
    }else if (any(colnames(testx) != object$variablenames))
    {
      warning("test data variables names does not match training data...")
      varmatch = match(object$variablenames, colnames(testx))
      if (any(is.na(varmatch))) stop("test data missing some variables...")
      testx = testx[, varmatch]
    }
    
    testx <- data.matrix(testx)
    
    K <- ForestKernelUni(object$FittedForest$NodeType,
                         object$FittedForest$SplitVar,
                         object$FittedForest$SplitValue,
                         object$FittedForest$LeftNode,
                         object$FittedForest$RightNode,
                         object$FittedForest$NodeSize,
                         object$NodeRegi,
                         object$ObsTrack,
                         testx,
                         object$ncat,
                         object$obs.w,
                         object$parameters$use.obs.w,
                         ncores,
                         verbose)
    
    class(K) <- c("RLT", "kernel")
    return(K)
  }
  
}
