#' @title           kernel.RLT
#' 
#' @description     Get random forest induced kernel weights between two sets
#'                  of data.
#'                  
#' @param object    A fitted RLT object.
#' 
#' @param X1        The the first dataset.
#' 
#' @param X2        The the second dataset.
#' 
#' @param self      If `TRUE`, then calculate the n by n kernel matrix of `X1`.
#'                  If `FALSE` then calculate the kernel matrix between `X1` 
#'                  and `X2`. Default is `TRUE`.
#' 
#' @param as.train  If `X2`` is the training data and the kernel weights
#'                  against the training data is desired. To perform this, the
#'                  fitted object must contain `ObsTrack` (using `track.obs`).
#'                  Defult is `FALSE`.
#' 
#' @param ncores    Number of cores. Default is 1.
#' 
#' @param verbose   Whether fitting should be printed.
#' 
#' @param ... ...   Additional arguments.
#' @export

forest.kernel <- function(object, 
                          X1, 
                          X2 = NULL,
                          as.train = FALSE,
                          ncores = 1, 
                          verbose = FALSE, 
                          ...)
{

  if (!is.matrix(X1) & !is.data.frame(X1)) stop("X1 must be a matrix or a data.frame")
  
  if( class(object)[2] != "fit" )
    stop("object must be a fitted RLT object")
  
  if (is.null(colnames(X1))){
    if (ncol(X1) != object$parameters$p) stop("test data dimension does not match training data, variable names are not supplied...")
  }else if (any(colnames(X1) != object$xnames)){
    warning("X1 data variables names does not match training data ...")
    varmatch = match(object$xnames, colnames(X1))
    
    if (any(is.na(varmatch))) stop("X1 missing some variables from the orignal training data ...")
    
    X1 = X1[, varmatch]
  }
  
  X1 <- data.matrix(X1)

  if (is.null(X2))
  {
    # X1 self kernel 

    K <- ForestKernelUni_Self(object$FittedForest$NodeType,
                             object$FittedForest$SplitVar,
                             object$FittedForest$SplitValue,
                             object$FittedForest$LeftNode,
                             object$FittedForest$RightNode,
                             object$FittedForest$NodeSize,
                             X1,
                             object$ncat,
                             ncores,
                             verbose)
    
  }else{
    # X1 X2 cross kernel 

    # check data 
    
    if (!is.null(X2))
      if (!is.matrix(X2) & !is.data.frame(X2)) stop("X2 must be a matrix or a data.frame")    
  
    if ( as.train & is.null(object$ObsTrack) )
      stop(" must save ObsTrack to perform kernel calculations ")
    
    ObsTrack = object$ObsTrack
    
    if ( !as.train & is.null(ObsTrack) )
      ObsTrack = X2 = matrix(0)
      
    if (is.null(colnames(X2))){
      if (ncol(X2) != object$parameters$p) stop("test data dimension does not match training data, variable names are not supplied...")
    }else if (any(colnames(X2) != object$xnames)){
      warning("X2 data variables names does not match training data ...")
      
      varmatch = match(object$xnames, colnames(X2))
      
      if (any(is.na(varmatch))) stop("X2 missing some variables from the orignal training data ...")
      
      X2 = X2[, varmatch]
    }
 
    K <- ForestKernelUni_Cross(object$FittedForest$NodeType,
                               object$FittedForest$SplitVar,
                               object$FittedForest$SplitValue,
                               object$FittedForest$LeftNode,
                               object$FittedForest$RightNode,
                               object$FittedForest$NodeSize,
                               X1,
                               X2,
                               object$ObsTrack,
                               object$ncat,
                               ncores,
                               verbose)

  }
  
  class(K) <- c("RLT", "kernel")
  
  return(K)

  
}
