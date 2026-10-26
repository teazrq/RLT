#' @title Compute a random forest kernel
#' @description Measure how often observations share terminal nodes in a fitted
#' forest, within one dataset or between two datasets.
#' This is an experimental feature; the available modes and
#' their data requirements are described below.
#' @details Ordinary kernel entries count trees in which two observations share
#' a terminal node. They are not normalized observation contributions to a forest
#' prediction. The training-data mode includes in-bag multiplicities; the OOB
#' mode instead normalizes by the number of trees eligible for each pair.
#'                  
#' @param object    A fitted RLT object.
#' 
#' @param X1        The dataset for prediction. This calculates an \eqn{n_1 \times n_1} kernel
#'                  matrix of \code{X1}. 
#' 
#' @param X2        The dataset for reference/training. 
#'                  If \code{X2} is supplied, then calculate an \eqn{n_1 \times n_2} 
#'                  kernel matrix. If \code{vs.train} is used, then this must be the original 
#'                  training data.
#' 
#' @param vs.train  Include each reference observation's in-bag multiplicity in
#'                  its leaf-sharing count. \code{X2} must be the original training
#'                  data in its original row order. To use this feature, you must
#'                  specify \code{resample.track = TRUE} in \code{param.control} when fitting 
#'                  the forest 
#'                  
#' @param oob       Logical. If \code{TRUE}, compute the OOB (out-of-bag) self-kernel, which 
#'                  counts co-occurrence only from trees where both observations are OOB. 
#'                  Each contributing tree was fitted without either observation
#'                  in the pair. Requires \code{resample.track = TRUE}
#'                  and \code{X1} to be the original training data. \code{X2} must be 
#'                  \code{NULL} (OOB kernel is defined for self-kernel only). The returned
#'                  list contains three matrices: \code{Kernel} (normalized co-occurrence
#'                  in \eqn{[0,1]} where \code{N > 0}), \code{N} (number of trees
#'                  where both are OOB), and \code{C} (number of trees where both
#'                  are OOB and share a leaf). Entries with \code{N = 0} are
#'                  returned as zero; inspect \code{N} to distinguish these
#'                  unsupported pairs from observed zero leaf sharing.
#'                  
#' @param verbose   Whether fitting should be printed.
#' 
#' @param ... ...   Additional arguments.
#' 
#' @return A list containing the kernel matrix. For \code{oob = TRUE}, the list also 
#'   contains \code{N} (OOB co-occurrence count) and \code{C} (OOB leaf-sharing count).
#' 
#' @export
#' @examples
#' \donttest{
#'   set.seed(42)
#'   x <- matrix(rnorm(200 * 5), ncol = 5)
#'   y <- rowSums(x[, 1:2]) + rnorm(200)
#'   fit <- RLT(x, y, ntrees = 100)
#'   K <- forest.kernel(fit, X1 = x[1:5, ])
#'   print(K$Kernel[1:3, 1:3])
#' }

forest.kernel <- function(object,
                          X1 = NULL,
                          X2 = NULL,
                          vs.train = FALSE,
                          oob = FALSE,
                          verbose = FALSE,
                          ...)
{
  if( class(object)[[2]] != "fit" )
    stop("object must be a fitted RLT object")

  if (is.null(X1))
    stop("X1 must be provided. To compute the kernel on training data, pass the training data as X1 and set X2 = NULL.")

    X1 <- encode_RLT_features(object, X1, "X1")
    
    if (is.null(X2))
    {
      
      if (oob)
      {
        # OOB self-kernel: requires resample.track = TRUE
        if (is.null(object$ObsTrack))
          stop("Must set resample.track = TRUE in param.control to use oob = TRUE")
        
        if (nrow(X1) != object$parameters$n)
          stop("OOB kernel requires X1 to be the original training data")
        
        if (object$parameters$linear.comb > 1)
        {
          K <- Kernel_Self_OOB_Comb(object$FittedForest$SplitVar,
                                    object$FittedForest$SplitLoad,
                                    object$FittedForest$SplitValue,
                                    object$FittedForest$LeftNode,
                                    object$FittedForest$RightNode,
                                    object$FittedForest$NodeWeight,
                                    X1,
                                    object$ncat,
                                    object$ObsTrack,
                                    verbose)
        }else{
          K <- Kernel_Self_OOB(object$FittedForest$SplitVar,
                               object$FittedForest$SplitValue,
                               object$FittedForest$LeftNode,
                               object$FittedForest$RightNode,
                               object$FittedForest$NodeWeight,
                               X1,
                               object$ncat,
                               object$ObsTrack,
                               verbose)
        }
        
        class(K) <- c("RLT", "kernel", "self", "oob")
        
      }else{
        
        if (object$parameters$linear.comb > 1)
        {
          K <- Kernel_Self_Comb(object$FittedForest$SplitVar,
                                object$FittedForest$SplitLoad,
                                object$FittedForest$SplitValue,
                                object$FittedForest$LeftNode,
                                object$FittedForest$RightNode,
                                object$FittedForest$NodeWeight,
                                X1,
                                object$ncat,
                                verbose)
        }else{
          K <- Kernel_Self(object$FittedForest$SplitVar,
                           object$FittedForest$SplitValue,
                           object$FittedForest$LeftNode,
                           object$FittedForest$RightNode,
                           object$FittedForest$NodeWeight,
                           X1,
                           object$ncat,
                           verbose)
        }
        
        class(K) <- c("RLT", "kernel", "self")
      }
      
    }else{
      
      if (oob)
        stop("oob = TRUE is only supported for self-kernel (X2 must be NULL)")
    
      X2 <- encode_RLT_features(object, X2, "X2")
      
      if (!vs.train)
      {
        # cross-kernel of X1 and X2
    
        if (object$parameters$linear.comb > 1)
        {
          K <- Kernel_Cross_Comb(object$FittedForest$SplitVar,
                                 object$FittedForest$SplitLoad,
                                 object$FittedForest$SplitValue,
                                 object$FittedForest$LeftNode,
                                 object$FittedForest$RightNode,
                                 object$FittedForest$NodeWeight,
                                 X1,
                                 X2,
                                 object$ncat,
                                 verbose)
        }else{
          K <- Kernel_Cross(object$FittedForest$SplitVar,
                            object$FittedForest$SplitValue,
                            object$FittedForest$LeftNode,
                            object$FittedForest$RightNode,
                            object$FittedForest$NodeWeight,
                            X1,
                            X2,
                            object$ncat,
                            verbose)
        }
        
        class(K) <- c("RLT", "kernel", "cross")
        
      }else{
        
        # kernel matrix as to the training process 
        # resample.track must be TRUE 
        if ( is.null(object$ObsTrack) )
          stop("Must set resample.track = TRUE in param.control to perform vs.train")
      
        ObsTrack = object$ObsTrack
      
        if (nrow(ObsTrack) != nrow(X2))
          stop("X2 must be the original training data")
        
        if (object$parameters$linear.comb > 1)
        {
          K <- Kernel_Train_Comb(object$FittedForest$SplitVar,
                                 object$FittedForest$SplitLoad,
                                 object$FittedForest$SplitValue,
                                 object$FittedForest$LeftNode,
                                 object$FittedForest$RightNode,
                                 object$FittedForest$NodeWeight,
                                 X1,
                                 X2,
                                 object$ncat,
                                  ObsTrack,
                                 verbose)
        }else{
          K <- Kernel_Train(object$FittedForest$SplitVar,
                            object$FittedForest$SplitValue,
                            object$FittedForest$LeftNode,
                            object$FittedForest$RightNode,
                            object$FittedForest$NodeWeight,
                            X1,
                            X2,
                            object$ncat,
                            ObsTrack,
                            verbose)
        }
        
        class(K) <- c("RLT", "kernel", "train")
        
      }
    }

  return(K)
}
