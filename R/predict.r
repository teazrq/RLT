#' @title prediction using RLT
#' @description Predict the outcome (regression, classification or survival) 
#'              using a fitted RLT object
#' @param object   A fitted RLT object
#' @param testx    The testing samples, must have the same structure as the 
#'                 training samples
#' @param var.est  Whether to estimate the variance of each testing data. 
#'                 The original forest must be fitted with \code{var.ready = TRUE}.
#'                 For survival forests, calculates the covariance matrix over all
#'                 observed time points and calculates critical value for the confidence 
#'                 band.
#' @param keep.all whether to keep the prediction from all trees. Warning: this can 
#'                 occupy a large storage space, especially in survival model
#' @param ncores   number of cores
#' @param verbose  print additional information
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
#' \strong{For Survival Forests}
#' \item{hazard}{predicted hazard functions}
#' \item{CumHazard}{predicted cumulative hazard function}
#' \item{Survival}{predicted survival function}
#' \item{Allhazard}{if \code{keep.all = TRUE}, the predicted hazard function for each 
#'                 observation and each tree}
#' \item{AllCHF}{if \code{keep.all = TRUE}, the predicted cumulative hazard function for each 
#'                 observation and each tree}
#' \item{Cov}{if \code{var.est = TRUE} and the fitted object is 
#'                 \code{var.ready = TRUE}. For each test subject, a matrix of size NFail\eqn{\times}NFail
#'                 where NFail is the number of observed failure times in the training data}
#' \item{Var}{if \code{var.est = TRUE} and the fitted object is 
#'                 \code{var.ready = TRUE}. Marginal variance for each subject}
#'  \item{timepoints}{ordered observed failure times from the training data}               
#' \item{MarginalVar}{if \code{var.est = TRUE} and the fitted object is 
#'                 \code{var.ready = TRUE}. Marginal variance for each subject
#'                 from the Cov matrix projected to the nearest positive definite
#'                 matrix}
#' \item{MarginalVarSmooth}{if \code{var.est = TRUE} and the fitted object is 
#'                 \code{var.ready = TRUE}. Marginal variance for each subject
#'                 from the Cov matrix projected to the nearest positive definite
#'                 matrix and then smoothed using Gaussian kernel smoothing}
#' \item{CVproj}{if \code{var.est = TRUE} and the fitted object is 
#'                 \code{var.ready = TRUE}. Critical values to calculate confidence bands around
#'                 cumulative hazard predictions at several confidence levels. Calculated using 
#'                 \code{MarginalVar}}
#' \item{CVprojSmooth}{if \code{var.est = TRUE} and the fitted object is 
#'                 \code{var.ready = TRUE}. Critical values to calculate confidence bands around
#'                 cumulative hazard predictions at several confidence levels. Calculated using 
#'                 \code{MarginalVarSmooth}}
#'                 
#'                 
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
  
  # insample prediction
  if (is.null(testx))
    return(object$OOBPrediction)
  
  # check test data
  if (!is.matrix(testx) & !is.data.frame(testx)) stop("testx must be a matrix or a data.frame")
  
  if (ncol(testx) < object$parameters$p) stop("test data dimension does not match training data") 
    
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
  
  ncomb = object$parameters$linear.comb
  
  if( class(object)[2] == "fit" &  class(object)[3] == "reg" & ncomb == 1)
  {
    pred <- RegUniForestPred(object$FittedForest$SplitVar,
                             object$FittedForest$SplitValue,
                             object$FittedForest$LeftNode,
                             object$FittedForest$RightNode,
                             object$FittedForest$NodeWeight,
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
  
  
  if( class(object)[2] == "fit" & class(object)[3] == "reg" & ncomb > 1)
  {
    pred <- RegUniCombForestPred(object$FittedForest$SplitVar,
                                 object$FittedForest$SplitLoad,
                                 object$FittedForest$SplitValue,
                                 object$FittedForest$LeftNode,
                                 object$FittedForest$RightNode,
                                 object$FittedForest$NodeWeight,
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
  
  
  if( class(object)[2] == "fit" &  class(object)[3] == "cla" )
  {
    pred <- ClaUniForestPred(object$FittedForest$SplitVar,
                             object$FittedForest$SplitValue,
                             object$FittedForest$LeftNode,
                             object$FittedForest$RightNode,
                             object$FittedForest$NodeWeight,
                             object$FittedForest$NodeProb,
                             testx,
                             object$ncat,
                             var.est,
                             keep.all,
                             ncores,
                             verbose)
    
    pred$Prediction = as.factor( c(1:object$nclass, pred$Prediction+1) )[-(1:object$nclass)]
    levels(pred$Prediction) = object$ylabels

    class(pred) <- c("RLT", "pred", "cla")
    return(pred)
  }  
  
  if( class(object)[2] == "fit" &  class(object)[3] == "surv" )
  {
    pred <- SurvUniForestPred(object$FittedForest$SplitVar,
                              object$FittedForest$SplitValue,
                              object$FittedForest$LeftNode,
                              object$FittedForest$RightNode,
                              object$FittedForest$NodeWeight,
                              object$FittedForest$NodeHaz,
                              testx,
                              object$ncat,
                              length(object$timepoints),
                              var.est,
                              keep.all,
                              ncores,
                              verbose)
    
    pred$timepoints <- object$timepoints

    class(pred) <- c("RLT", "pred", "surv")
    return(pred)
  }
}
