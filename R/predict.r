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
    
    if(var.est){
      #alpha_options = seq(1.5, 12, by=0.25)
      cv_seq <- c(seq(0.5,0.95,.05),.99)
      MarginalVar <- matrix(0, nrow = dim(testx)[1], 
                            ncol=length(object$timepoints))
      MarginalVarSmooth <- matrix(0, nrow = dim(testx)[1], 
                                  ncol=length(object$timepoints))
      CVproj <- matrix(0, nrow=dim(testx)[1],ncol=length(cv_seq))
      CVprojSmooth <- matrix(0, nrow=dim(testx)[1],ncol=length(cv_seq))
      
      colnames(CVproj) <- colnames(CVprojSmooth) <- paste0("CV", cv_seq)
      
      for(n in 1:dim(testx)[1]){
        require(Matrix)
        pd_proj <- nearPD(as.matrix(pred$Cov[,,n]), maxit = 100,
                          ensureSymmetry = FALSE,
                          conv.norm.type = "F", trace = FALSE,
                          base.matrix = TRUE, corr = FALSE)
        MarginalVar[n,] <- diag(pd_proj$mat)
        
        require(MASS)
        norm_samps <- mvrnorm(10000, pred$CumHazard[n,], pd_proj$mat)

        norm_samps_pos <- apply(norm_samps, 1, function(r) ifelse(r>0,r,0))
        
        centered_mat <- apply(norm_samps_pos, 2, function(r) r-pred$CumHazard[n,])
        cent_scaled_mat <- apply(centered_mat, 2, function(co) co/sqrt(MarginalVar[n,]))
        cover_cv <- apply(cent_scaled_mat, 2, function(r) max(abs(r)))
        CVproj[n,] <- quantile(cover_cv, )
        
        b <- bw.nrd(c(1:length(object$timepoints)))
        MarginalVarSmooth[n,] <- ksmooth(x=c(1:length(object$timepoints)),
                                         y=MarginalVar[n,],
                                         "normal",
                                         bandwidth = b,
                                         x.points = c(1:length(object$timepoints))
        )$y
        diag(pd_proj$mat) <- MarginalVarSmooth[n,]
        pd_proj_sm <- nearPD(pd_proj$mat, maxit = 100,
                             ensureSymmetry = FALSE,
                             conv.norm.type = "F", trace = FALSE,
                             base.matrix = TRUE, corr = FALSE)
        norm_samps <- mvrnorm(1000, pred$CumHazard[n,], pd_proj_sm$mat)

        norm_samps_pos <- apply(norm_samps, 1, function(r) ifelse(r>0,r,0))
        
        centered_mat <- apply(norm_samps_pos, 2, function(r) r-pred$CumHazard[n,])
        cent_scaled_mat <- apply(centered_mat, 2, function(co) co/sqrt(MarginalVarSmooth[n,]))
        cover_cv <- apply(cent_scaled_mat, 2, function(r) max(abs(r)))
        CVprojSmooth[n,] <- quantile(cover_cv, c(seq(0.5,0.95,.05),.99))
      }
        
      pred$MarginalVar <- MarginalVar
      pred$MarginalVarSmooth <- MarginalVarSmooth
      pred$CVproj <- CVproj
      pred$CVprojSmooth <- CVprojSmooth
      
    }
    
    
    class(pred) <- c("RLT", "pred", "surv")
    return(pred)
  }
}
