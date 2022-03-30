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
#' @param calc.cv  For survival forests only: calculate the critical value. IN DEVELOPMENT
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
                       calc.cv = FALSE,
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
    
    if(calc.cv){
      alpha_options = seq(1.5, 12, by=0.25)
      MarginalVar <- matrix(0, nrow = dim(testx)[1], 
                            ncol=length(object$timepoints))
      MarginalVarSmooth <- matrix(0, nrow = dim(testx)[1], 
                                  ncol=length(object$timepoints))
      CVproj <- numeric(dim(testx)[1])
      CVprojSmooth <- numeric(dim(testx)[1])
      
      for(n in 1:dim(testx)[1]){
        require(Matrix)
        pd_proj <- nearPD(as.matrix(pred$Cov[,,n]), maxit = 100,
                          ensureSymmetry = FALSE,
                          conv.norm.type = "F", trace = FALSE,
                          base.matrix = TRUE, corr = FALSE)
        MarginalVar[n,] <- diag(pd_proj$mat)
        
        require(MASS)
        norm_samps <- mvrnorm(10000, pred$CumHazard[n,], pd_proj$mat)
        cvg_list <- matrix(0, nrow = length(alpha_options), 
                           ncol = dim(norm_samps)[1])
        
        for(i in 1:length(alpha_options)){
          high <- alpha_options[i] * sqrt(MarginalVar[n,]) + pred$CumHazard[n,]
          low <- -alpha_options[i] * sqrt(MarginalVar[n,]) + pred$CumHazard[n,]
          for(k in 1:dim(norm_samps)[1]){#length(object$timepoints)
            cvg_list[i,k] <- mean(high >= pmax(norm_samps[k,],0) &
                                    low <=
                                    pmax(norm_samps[k,],0))
          }
        }
        full_coverage <- apply(cvg_list, 1, function(x)
          mean(x==1))  
        if(any(full_coverage>=0.95)){
          CVproj[n] <- alpha_options[min(which(full_coverage>=0.95))]
        }else{
          CVproj[n] <- 8.25
        }
        
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
        
        for(i in 1:length(alpha_options)){
          high <- alpha_options[i] * sqrt(MarginalVarSmooth[n,]) + pred$CumHazard[n,]
          low <- -alpha_options[i] * sqrt(MarginalVarSmooth[n,]) + pred$CumHazard[n,]
          for(k in 1:length(object$timepoints)){
            cvg_list[i,k] <- mean(high[k] >= pmax(norm_samps[,k],0) &
                                    low[k] <=
                                    pmax(norm_samps[,k],0))
          }
        }
        full_coverage <- apply(cvg_list, 1, function(x)
          mean(x==1))  
        if(any(full_coverage>=0.95)){
          CVprojSmooth[n] <- alpha_options[min(which(full_coverage>=0.95))]
        }else{
          CVprojSmooth[n] <- 8.25
        }
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
