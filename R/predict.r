clean.variance <- function(pred) {
  if (!is.null(pred$Variance)) {
    pred$Variance[pred$Variance < 0] <- NA
  }
  pred
}

#' @title Predict with an RLT forest
#' @description  Predict the outcome (regression, classification or survival) 
#'              using a fitted RLT object
#' @param object          A fitted RLT object
#' @param testx           New observations with the training feature structure.
#'                        If omitted, returns the stored OOB \code{object$Prediction}
#'                        directly. For supplied data, labels are
#'                        matched using the saved training dictionaries; numerical values for
#'                        encoded features mean integer training codes. Legacy fits require
#'                        their original dictionaries or refitting before accepting labels.
#' @param var.est         Whether to estimate prediction uncertainty. Fit with an
#'                        active \code{var.mode}, or satisfy the requirements for
#'                        an explicit prediction-time mode described below.
#'                        Regression returns prediction variance; classification
#'                        returns variance for each class probability. Survival returns
#'                        cumulative-hazard covariance over the evaluation times.
#' @param keep.all        whether to keep the prediction from all trees. Warning: this can 
#'                        occupy a large storage space, especially in survival model
#' @param ncores          number of cores
#' @param verbose         print additional information
#' @param var.mode        Variance estimation mode for prediction. Can be \code{"none"},
#'                        \code{"matched"}, \code{"IJ"}, or \code{"jack"}. If \code{NULL}
#'                        (default), uses the mode from the fitted object. Only used when
#'                        \code{var.est = TRUE}. \code{"matched"} requires the forest to be
#'                        fitted with \code{var.mode = "matched"}. \code{"IJ"} and \code{"jack"}
#'                        require the forest to have been fitted with resample tracking
#'                        (automatically enabled when using IJ or jack variance).
#' @param band.grid.size  A nonnegative integer specifying the requested maximum number of evaluation time points.
#'                        Default is 0, which uses all unique failure time points. If a positive integer
#'                        is provided, a subset of time points will be selected using quantiles
#'                        from the fifth percentile through the last time point. Quantile selection
#'                        may return fewer distinct points. A request of 1 selects the fifth-percentile point.
#' @param ... ...
#'
#' @return 
#' 
#' With \code{testx = NULL}, returns \code{object$Prediction} directly,
#' without a prediction list. Otherwise returns an \code{RLT} prediction list.
#'
#' For regression and classification:
#' \describe{
#' \item{Prediction}{Regression: an \eqn{N \times 1} numeric matrix.
#' Classification: a factor of predicted labels using the training response levels.
#' Here \code{N = nrow(testx)}.}
#' \item{Prob}{Classification only: an \eqn{N \times K} matrix of class
#' probabilities, with columns in the order of the fitted object's \code{ylabels},
#' where \code{K} is the number of classes.}
#' \item{Variance}{When variance estimation is requested, regression prediction
#' variances or an \eqn{N \times K} matrix of class-probability variances.
#' Negative variance estimates are returned as \code{NA}.}
#' }
#'                 
#' \strong{For Survival Forests}
#'
#' \code{Hazard}, \code{CHF} and \code{Survival} each have one row per test
#' observation and one column per evaluation time. There is no \code{Prediction}
#' component in this list.
#' \describe{
#' \item{Hazard}{predicted pointwise hazard increments at the selected times}
#' \item{CHF}{predicted cumulative hazard, accumulated on the complete training grid before
#'              selecting evaluation times. On a reduced grid, accumulating only \code{Hazard}
#'              does not in general reproduce \code{CHF}.}
#' \item{Survival}{ordinary forests return \code{exp(-CHF)}. Linear combination forests
#'                 return the mean of the tree product-limit survival curves. The
#'                 \code{get.surv.band()} interface uses \code{exp(-CHF)} as its reference curve.}
#' \item{AllHazard}{if \code{keep.all = TRUE}, the pointwise hazard increments for each
#'                 tree, selected time, and observation, in that array order}
#' \item{Cov}{when variance estimation is requested, an array with one
#'             \eqn{T\times T} cumulative-hazard covariance matrix per test subject,
#'             where \code{T = length(timepoints)}. Its dimensions are time,
#'             time, and test observation, in that order.}
#' \item{timepoints}{ordered evaluation times selected from the training failure grid}
#' }
#' @export
#' @examples
#' \donttest{
#'   set.seed(42)
#'   x <- matrix(rnorm(300 * 5), ncol = 5)
#'   y <- rowSums(x[, 1:2]) + rnorm(300)
#'   fit <- RLT(x, y, ntrees = 100)
#'   pred <- predict(fit, testx = x[1:5, ])
#'   print(pred$Prediction)
#' }
predict.RLT<- function(object,
                       testx = NULL,
                       var.est = FALSE,
                       var.mode = NULL,
                       keep.all = FALSE, 
                       ncores = 1,
                       verbose = 0,
                       band.grid.size = 0, 
                       ...)
{
  if (!is.numeric(band.grid.size) || is.complex(band.grid.size) ||
      length(band.grid.size) != 1L || !is.finite(band.grid.size) ||
      band.grid.size < 0 || band.grid.size != floor(band.grid.size))
    stop("band.grid.size must be a finite nonnegative integer scalar")
  
  # insample prediction
  if (is.null(testx))
    return(object$Prediction)
  
  testx <- encode_RLT_features(object, testx, "testx")

  if (var.est && is.null(var.mode) && object$parameters$var.mode == 0)
    stop("The original forest is not fitted with `var.mode`. Please check the conditions and build another forest, or specify var.mode explicitly.")
  
  # Validate that requested var.mode is compatible with fitted forest
  if (var.est && !is.null(var.mode)) {
    req_mode <- match.arg(var.mode, c("none", "matched", "IJ", "jack"))
    fitted_mode <- c("none", "matched", "IJ", "jack")[object$parameters$var.mode + 1]
    
    # matched requires the forest to be built with matched sampling (half-and-half tree structure)
    if (req_mode == "matched" && object$parameters$var.mode != 1)
      stop("matched variance requires the forest to be fitted with var.mode = 'matched'")
    
    # IJ/jack require ObsTrack (resample tracking)
    if (req_mode %in% c("IJ", "jack") && is.null(object$ObsTrack))
      stop("IJ/jack variance requires resample tracking. Fit with var.mode = 'IJ' or 'jack', or resample.track = 1")
  }
  
  ncomb = object$parameters$linear.comb
  
  if( class(object)[[2]] == "fit" && class(object)[[3]] == "reg" && ncomb == 1)
  {
    vm <- if (!is.null(var.mode)) {
      match.arg(var.mode, c("none", "matched", "IJ", "jack"))
      c("none"=0L, "matched"=1L, "IJ"=2L, "jack"=3L)[var.mode]
    } else {
      object$parameters$var.mode
    }
    cxx_var_mode <- if (var.est) vm else 0L
    
    # Get ObsTrack, or empty matrix if not available
    ObsTrack <- object$ObsTrack
    if (is.null(ObsTrack)) ObsTrack <- matrix(0L, nrow = 0, ncol = 0)
    
    pred <- RegUniForestPred(object$FittedForest$SplitVar,
                             object$FittedForest$SplitValue,
                             object$FittedForest$LeftNode,
                             object$FittedForest$RightNode,
                             object$FittedForest$NodeWeight,
                             object$FittedForest$NodeAve,
                             testx,
                             object$ncat,
                             ObsTrack,
                             cxx_var_mode,
                             keep.all,
                             ncores,
                             verbose)
    
    class(pred) <- c("RLT", "pred", "reg")
    pred <- clean.variance(pred)
    return(pred)
  }
  
  
  if( class(object)[[2]] == "fit" && class(object)[[3]] == "reg" && ncomb > 1)
  {
    vm <- if (!is.null(var.mode)) {
      match.arg(var.mode, c("none", "matched", "IJ", "jack"))
      c("none"=0L, "matched"=1L, "IJ"=2L, "jack"=3L)[var.mode]
    } else {
      object$parameters$var.mode
    }
    cxx_var_mode <- if (var.est) vm else 0L
    
    # Get ObsTrack, or empty matrix if not available
    ObsTrack <- object$ObsTrack
    if (is.null(ObsTrack)) ObsTrack <- matrix(0L, nrow = 0, ncol = 0)
    
    pred <- RegUniCombForestPred(object$FittedForest$SplitVar,
                                 object$FittedForest$SplitLoad,
                                 object$FittedForest$SplitValue,
                                 object$FittedForest$LeftNode,
                                 object$FittedForest$RightNode,
                                 object$FittedForest$NodeWeight,
                                 object$FittedForest$NodeAve,
                                 testx,
                                 object$ncat,
                                 ObsTrack,
                                 cxx_var_mode,
                                 keep.all,
                                 ncores,
                                 verbose)
    
    class(pred) <- c("RLT", "pred", "reg")
    pred <- clean.variance(pred)
    return(pred)
  }
  
  
  if( class(object)[[2]] == "fit" && class(object)[[3]] == "cla" && ncomb == 1)
  {
    vm <- if (!is.null(var.mode)) {
      match.arg(var.mode, c("none", "matched", "IJ", "jack"))
      c("none"=0L, "matched"=1L, "IJ"=2L, "jack"=3L)[var.mode]
    } else {
      object$parameters$var.mode
    }
    cxx_var_mode <- if (var.est) vm else 0L
    
    # Get ObsTrack, or empty matrix if not available
    ObsTrack <- object$ObsTrack
    if (is.null(ObsTrack)) ObsTrack <- matrix(0L, nrow = 0, ncol = 0)
    
    pred <- ClaUniForestPred(object$FittedForest$SplitVar,
                             object$FittedForest$SplitValue,
                             object$FittedForest$LeftNode,
                             object$FittedForest$RightNode,
                             object$FittedForest$NodeWeight,
                             object$FittedForest$NodeProb,
                             testx,
                             object$ncat,
                             ObsTrack,
                             cxx_var_mode,
                             keep.all,
                             ncores,
                             verbose)
    
    pred$Prediction = as.factor( c(1:object$nclass, pred$Prediction+1) )[-(1:object$nclass)]
    levels(pred$Prediction) = object$ylabels

    class(pred) <- c("RLT", "pred", "cla")
    pred <- clean.variance(pred)
    return(pred)
  }  
  
  if( class(object)[[2]] == "fit" && class(object)[[3]] == "cla" && ncomb > 1)
  {
    vm <- if (!is.null(var.mode)) {
      match.arg(var.mode, c("none", "matched", "IJ", "jack"))
      c("none"=0L, "matched"=1L, "IJ"=2L, "jack"=3L)[var.mode]
    } else {
      object$parameters$var.mode
    }
    cxx_var_mode <- if (var.est) vm else 0L
    
    # Get ObsTrack, or empty matrix if not available
    ObsTrack <- object$ObsTrack
    if (is.null(ObsTrack)) ObsTrack <- matrix(0L, nrow = 0, ncol = 0)
    
    pred <- ClaUniCombForestPred(object$FittedForest$SplitVar,
                                 object$FittedForest$SplitLoad,
                                 object$FittedForest$SplitValue,
                                 object$FittedForest$LeftNode,
                                 object$FittedForest$RightNode,
                                 object$FittedForest$NodeWeight,
                                 object$FittedForest$NodeProb,
                                 testx,
                                 object$ncat,
                                 ObsTrack,
                                 cxx_var_mode,
                                 keep.all,
                                 ncores,
                                 verbose)
    
    pred$Prediction = as.factor( c(1:object$nclass, pred$Prediction+1) )[-(1:object$nclass)]
    levels(pred$Prediction) = object$ylabels

    class(pred) <- c("RLT", "pred", "cla")
    pred <- clean.variance(pred)
    return(pred)
  }  
  
  if( class(object)[[2]] == "fit" && class(object)[[3]] == "surv" )
  {
    # Resolve variance mode
    vm <- if (!is.null(var.mode)) {
      match.arg(var.mode, c("none", "matched", "IJ", "jack"))
      c("none"=0L, "matched"=1L, "IJ"=2L, "jack"=3L)[var.mode]
    } else {
      object$parameters$var.mode
    }
    cxx_var_mode <- if (var.est) vm else 0L
    
    # Get ObsTrack, or empty matrix if not available
    ObsTrack <- object$ObsTrack
    if (is.null(ObsTrack)) ObsTrack <- matrix(0L, nrow = 0, ncol = 0)
    
    # Get original timepoints and NFail
    original_timepoints <- object$timepoints
    NFail <- length(object$timepoints)
    
    # Default: use all points
    new_grid_timepoints <- original_timepoints

    # Only perform reduction when user specifies valid band.grid.size
    if (band.grid.size > 0) {
      effective_grid_size <- min(band.grid.size, NFail)
      
      # Only execute selection when reduction is actually needed
      if (effective_grid_size < NFail){
        probs_to_select <- seq(from = 0.05, to = 1.0, length.out = effective_grid_size)
        new_grid_timepoints <- unique(as.numeric(stats::quantile(original_timepoints, probs = probs_to_select, type = 1)))
      }
    }
    
    new_grid_timepoints <- sort(new_grid_timepoints)
    
    # Create mapping indices for C++ function
    mapping_indices <- match(new_grid_timepoints, original_timepoints) - 1  # Convert to 0-based indexing
    
    if (ncomb > 1)
    {
      # Linear combination (Comb) survival forest prediction
      pred <- SurvUniCombForestPred(object$FittedForest$SplitVar,
                                     object$FittedForest$SplitLoad,
                                     object$FittedForest$SplitValue,
                                     object$FittedForest$LeftNode,
                                     object$FittedForest$RightNode,
                                     object$FittedForest$NodeWeight,
                                     object$FittedForest$NodeHaz,
                                     testx,
                                     object$ncat,
                                     NFail, 
                                     mapping_indices,
                                     ObsTrack,
                                     cxx_var_mode,
                                     keep.all,
                                     ncores,
                                     verbose)
    }else{
      # Single-variable survival forest prediction
      pred <- SurvUniForestPred(object$FittedForest$SplitVar,
                                object$FittedForest$SplitValue,
                                object$FittedForest$LeftNode,
                                object$FittedForest$RightNode,
                                object$FittedForest$NodeWeight,
                                object$FittedForest$NodeHaz,
                                testx,
                                object$ncat,
                                NFail, 
                                mapping_indices,
                                ObsTrack,
                                cxx_var_mode,
                                keep.all,
                                ncores,
                                verbose)
    }

    
    pred$timepoints <- new_grid_timepoints
    
    class(pred) <- c("RLT", "pred", "surv")
    pred <- clean.variance(pred)
    return(pred)
  }
}
