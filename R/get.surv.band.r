#' @title get.surv.band
#' @description Calculate the survival function (two-sided) confidence band from 
#'              a RLT survival prediction. 
#' @param x A RLT prediction object. This must be an object calculated from a forest 
#'          with \code{var.ready = TRUE}.
#' @param i Observation number in the prediction, default to calculate all
#' @param alpha alpha level for interval \eqn{(\alpha/2, 1 - \alpha/2)}
#' @param ... ...
#' @export

get.surv.band <- function(x, i = 0, alpha = 0.05, ...)
{
  if (any(class(x)[1:3] != c("RLT", "pred", "surv")))
    stop("Not an RLT survival prediction object.")
  
  if (is.null(x$CHFCov))
    stop("Not an RLT object fitted with var.ready")
  
  N = nrow(x$Hazard)
  Band = list()

  if (i == 0)
  {
    for (k in 1:N)
    {
      Band[[k]] = exp(-cbind(x$CHF[k, ] - sqrt(x$CHFMarVar[k, ])*qnorm(1-alpha/2), 
                             x$CHF[k, ] + sqrt(x$CHFMarVar[k, ])*qnorm(1-alpha/2)))      
        
      colnames(Band[[k]]) = c("Lower", "Upper")
    }
  }else{
    
    if (i < 0 | i > N)
      stop(paste("Observation", i, "does not exist"))
      
    Band[[1]] = as.matrix(exp(-cbind(x$CHF[i, ] - sqrt(x$CHFMarVar[i, ])*qnorm(1-alpha/2), 
                      x$CHF[i, ] + sqrt(x$CHFMarVar[i, ])*qnorm(1-alpha/2))))

    colnames(Band[[1]]) = c("Lower", "Upper")
  }
  
  Band[["timepoints"]] = x$timepoints
  
  class(Band) <- c("RLT", "band", "surv")
  
  return(Band)
}
