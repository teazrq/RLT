#' @title get.surv.band
#' @description Calculate the survival function (two-sided) confidence band from 
#'              a RLT survival prediction. 
#' @param x A RLT prediction object. This must be an object calculated from a forest 
#'          with \code{var.ready = TRUE}.
#' @param i Observation number in the prediction. Default to calculate all (\eqn{i = 0})
#' @param alpha alpha level for interval \eqn{(\alpha/2, 1 - \alpha/2)}
#' @param method what method is used to calculate the confidence band. Can be
#'               \itemize{
#'               \item \code{naive-mc}: positive-definite projection of the covariance matrix.
#'               the confidence band is non-smooth
#'               \item \code{smoothed-mc}: use a smoothed marginal variance to perform the Monte Carlo 
#'               approximation of the critical value. This is only recommended for large 
#'               number of time points. 
#'               \item \code{smoothed-lr}: use a smoothed low-rank approximation of the covariance 
#'               matrix and apply an adaptive Bonferroni correction to derive the critical values.
#'               Note that this method relies on the assumption of the smoothness and low rank of the
#'               covariance matrix. 
#'               }
#' @param r    maximum number of ranks used in the \code{smoothed-lr} approximation. Usually 5 is 
#'             enough for approximating the covariance matrix due to smoothness. 
#' @param nsim number of simulations for estimating the Monte Carlo critical value. 
#'             Set this to be a large number. Default is 1000.          
#' @param ... ...
#' @export
get.surv.band <- function(x, 
                          i = 0, 
                          alpha = 0.05, 
                          method = "naive-mc",
                          nsim = 1000, 
                          r = 5,
                          ...)
{
  if (any(class(x)[1:3] != c("RLT", "pred", "surv")))
    stop("Not an RLT survival prediction object.")
  
  if (is.null(x$Cov))
    stop("Not an RLT object fitted with var.ready")
  
  all.methods = c("naive-mc", "smoothed-mc", "smoothed-lr")
  
  if(match(method, all.methods, nomatch = 0) == 0)
    stop("method not avaliable")
  
  
  N = dim(x$Cov)[3]
  p = dim(x$Cov)[2]
    
  # what subject to estimate
  
  if (i == 0)
  {
    allid = 1:N
  }else{
    if (i < 0 | i > N)
      stop(paste("Observation", i, "does not exist"))
    
    allid = i
  }
  
  if (any(alpha < 0) | any(alpha > 0.5))
    stop(paste("alpha not valid"))
  
  SurvBand = list()

  for (k in allid)
  {
    # naive method 
    if (method == "naive-mc")
    {
      marsd = sqrt(diag(x$Cov[,,k]))
      marsd = marsd / sqrt(sum(marsd^2))
      bandk = mc_band(marsd, x$Cov[,,k], alpha, nsim)
      approxerror = NULL
    }
      

    if (method == "smoothed-mc")
    {
      # will add two points at the boundary to improve the behavior. 
      alltime = 0:(p+1)
      alltime = alltime / sd(alltime) * 4
      nknots = ceiling( max(alltime)/orthoDr::silverman(1, p) ) + 2
      
      basis = orthoDr::kernel_weight(matrix(alltime), 
                                     matrix(seq(0, max(alltime), length.out = nknots)))
      
      # raw marginal variance
      mar_var = diag(x$Cov[,,k])

      fit <- glmnet::glmnet(basis, c(mar_var[1], mar_var, tail(mar_var, 1)),
                            alpha = 0, intercept = FALSE,
                            lower.limits = 0, lambda = 1e-5)
      smarvar = predict(fit, basis)
      smarvar = smarvar[2:(p+1)]
      
      # plot(1:p, mar_var)
      # lines(1:p, smarvar, type = "l", col = "red", lwd = 2)     

      smarsd = sqrt(smarvar)
      smarsd = smarsd / sqrt(sum(smarsd^2))
      
      bandk = mc_band(smarsd, x$Cov[,,k], alpha, nsim)
      approxerror = NULL
    }
      
    if (method == "smoothed-lr")
    {
      ccov = x$Cov[,,k]
      coveigen = eigen(ccov)
      
      r = min(r, sum(coveigen$values > coveigen$values[1]*1e-6))
      
      # some matrix to store
      U = matrix(NA, p, r)
      d = rep(NA, r)
      
      # generate grid points for kernel
      alltime = 1:p
      alltime = alltime / sd(alltime) * 4
      nknots = ceiling( max(alltime)/orthoDr::silverman(1, p) ) + 2
      
      basis = orthoDr::kernel_weight(matrix(alltime), 
                                     matrix(seq(0, max(alltime), length.out = nknots)))
      
      for (j in 1:r)
      {
        uj = coveigen$vectors[, j]
        uj = uj*sign(mean(uj))
        
        fit <- glmnet::glmnet(basis, uj, alpha = 0, intercept = FALSE,
                              lower.limits = 0, lambda = 1e-5)
        suj = predict(fit, basis)
        suj = suj / sqrt(sum(suj^2))
        
        dj = sum(suj * uj) * coveigen$values[j]
        
        U[, j] = suj
        d[j] = dj
      }
      
      used = (d > max(d)*1e-6)
      d = d[used]
      U = U[, used, drop = FALSE]
      
      w = sqrt(d)/sum(sqrt(d))
      
      cv = qnorm(1 - w %*% t(alpha)/2)
      bandk = U %*% sweep(cv, 1, sqrt(d), FUN = "*")

      approxerror = sum( (sweep(U, 2, d, FUN = "*") %*% t(U) - ccov)^2 ) / sum(ccov^2)
      
      #newcov = sweep(U, 2, d, FUN = "*") %*% t(U)
      #smarsd = sqrt(diag(newcov))
      #smarsd = smarsd / sqrt(sum(smarsd^2))
  
      #bandk = mc_band(smarsd, newcov, alpha, nsim)
      
      # max(abs(newcov - x$Cov[,,k]))
      # heatmap(x$Cov[,,k], Rowv = NA, Colv = NA, symm = TRUE)
      # heatmap(newcov, Rowv = NA, Colv = NA, symm = TRUE)
    }

    SurvBand[[k]] = list("lower" = x$Survival[k, ] - bandk,
                         "upper" = x$Survival[k, ] + bandk,
                         "approx.error" = approxerror)
    
    # SurvBand[[k]] = list("lower" = exp(-x$CHF[k, ] + bandk),
    #                      "upper" = exp(-x$CHF[k, ] - bandk),
    #                      "approx.error" = approxerror)
  }
  
  names(SurvBand) <- paste("Subject", allid, sep = "")
  
  SurvBand[["timepoints"]] = x$timepoints
  
  class(SurvBand) <- c("RLT", "band", "surv")
  
  return(SurvBand)
}
