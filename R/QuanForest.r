#' @title QuanForest
#' @name QuanForest
#' @description Internal function for fitting quantile forest
#' @keywords internal

QuanForest <- function(x, y, ncat,
                       obs.w, var.w,
                       resample.preset,
                       param,
                       ...)
{
  # prepare y
  storage.mode(y) <- "double"
    
    if (param$verbose > 0)
      cat("Fitting Quantile Forest ... \n")
      
    # check splitting rules
    all.split.rule = c("default")
    param$"split.rule" <- match.arg(param$"split.rule", all.split.rule)
    param$"split.rule" <- as.integer(match(param$"split.rule", all.split.rule))
    
    # fit single variable split model
    fit = QuanUniForestFit(x, y, ncat,
                          obs.w, var.w,
                          resample.preset,
                          param)
    
    fit[["parameters"]] = param
    fit[["ncat"]] = ncat
    fit[["obs.w"]] = obs.w
    fit[["var.w"]] = var.w
    fit[["y"]] = y
    
    class(fit) <- c("RLT", "fit", "quan", "uni", "single")

    return(fit)
}
