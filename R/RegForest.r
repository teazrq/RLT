#' @title RegForest
#' @name RegForest
#' @description Internal function for fitting regression forest
#' @keywords internal

RegForest <- function(x, y,
                      ncat,
                      param,
                      RLT.control,
                      obs.w,
                      var.w,
                      ncores,
                      verbose,
                      ObsTrack,
                      ...)
{
  # prepare y
  storage.mode(y) <- "double"

  # check splitting rule 
  all.split.rule = c("var")
    
  param$"split.rule" <- match.arg(param$"split.rule", all.split.rule)
  param$"split.rule" <- match(param$"split.rule", all.split.rule)
  
  # fit model
  fit = RegForestUniFit(x, y, ncat,
                        param, RLT.control,
                        obs.w, var.w,
                        ncores, verbose,
                        ObsTrack)

  fit[["parameters"]] = param
  fit[["RLT.control"]] = RLT.control
  fit[["ncat"]] = ncat  
  fit[["obs.w"]] = obs.w
  fit[["var.w"]] = var.w
  fit[["y"]] = y
  
  class(fit) <- c("RLT", "fit", "reg")
  return(fit)
}
