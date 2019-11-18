#' @title C-index
#' @name cindex
#' @description calculate c-index for survival data
#' @param y survival time
#' @param censor The censoring indicator if survival model is used
#' @param pred the predicted value for each subject
#' @export
#' @return c-index

cindex <- function(y, censor, pred, ...)
{
    if (length(y) != length(censor) | length(y) != length(pred))
        stop("arguments length differ")
    
    if ( any( ! (censor %in% c(0, 1)) ) )
        stop("censoring indicator must be 0 or 1")
        
	return( cindex_d(y, censor, pred) )
}
