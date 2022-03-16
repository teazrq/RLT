#' @title                 samples
#' @description           testing function
#' @export mysample
my_sample <- function(Num, 
                      min, 
                      max)
{
  seed = runif(1) * .Machine$integer.max
  return( mysample(Num, min, max, seed) )
}
