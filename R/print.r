#' @title Print a RLT object
#' @description Print a RLT object
#' @param x A fitted RLT object
#' @param ... ...
#' @export

print.RLT<- function(x, ...)
{
  if(class(x)[2] == "fit")
    cat("\n This is an RLT fitted object \n")
  
  if(class(x)[2] == "pred")
    cat("\n This is an RLT prediction object \n")
  
  if(class(x)[2] == "kernel")
    cat("\n This is an RLT kernel object \n")
  
  
}
