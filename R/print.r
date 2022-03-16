#' @title Print a RLT object
#' @description Print a RLT object
#' @param x A fitted RLT object
#' @param ... ...
#' 
#' @export

print.RLT<- function(x, ...)
{
  if (class(x)[3] == "reg")
    model = "regression"
  
  if (class(x)[3] == "surv")
    model = "survival"  
  
  if (class(x)[3] == "cla")
    model = "classification"    
  
  if(class(x)[2] == "fit")
    cat(paste("A RLT fitted", model, "forest.\n"))
    
  if(class(x)[2] == "pred")
    cat("\n This is an RLT prediction object \n")

  if(class(x)[2] == "Var")
    cat(paste("A RLT", model, "variance estimation object.\n"))
}
