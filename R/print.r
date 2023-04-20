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
  
  if (class(x)[3] == "quan")
      model = "quantile"   
  
  if(class(x)[2] == "fit")
    cat(paste("An RLT fitted", model, "forest \n"))
    
  if(class(x)[2] == "pred")
    cat(paste("An RLT", model, "prediction object \n"))

  if(class(x)[2] == "band")
    cat("An RLT survival confidence band object \n")
  
  if(class(x)[2] == "kernel")
    cat(paste("An RLT", class(x)[3], "kernel object.\n"))
}
