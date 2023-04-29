#' @title mytest
#' @name mytest
#' @description my function
#' @param n n
#' @param ... other arguments
#' @export
#' @return output

mytest <- function(n, ...)
{
    cat("Run test: \n")
    
	testcpp(n)
}
