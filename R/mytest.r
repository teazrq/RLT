#' @title mytest
#' @name mytest
#' @description my function
#' @param n n
#' @export
#' @return output

mytest <- function(n, ...)
{
    cat("Run test: \n")
    
	testcpp(n)
}
