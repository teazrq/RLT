#' @title Muting rate
#' @description Get the muting rate based on sample size \code{N} and dimension \code{P}. This is an experimental feature. When P is too small, this is not recommended.
#' @param N sample size
#' @param P dimension
#' @param speed Muting speed: moderate or aggressive
#' @param info Whether to output detailed information
#' @return A suggested muting rate
#' @examples
#' MuteRate(500, 100, speed = "aggressive")

MuteRate <- function(N, P, speed = NULL, info = FALSE)
{
	if (is.null(speed)) cat("please specify `speed`: moderate or aggressive")

	match.arg(speed, c("moderate", "aggressive"))

	if (N < 50 & info)
	{
		warning("N is very small for the embedded model.")
	}

	if (P < 10 & info)
	{
		warning("P is very small, do you really want to mute?")
	}

	if (speed == "moderate")
	{
		n0 = 25
		p0 = sqrt(P)
		rate = max(0, 1-(p0/P)^(log(2)/log(N/n0)))
		if (info) cat("suggested moderate muting rate", paste(round(rate*100,2), "%", sep=""), ": reach", round(sqrt(P), 0), "variables when node sample is 25. \n")
		return(rate)
	}

	if (speed == "aggressive")
	{
		n0 = 50
		p0 = log(P)
		rate = max(0, 1-(p0/P)^(log(2)/log(N/n0)))
		if (info) cat("suggested aggressive muting rate", paste(round(rate*100,2), "%", sep=""), ": reach", round(log(P), 0), "variables when node sample is 50. \n")
		return(rate)
	}
}
