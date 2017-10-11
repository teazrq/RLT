#' @title Prediction function for reinforcement learning trees
#' @description Predict future subjects with a fitted RLT model
#' @param object A fitted RLT object
#' @param testx Testing data
#' @param ... ...
#' @return The predicted values. For survival model, it returns the fitted survival functions
#' @examples
#' x = matrix(rnorm(100), ncol = 10)
#' y = rowMeans(x)
#' fit = RLT(x, y, ntrees = 5)
#' predict(fit, x)

predict.RLT <- function(object, testx, ...)
{
	# check test data

	if (missing(testx)) stop("testing data missing ...")

	if (is.null(colnames(testx)))
	{
		if (ncol(testx) != object$p) stop("test data dimension does not match training data, variable names are not supplied...")
	}else if (any(colnames(testx) != object$variablenames))
	{
		warning("test data variables names does not match training data...")
		varmatch = match(object$variablenames, colnames(testx))
		if (any(is.na(varmatch))) stop("test data missing some variables...")
		testx = testx[, varmatch]
	}

	# converting categorical data

	for (j in 1:object$p)
	{
		if (object$ncat[j] > 1)
		{
			if (!is.factor(testx[, j])) stop(paste("data format of test date is not correct: column", j, "should be a factor"))

			tempx = match(testx[, j], object$xlevels[[j]])

			if (any(is.na(tempx))) stop(paste("some categories of column", j, "in the test date were never presented in the training data"))

			testx[, j] = tempx
		}else
			if (is.factor(testx[, j])) stop(paste("data format of test date is not correct: column", j, "should be numerical"))
	}

	testx = data.matrix(testx)
	storage.mode(testx) <- "double"

	if (object$model == "regression")	RLT.predict = .Call(RLT_regression_predict,
															datasetX = testx,
															FittedTrees = object$FittedTrees,
															ncat = object$ncat,
															parameters.int = object$parameters.int,
															parameters.double = object$parameters.double)

	if (object$model == "classification")	RLT.predict = .Call(RLT_classification_predict,
																datasetX = testx,
																FittedTrees = object$FittedTrees,
																ncat = object$ncat,
																parameters.int = object$parameters.int,
																parameters.double = object$parameters.double)

	if (object$model == "survival"){	RLT.predict = .Call(RLT_survival_predict,
															datasetX = testx,
															FittedTrees = object$FittedTrees,
															FittedSurv = object$FittedSurv,
															ncat = object$ncat,
															parameters.int = object$parameters.int,
															parameters.double = object$parameters.double)

										RLT.predict$timepoints = object$timepoints
									}

	return(RLT.predict)
}

