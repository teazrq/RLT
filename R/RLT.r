#' @title Main function of reinforcement learning trees
#' @description Fit models for regression, classification and survival analysis using reinforced splitting rules
#' @param x A matrix or data.frame for features
#' @param y Response variable, a numeric/factor vector or a Surv object
#' @param censor The censoring indicator if survival model is used
#' @param model The model type: \code{regression}, \code{classification} or \code{survival}
#' @param print.summary Whether summary should be printed
#' @param use.cores Number of cores
#' @param ntrees Number of trees, \code{ntrees = 100} if use reinforcement, \code{ntrees = 1000} otherwise
#' @param mtry Number of variables used at each internal node, only for \code{reinforcement = FALSE}
#' @param nmin Minimum number of observations reqired in an internal node to perform a split. Set this to twice of the desired terminal node size.
#' @param alpha Minimum number of observations required for each child node as a portion of the parent node. Must be within \code{(0, 0.5]}.
#' @param split.gen How the cutting points are generated
#' @param nsplit Number of random cutting points to compare for each variable at an internal node
#' @param resample.prob Proportion of in-bag samples
#' @param replacement Whether the in-bag samples are sampled with replacement
#' @param npermute Number of imputations (currently not implemented, saved for future use)
#' @param select.method Method to compare different splits
#' @param subject.weight Subject weights
#' @param variable.weight Variable weights when randomly sample \code{mtry} to select the splitting rule
#' @param track.obs Track which terminal node the observation belongs to
#' @param importance Should importance measures be calculated
#' @param reinforcement If reinforcement splitting rules should be used. There are default values for all tuning parameters under this feature.
#' @param muting Muting method, \code{-1} for muting by proportion, positive for muting by count
#' @param muting.percent Only for \code{muting = -1} the proportion of muting
#' @param protect Number of protected variables that will not be muted. These variables are adaptived selected for each tree.
#' @param combsplit Number of variables used in a combination split. \code{combsplit = 1} gives regular binary split; \code{combsplit > 1} produces linear combination splits.
#' @param combsplit.th The mininum threshold (as a relative measurement compared to the best variable) for a variable to be used in the combination split.
#' @param random.select Randomly select a varaible from the top variable in the linear combination as the splitting rule.
#' @param embed.n.th Number of observations to stop the embedded model and choose randomly from the current protected variables.
#' @param embed.ntrees Number of embedded trees
#' @param embed.resample.prob Proportion of in-bag samples for embedded trees
#' @param embed.mtry Number of variables used for embedded trees, as proportion
#' @param embed.nmin Terminal node size for embedded trees
#' @param embed.split.gen How the cutting points are generated in the embedded trees
#' @param embed.nsplit Number of random cutting points for embedded trees
#' @return A \code{RLT} object; a list consisting of
#' \item{FittedTrees}{Fitted tree structure}
#' \item{FittedSurv, timepoints}{Terminal node survival estimation and all time points, if survival model is used}
#' \item{AllError}{All out-of-bag errors, if \code{importance = TRUE}}
#' \item{VarImp}{Variable importance measures, if \code{importance = TRUE}}
#' \item{ObsTrack}{Registration of each observation in each fitted tree}
#' \item{...}{All the tuning parameters are saved in the fitted \code{RLT} object}
#' @references Zhu, R., Zeng, D., & Kosorok, M. R. (2015) "Reinforcement Learning Trees." Journal of the American Statistical Association. 110(512), 1770-1784.
#' @references Zhu, R., & Kosorok, M. R. (2012). Recursively imputed survival trees. Journal of the American Statistical Association, 107(497), 331-340.
#' @examples
#'
#' N = 600
#' P = 100
#'
#' X = matrix(runif(N*P), N, P)
#' Y = rowSums(X[,1:5]) + rnorm(N)
#'
#' trainx = X[1:200,]
#' trainy = Y[1:200]
#' testx = X[-c(1:200),]
#' testy = Y[-c(1:200)]
#'
#' # Regular ensemble trees (Extremely Randomized Trees, Geurts, et. al., 2006)
#'
#' RLT.fit = RLT(trainx, trainy, model = "regression", use.cores = 6)
#'
#' barplot(RLT.fit$VarImp)
#' RLT.pred = predict(RLT.fit, testx)
#' mean((RLT.pred$Prediction - testy)^2)
#'
#' # Reinforcement Learning Trees, using an embedded model to find the splitting rule
#' \dontrun{
#' Mark0 = proc.time()
#' RLT.fit = RLT(trainx, trainy, model = "regression", use.cores = 6, ntrees = 100,
#'               importance = TRUE, reinforcement = TRUE, combsplit = 3, embed.ntrees = 25)
#' proc.time() - Mark0
#'
#' barplot(RLT.fit$VarImp)
#' RLT.pred = predict(RLT.fit, testx)
#' mean((RLT.pred$Prediction - testy)^2)
#' }

RLT <- function(x, y, censor = NULL, model = "regression",
				print.summary = 0,
				use.cores = 1,
				ntrees = if (reinforcement) 100 else 500,
				mtry = max(1, as.integer(ncol(x)/3)),
				nmin = max(1, as.integer(log(nrow(x)))),
				alpha = 0.4,
				split.gen = "random",
				nsplit = 1,
				resample.prob = 0.9,
				replacement = TRUE,
				npermute = 1,
				select.method = "var",
				subject.weight = NULL,
				variable.weight = NULL,
				track.obs = FALSE,
				importance = TRUE,
				reinforcement = FALSE,
				muting = -1,
				muting.percent = if (reinforcement) MuteRate(nrow(x), ncol(x), speed = "aggressive", info = FALSE) else 0,
				protect = as.integer(log(ncol(x))),
				combsplit = 1,
				combsplit.th = 0.25,
				random.select = 0,
				embed.n.th = 4*nmin,
				embed.ntrees = max(1, -atan(0.01*(ncol(x) - 500))/pi*100 + 50),
				embed.resample.prob = 0.8,
				embed.mtry = 1/2,
				embed.nmin = as.integer(nrow(x)^(1/3)),
				embed.split.gen = "random",
				embed.nsplit = 1)
{
	# check inputs

 	if (missing(x)) stop("x is missing")
	if (missing(y)) stop("y is missing")

	allsplitting.generator = c("random", "uniform", "rank", "best")

	match.arg(model, c("regression", "classification", "survival"))
	match.arg(split.gen, allsplitting.generator)
	match.arg(embed.split.gen, allsplitting.generator)

	allselect.matric = c("var", "gini", "logrank", "suplogrank")
	match.arg(select.method, allselect.matric)

	xnames <- colnames(x)
	p = ncol(x)
	n = nrow(x)

    if (any(is.na(x))) stop("NA not permitted in x")
    if (any(is.na(y))) stop("NA not permitted in y")

	# prepare x, continuous and categorical
	if (is.data.frame(x))
	{
		if (nrow(x) != length(y)) stop("number of observations does not match")

	    xlevels <- lapply(x, function(x) if (is.factor(x)) levels(x) else 0)
        ncat <- sapply(xlevels, length)
		x <- data.matrix(x)
	} else {
        ncat <- rep(1, p)
        xlevels <- as.list(rep(0, p))
    }

	storage.mode(ncat) <- "integer"
	storage.mode(x) <- "double"

    if (max(ncat) > 53)
        stop("Cannot handle categorical predictors with more than 53 categories")

	# prepare y
	if (model == "regression")
	{
		if (nrow(x) != length(y)) stop("number of observations does not match")

		if (!(select.method %in% c("var")))
			select.method = "var" # variance reduction

		if (is.factor(y)) stop("y should be continuous for regression model")
		y <- data.matrix(y)
		storage.mode(y) <- "double"

		nclass = 0	# for classification only
		censor = 0	# for censored data only
	}

	if (model == "classification")
	{
		if (!(select.method %in% c("gini")))
			select.method = "gini" # gini impurity

		if (!is.factor(y)) stop("y should be factor for classification model")

		ylevels = levels(y)
		nclass = length(ylevels)
		if (nclass <= 1) stop("y is identical")

		y <- as.integer(y)-1L
		storage.mode(y) <- "integer"

		censor = 0	# for censored data only
	}

	if (model == "survival")
	{
		if (nrow(x) != length(y)) stop("number of observations does not match")
		if (nrow(x) != length(censor)) stop("number of observations does not match")

		nclass = 0	# for classification only

		if (is.null(censor)) stop("must specify censoring indicator")
		if (any(is.na(censor))) stop("NA not permitted in x")

		if (!(select.method %in% c("logrank", "suplogrank")))
			select.method = "logrank" # logrank

		if (is.factor(y)) stop("y should be continuous for regression model")
		if (any(y <= 0)) stop("y should be positive")

		timepoints = sort(unique(y[censor == 1]))
		y.point = rep(NA, length(y))
		for (i in 1:length(y))
		{
			if (censor[i] == 1)
				y.point[i] = match(y[i], timepoints)
			else
				y.point[i] = sum(y[i] >= timepoints)
		}

		y.point <- data.matrix(y.point)
		storage.mode(y.point) <- "integer"

		censor <- as.integer(censor)
		storage.mode(censor) <- "integer"

		interval = timepoints - c(0, timepoints[-length(timepoints)])
		interval <- data.matrix(c(0, interval))
		storage.mode(interval) <- "double"
	}

	# set up and check parameters

	if (!reinforcement)
		combsplit = 1

	# if (reinforcement) stop("reinforcement mode is not implemented yet... ")

	nmin = max(2, floor(nmin)) # need at least 2 observations to split
	embed.nmin = max(2, floor(embed.nmin))

	resample.prob = max(0, min(resample.prob, 1))
	if (!importance & resample.prob*n < nmin) warning("Re-sampling probability too small, cannot afford a split...")

	alpha = max(0, min(alpha, 0.5))

	split.gen.C = match(split.gen, allsplitting.generator)
	embed.split.gen.C = match(embed.split.gen, allsplitting.generator)
	select.method.C = match(select.method, allselect.matric)

	use.sub.weight = !is.null(subject.weight)	# subject weights for calculating the score of each split
	if (is.null(subject.weight)) subject.weight = rep(1/n, n) else subject.weight = subject.weight/sum(subject.weight)
	if (length(subject.weight) != n) {warning("Subject weights length must be n, reset to equal weights"); subject.weight = rep(1/n, n); use.sub.weight = FALSE}
	if (any(subject.weight<=0)) {warning("Subject weights cannot be 0 or negative, reset to equal weights"); subject.weight = rep(1/n, n); use.sub.weight = FALSE}
	storage.mode(subject.weight) <- "double"

	use.var.weight = !is.null(variable.weight)	# variable weights for mtry
	if (is.null(variable.weight)) variable.weight = rep(1/p, p) else variable.weight = variable.weight/sum(variable.weight)
	if (length(variable.weight) != p) {warning("Variable weights length must by p, reset to equal weights"); subject.weight = rep(1/p, p); use.var.weight = FALSE}
	if (any(variable.weight<0)) {warning("Variable weights cannot be negative, reset to equal weights"); subject.weight = rep(1/p, p); use.var.weight = FALSE}
	storage.mode(variable.weight) <- "double"

	parameters.int = c(print.summary, use.cores, ntrees, mtry, nmin,
	                   split.gen.C, nsplit, select.method.C, nclass, replacement,
	                   npermute, reinforcement, muting, protect, combsplit,
	                   embed.ntrees, embed.nmin, embed.split.gen.C, embed.nsplit, embed.n.th,
	                   importance, use.sub.weight, use.var.weight, track.obs, random.select)

	storage.mode(parameters.int) <- "integer"

	parameters.double = c(resample.prob, muting.percent, combsplit.th, embed.resample.prob, embed.mtry, alpha)
	storage.mode(parameters.double) <- "double"

	# fit model

	if (model == "regression")
	{
		RLT.fit = .Call(RLT_regression,
						datasetX = x,
						datasetY = y,
						ncat = ncat,
						subject.weight = subject.weight,
						variable.weight = variable.weight,
						parameters.int = parameters.int,
						parameters.double = parameters.double)
	}

	if (model == "classification")
	{
		RLT.fit = .Call(RLT_classification,
						datasetX = x,
						datasetY = y,
						ncat = ncat,
						subject.weight = subject.weight,
						variable.weight = variable.weight,
						parameters.int = parameters.int,
						parameters.double = parameters.double)

		RLT.fit$ylevels = ylevels
		RLT.fit$nclass = nclass
	}

	if (model == "survival")
	{
		RLT.fit = .Call(RLT_survival,
						datasetX = x,
						datasetY = y.point,
						datasetCensor = censor,
						datasetInterval = interval,
						ncat = ncat,
						subject.weight = subject.weight,
						variable.weight = variable.weight,
						parameters.int = parameters.int,
						parameters.double = parameters.double)

		RLT.fit$timepoints = timepoints
	}


	if (importance)
	{
		if (is.null(colnames(x))) colnames(x) = paste("x", c(1:ncol(x)), sep = "")
		colnames(RLT.fit$AllError) = c(colnames(x), "nonPermute")
		rownames(RLT.fit$AllError) = paste("Tree", 1:ntrees, sep = "")
		colnames(RLT.fit$VarImp) = colnames(x)
	}

	if (track.obs)
	{
		rownames(RLT.fit$ObsTrack) = rownames(x)
		colnames(RLT.fit$ObsTrack) = paste("Tree", 1:ntrees, sep = "")
	}

	RLT.fit$model = model
	RLT.fit$n = n
	RLT.fit$p = p
	RLT.fit$variablenames = colnames(x)
	RLT.fit$subjectnames = rownames(x)
	RLT.fit$ncat = ncat
    RLT.fit$xlevels = xlevels
	RLT.fit$ntrees = ntrees
	RLT.fit$mtry = mtry
	RLT.fit$nmin = nmin
	RLT.fit$split.gen = split.gen
	RLT.fit$nsplit = nsplit
	RLT.fit$select.method = select.method
	RLT.fit$replacement = replacement
	RLT.fit$resample.prob = resample.prob
	RLT.fit$npermute = npermute
	RLT.fit$subject.weight = subject.weight
	RLT.fit$track.obs = track.obs
	RLT.fit$importance = importance
	RLT.fit$use.sub.weight = use.sub.weight
	RLT.fit$subject.weight = subject.weight
	RLT.fit$reinforcement = reinforcement
	RLT.fit$combsplit = combsplit

	if (reinforcement)
	{
		RLT.fit$muting = muting
		RLT.fit$muting.percent = muting.percent
		RLT.fit$protect = protect
		RLT.fit$combsplit.th = combsplit.th
		RLT.fit$embed.ntrees = embed.ntrees
		RLT.fit$embed.resample.prob = embed.resample.prob
		RLT.fit$embed.mtry = embed.mtry
		RLT.fit$embed.split.gen = embed.split.gen
		RLT.fit$embed.nsplit = embed.nsplit
		RLT.fit$embed.n.th = embed.n.th
	}

	RLT.fit$parameters.int = parameters.int
	RLT.fit$parameters.double = parameters.double

	class(RLT.fit) <- "RLT"
	return(RLT.fit)
}
