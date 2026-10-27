#' @title Reinforcement Learning Trees
#' @description Fit random forest models for regression, classification and
#' survival analysis. By default, splits use one variable at a time.
#' Set \code{reinforcement = TRUE} to use embedded-model guidance for
#' single-variable splits, or \code{linear.comb > 1} for linear combination
#' splits. Linear combination splits do not require \code{reinforcement = TRUE};
#' regression and classification combination splits still use embedded screening
#' at sufficiently large nodes. See \code{param.control} for details.
#'
#' @param x               A numeric `matrix` or a `data.frame` of features. Unordered
#'                        factors in a data frame use categorical splits. Ordered factors
#'                        and character columns retain their numerical rank-code treatment.
#'                        Training label dictionaries are saved for all three types, so
#'                        prediction and kernels match labels to the training codes.
#'                        Numeric inputs for these columns are interpreted as integer
#'                        training codes in \code{1:length(training_labels)}, not as labels.
#'                        Character matrices are not supported.
#'
#' @param y               Response vector: numeric for regression or observed survival
#'                        times, and a factor for classification.
#'
#' @param censor          Survival event indicator: 1 for an observed event and 0
#'                        for a right-censored observation.
#'
#' @param model           The model type: `"regression"`, `"classification"`,
#'                        or `"survival"`. Quantile forest is not yet implemented.
#'
#' @param reinforcement   Whether to use embedded-model guidance for
#'                        single-variable splits. Default: \code{FALSE}.
#'                        Linear combination splits are selected separately by
#'                        \code{linear.comb}; see \code{param.control} for their
#'                        model-specific screening behavior.
#'
#' @param ntrees          Number of trees, `ntrees = 100` if reinforcement is
#'                        used and `ntrees = 500` otherwise.
#'
#' @param mtry            Number of randomly selected variables used at each
#'                        internal node. Default: max(1, floor(p/2)).
#'
#' @param nmin            Terminal node size. Splitting will stop when the internal
#'                        node size is less equal to \code{nmin}. Default: 5.
#'
#' @param alpha           Minimum proportion of samples (of the parent node)
#'                        enforced in each child node. Default: 0 (no constraint).
#'                        Clamped to the range 0 to 0.5.
#'
#' @param nsplit          Number of random cutting points to compare for each
#'                        variable at an internal node. Default: 0 (use all unique
#'                        values, i.e., best split). When nsplit > 0, random
#'                        cutting points are generated.
#'
#' @param resample.replace Whether the in-bag samples are obtained with
#'                        replacement.
#'
#' @param resample.prob   Proportion of in-bag samples.
#'
#' @param resample.preset A pre-specified matrix for in-bag data indicator/count
#'                        matrix. It must be an \eqn{n \times} \code{ntrees}
#'                        matrix with integer entries. Positive number indicates
#'                        the number of copies of that observation (row) in the
#'                        corresponding tree (column); zero indicates out-of-bag;
#'                        negative values indicates not being used in either.
#'                        Extremely large counts should be avoided. The sum of
#'                        each column should not exceed \eqn{n}.
#'
#' @param obs.w           Observation weights. The weights will be used for calculating
#'                        the splitting scores, such as a weighted variance reduction
#'                        or weighted gini index. But they will not be used for
#'                        sampling observations. In that case, one can pre-specify
#'                        \code{resample.preset} instead for balanced sampling, etc.
#'                        For survival analysis, observation weights are supported
#'                        in the \code{"logrank"}, \code{"suplogrank"}, and
#'                        \code{"coxgrad"} splitting rules. Weighted logrank and
#'                        suplogrank use a variance estimator that accounts for
#'                        the observation weights.
#'                        Weights must be finite and nonnegative with positive total mass;
#'                        individual zero weights are allowed. A sampled outer or embedded
#'                        tree with zero total observation weight produces a catchable error.
#'
#' @param var.prob        Variable probabilities for split variable selection. A
#'                        numeric vector of length \code{p} (number of predictors)
#'                        with finite nonnegative weights and positive total mass.
#'                        The vector is normalized internally and guides sampling
#'                        of split candidates. In particular,
#'                        requesting the whole candidate pool returns all variables,
#'                        and some regression/classification linear combination screening
#'                        paths sample uniformly. Zero-probability exclusion is therefore
#'                        not guaranteed across these paths.
#'                        If \code{NULL} (default), uniform sampling is used.
#'
#' @param importance      Whether to calculate variable importance. Use \code{TRUE}
#'                        or \code{"permute"} to measure the deterioration in
#'                        out-of-bag (OOB) performance after permuting a predictor.
#'                        With \code{"distribute"}, an OOB observation follows both
#'                        children whenever a split uses the assessed predictor.
#'                        Routing probabilities are proportional to child node weights,
#'                        or sample counts for unweighted data. Regression and
#'                        classification average terminal-node losses; survival
#'                        evaluates concordance using averaged terminal-node risk scores.
#'                        Both strategies are available for regression, classification,
#'                        and survival. Survival importance uses OOB concordance and
#'                        is unavailable when no usable OOB comparisons exist.
#'
#' @param linear.comb    Maximum number of variables in each linear combination
#'                       split. Default: 1 (single-variable splits). Values greater
#'                       than 1 do not require \code{reinforcement = TRUE}.
#'                       See \code{linear.comb.method} and \code{param.control}.
#' @param linear.comb.method Method for constructing linear combinations.
#'                       \code{"default"} selects \code{"sir"} for regression,
#'                       \code{"lda"} for classification, and \code{"coxph"} for
#'                       survival. See \code{param.control} for all methods.
#' @param split.rule     Splitting criterion. Default \code{"default"} selects the
#'                       standard rule for each model. For survival: \code{"logrank"},
#'                       \code{"suplogrank"}, or \code{"coxgrad"}. See \code{param.control}.
#' @param var.mode        Variance estimation mode: \code{"none"} (default),
#'                        \code{"matched"}, \code{"IJ"}, or \code{"jack"}.
#'                        \code{TRUE} is equivalent to \code{"matched"}.
#'                        Matched sampling supports prediction and importance
#'                        variance; IJ and jackknife modes support prediction
#'                        variance. Active modes adjust resampling settings and
#'                        retain observation tracking. See \code{param.control}.
#'
#' @param param.control A list of additional forest and embedded-model settings.
#' Reinforcement is available for regression, classification, and survival.
#' Embedded-model defaults are provided automatically but may need tuning.
#'
#' Embedded-model settings:
#' \itemize{
#' \item \code{embed.ntrees}: number of embedded trees. Default: 50.
#' \item \code{embed.mtry}: candidate-variable proportion when less than 1,
#' or count when at least 1. Default: 0.5.
#' \item \code{embed.nmin}: embedded terminal node size. Default: 5.
#' \item \code{embed.nsplit}: random cutting points; 0 uses the best split. Default: 3.
#' \item \code{embed.resample.replace}: sample with replacement. Default: TRUE.
#' \item \code{embed.resample.prob}: sample proportion within the current node.
#' Default: 0.9.
#' \item \code{embed.mute}: variables to mute per split, as a count when at least
#' 1 or a proportion when less than 1. Default: 0 (no muting).
#' \item \code{embed.protect}: number of top variables protected from muting.
#' Default: \code{ceiling(log(n))}.
#' \item \code{embed.threshold}: threshold relative to the best importance for
#' inclusion in the protected set. Default: 0.25.
#' }
#'
#' Linear combination settings:
#' \itemize{
#' \item \code{linear.comb}: maximum variables per combination. Default: 1.
#' Values greater than 1 select combination splits independently of
#' \code{reinforcement}. Regression and classification use embedded importance
#' screening when the node has at least \code{3 * embed.nmin} observations and
#' direct screening at smaller nodes. Survival combination splits screen
#' \code{mtry} candidates without an embedded forest.
#' \item \code{linear.comb.method}: regression accepts \code{"naive"},
#' \code{"lm"}, \code{"pca"}, and \code{"sir"} (default); classification accepts
#' \code{"lda"} (default), \code{"naive"}, \code{"random"}, and \code{"logistic"};
#' survival accepts \code{"coxph"} (default) and \code{"naive"}.
#' }
#'
#' \code{split.rule} selects the splitting criterion:
#' \itemize{
#' \item Regression: \code{"var"} (variance reduction).
#' \item Classification: \code{"gini"} (Gini index).
#' \item Survival: \code{"logrank"} (default), \code{"suplogrank"}, or \code{"coxgrad"}.
#' }
#'
#' \code{resample.track} retains the in-bag count for each observation and tree.
#' Default: FALSE. All active variance modes enable it automatically.
#'
#' \code{var.mode} prepares variance estimation during fitting:
#' \itemize{
#' \item \code{"none"} (default): no variance estimation.
#' \item \code{"matched"} (or \code{TRUE}): matched-sample U-statistic
#' decomposition for prediction and variable importance variance. Uses paired
#' half-samples without replacement, sets \code{resample.prob = 0.5}, and
#' generates \code{resample.preset}, replacing any supplied preset. An odd
#' \code{ntrees} is reduced to the preceding even number; at least 2 are required.
#' \item \code{"IJ"}: infinitesimal jackknife prediction variance.
#' \item \code{"jack"}: jackknife-after-bootstrap prediction variance.
#' }
#' IJ and jackknife modes set \code{resample.replace = TRUE}, retain the requested
#' \code{resample.prob}, and generate bootstrap counts instead of using a supplied
#' \code{resample.preset}. They do not estimate variable importance variance.
#' All active variance modes enable \code{importance = "distribute"} when
#' importance was disabled; an explicitly requested importance method is retained.
#' Stable variance estimates generally require many trees. IJ and jackknife
#' modes recommend at least 2000 trees.
#'
#' \code{time.grid.size} is 0 by default and retains all distinct observed failure
#' times for survival fitting. A positive integer must be at least the number of
#' distinct failure times; it does not add times to the training grid. Requests
#' that would reduce the training grid are rejected. Use \code{band.grid.size} in
#' \code{predict()} to reduce the evaluation grid instead.
#'
#' @param ncores          Number of CPU logical cores. Default is 0 (using all
#'                        available cores when compiled with OpenMP). Builds
#'                        without OpenMP run serially.
#'
#' @param verbose         Whether info should be printed.
#'
#' @param seed            Random seed number to replicate a previously fitted forest.
#'                        Internally, the `xoshiro256++` generator is used. If not
#'                        specified, a seed will be generated automatically and
#'                        recorded.
#'
#' @param ...             Additional arguments.
#'
#' @return
#'
#' A fitted object of class \code{RLT}. OOB predictions for each training
#' observation use only trees that did not sample that observation.
#' \describe{
#' \item{Prediction}{OOB numeric predictions for regression, factor labels for
#' classification, or a matrix of hazard increments for survival. Survival rows
#' correspond to observations and columns to \code{timepoints}.}
#' \item{Error}{OOB mean squared error for regression, misclassification rate for
#' classification, or \eqn{1-C} for survival. The survival concordance score uses
#' the sum of the OOB cumulative hazards over the failure-time grid, not just
#' the cumulative hazard at the final time. Lower error is better.}
#' \item{VarImp}{Variable importance when requested. Use \code{\link{importance}}
#' to obtain a named summary.}
#' \item{FittedForest}{Fitted tree structures.}
#' \item{xlevels}{Training label dictionaries for factor, ordered-factor, and character
#' data-frame columns, with \code{NULL} entries for numerical features.}
#' \item{ObsTrack}{Provided if \code{resample.track = TRUE}, \code{var.mode != "none"},
#'                 or if \code{resample.preset} was supplied. This is an \code{n} \eqn{\times} \code{ntrees}
#'                 matrix that has the same meaning as \code{resample.preset}.}
#' }
#'
#' Classification forests also provide:
#' \describe{
#' \item{nclass}{Number of response classes.}
#' \item{Prob}{OOB class probabilities, with one row per observation and one
#' column per class in the order given by \code{ylabels}.}
#' }
#'
#' Survival forests also provide:
#' \describe{
#' \item{timepoints}{ordered observed failure times}
#' \item{NFail}{The number of observed failure times}
#' }
#'
#' @references
#' \itemize{
#'  \item Zhu, R., Zeng, D., & Kosorok, M. R. (2015) "Reinforcement Learning Trees." Journal of the American Statistical Association. 110(512), 1770-1784.
#'  \item Xu, T., Zhu, R., & Shao, X. (2024) "On variance estimation of random forests with Infinite-order U-statistics." Electronic Journal of Statistics. 18(1), 2135-2207. \doi{10.1214/24-EJS2247}.
#'  \item Formentini, S. E., Liang, W., & Zhu, R. (2022) "Confidence Band Estimation for Survival Random Forests." arXiv preprint arXiv:2204.12038. \url{https://arxiv.org/abs/2204.12038}.
#' }
#'
#' @examples
#' \donttest{
#'   set.seed(42)
#'   x <- matrix(rnorm(300 * 5), ncol = 5)
#'   y <- rowSums(x[, 1:2]) + rnorm(300)
#'   fit <- RLT(x, y, ntrees = 100)
#'   print(fit)
#' }
#' @export
RLT <- function(x, y, censor = NULL, model = NULL,
				ntrees = if (reinforcement) 100 else 500,
				mtry = max(1, as.integer(ncol(x)/2)),
				nmin = 5,
				alpha = 0,
				nsplit = 0,
				resample.replace = TRUE,
				resample.prob = if(resample.replace) 1 else 0.8,
				resample.preset = NULL,
				obs.w = NULL,
				var.prob = NULL,
		importance = FALSE,
			reinforcement = FALSE,
			linear.comb = 1,
			linear.comb.method = "default",
			split.rule = "default",
			var.mode = "none",
				param.control = list(),
				ncores = 0,
				verbose = 0,
				seed = NULL,
				...)
{
  # check model type
  if (is.null(model))
  {
    if (is.factor(y)) model = "classification"

    if (is.numeric(y) && !is.null(censor)) model = "survival"

    if (is.numeric(y) && is.null(censor)) model = "regression"

    if (is.null(model)) stop("Please specify the model type")
  }

  if (!match(model, c("regression", "classification",
                     "quantile", "survival"), nomatch = 0))
    stop("model type not recognized")

  # check input data
  if (missing(x)) stop("x is missing")

  if (missing(y)) stop("y is missing")

  if (!is.matrix(x) && !is.data.frame(x)) stop("x must be a matrix or a data.frame")
  if (is.matrix(x) && !is.numeric(x) && !is.logical(x))
    stop("x must be a numeric matrix; character matrices are not supported")

  if (model == "survival")
    if (missing(censor)) stop("censor is missing")

  p = ncol(x)
  n = nrow(x)

  # check some parameters
  # we only check common parameters here
  # model specific parameters will be checked inside each model fitting function
  ntrees = check_ntrees(ntrees)
  mtry = check_mtry(mtry, p)
  nmin = check_nmin(nmin)
  alpha = max(0, min(alpha, 0.5))
  storage.mode(alpha) <- "double"
  nsplit = check_nsplit(nsplit)
  resample.replace = check_resamplereplace(resample.replace)
  resample.prob = check_resampleprob(resample.prob)
  importance = check_importance(importance)
  reinforcement = check_reinforcement(reinforcement)
  ncores = check_ncores(ncores)
  verbose = check_verbose(verbose)
  seed = check_seed(seed) # will randomly generate seed if not provided

  # check observation weights
  if (is.null(obs.w))
  {
    obs.w = ARMA_EMPTY_VEC()
    use.obs.w = 0L
  }else{
    obs.w = check_obsw(obs.w, n)
    use.obs.w = 1L
  }

  # check variable probabilities
  if (is.null(var.prob))
  {
    var.prob = ARMA_EMPTY_VEC()
    use.var.prob = 0L
  }else{
    var.prob = check_varprob(var.prob, p)
    use.var.prob = 1L
  }

  #
  param = list("n" = n,
               "p" = p,
               "ntrees" = ntrees,
               "mtry" = mtry,
               "nmin" = nmin,
               "alpha" = alpha,
               "nsplit" = nsplit,
               "resample.replace" = resample.replace,
               "resample.prob" = resample.prob,
               "use.obs.w" = use.obs.w,
               "use.var.prob" = use.var.prob,
               "importance" = importance,
               "reinforcement" = reinforcement,
               "ncores" = ncores,
               "verbose" = verbose,
               "seed" = seed)

  # time.grid.size for survival
  if (is.null(param.control$time.grid.size)) {
    time.grid.size <- 0
  } else time.grid.size = param.control$time.grid.size

  # inject var.mode from top-level argument into param.control
  # (user can also pass via param.control, but top-level takes precedence)
  if (!identical(var.mode, "none"))
    param.control$var.mode <- var.mode

  # inject linear.comb and linear.comb.method from top-level arguments
  if (!identical(linear.comb, 1))
    param.control$linear.comb <- linear.comb
  if (!identical(linear.comb.method, "default"))
    param.control$"linear.comb.method" <- linear.comb.method
  if (!identical(split.rule, "default"))
    param.control$"split.rule" <- split.rule

  # check control parameters
  param.control = check_control(param.control, param)

  # check importance feasibility (OOB size vs importance type)
  check_importance_settings(importance, n, resample.replace, resample.prob)

  # reset some parameters if var.mode is needed
  if (param.control$var.mode != "none")
  {
    var.msg = character()
    vm = param.control$var.mode

    # --- matched U-statistic specific settings ---
    if (vm == "matched")
    {
      # subsampling (no replacement)
      if (param$resample.replace)
      {
        var.msg = c(var.msg, paste0("  resample.replace: TRUE -> FALSE"))
        param$resample.replace = 0L
      }

      # resample.prob = 0.5 (each half gets n/2)
      if (param$resample.prob != 0.5)
      {
        var.msg = c(var.msg, paste0("  resample.prob: ", param$resample.prob, " -> 0.5"))
        param$resample.prob = 0.5
      }

      # even ntrees (first half paired with second half)
      if (param$ntrees %% 2 != 0)
      {
        old.ntrees = param$ntrees
        param$ntrees = 2*floor(param$ntrees/2)
        var.msg = c(var.msg, paste0("  ntrees: ", old.ntrees, " -> ", param$ntrees))
      }

      if (param$ntrees < 2)
        stop("var.mode requires at least 2 trees for matched sampling")

      # resample.preset auto-generated by matched sampling
      if (!is.null(resample.preset))
        var.msg = c(var.msg, "  resample.preset: user-supplied -> auto-generated (matched sampling)")
    }

    # --- IJ / jackknife specific settings ---
    if (vm %in% c("IJ", "jack"))
    {
      # IJ/jack work best with bootstrap sampling; ensure it is enabled
      if (!param$resample.replace)
      {
        var.msg = c(var.msg, paste0("  resample.replace: FALSE -> TRUE"))
        param$resample.replace = 1L
      }

      # recommend large B for stable variance estimates
      if (param$ntrees < 2000)
        message("Note: ", vm, " variance estimation is more stable with ntrees >= 2000 (current: ",
                param$ntrees, ")")

      # resample.preset auto-generated for standard bootstrap tracking
      if (!is.null(resample.preset))
        var.msg = c(var.msg, "  resample.preset: user-supplied -> auto-generated (bootstrap tracking)")
    }

    # --- common for all variance modes ---
    # importance must be set for VI variance estimation
    if (param$importance == 0)
    {
      # user did not request importance; enable it automatically
      var.msg = c(var.msg, paste0("  importance: FALSE -> distribute"))
      param$importance = 2L
    }

    # output summary of changes
    if (length(var.msg) > 0)
      message("var.mode ('", vm, "') adjusted the following settings:\n",
              paste(var.msg, collapse = "\n"))
  }

  # construct resample.preset
  if (is.null(resample.preset) && param.control$var.mode == "none")
  {
    resample.preset = ARMA_EMPTY_UMAT()
  }else if (param.control$var.mode %in% c("IJ", "jack")){
    # IJ/jack: empty ObsTrack, let C++ fill it with standard bootstrap counts
    resample.preset = ARMA_EMPTY_UMAT()
    param.control$resample.track = 1L
  }else{
    resample.preset = check_resamplepreset(resample.preset, param, param.control)
    param.control$resample.track = 1L
  }

  # prepare x, continuous and categorical
  if (is.data.frame(x))
  {
    # data.frame, check for categorical variables
    xlevels <- lapply(x, function(value) {
      if (is.factor(value)) levels(value)
      else if (is.character(value)) levels(factor(value))
      else NULL
    })
    # Ordered factors and character columns retain their continuous rank codes.
    ncat <- vapply(x, function(value) {
      if (is.factor(value) && !is.ordered(value)) nlevels(value) else 1L
    }, integer(1))
  }else{
    # numerical matrix for x, all continuous
    ncat <- rep(1, p)
    names(ncat) <- colnames(x)
    xlevels <- rep(list(NULL), p)

    # Check for variables with very few unique values (potential categoricals)
    for (j in 1:p)
    {
      n_unique <- length(unique(x[, j]))
      if (n_unique <= 10)
      {
        col_name <- if (!is.null(colnames(x))) colnames(x)[j] else paste0("V", j)
        warning(sprintf(
          "Variable '%s' (column %d) has only %d unique values. Consider converting to a factor and using a data.frame input to treat it as categorical.",
          col_name, j, n_unique), call. = FALSE)
      }
    }
  }

  storage.mode(ncat) <- "integer"

  if (max(ncat) > 53)
    stop("cannot handle categorical predictors with more than 53 categories")

  xnames = colnames(x)
  x <- data.matrix(x)

  # set all parameters
  # convert var.mode string to integer for C++
  param.control$var.mode = switch(param.control$var.mode,
                                   "none" = 0L,
                                   "matched" = 1L,
                                   "IJ" = 2L,
                                   "jack" = 3L)
  param.all = append(param, param.control)

  # fit model

  if (model == "regression")
  {
    RLT.fit = RegForest(x, y, ncat,
                        obs.w, var.prob,
                        resample.preset,
                        param.all, ...)
  }

  if (model == "classification")
  {
    RLT.fit = ClaForest(x, y, ncat,
                        obs.w, var.prob,
                        resample.preset,
                        param.all, ...)
  }

  if (model == "quantile")
  {
    RLT.fit = QuanForest(x, y, ncat,
                        obs.w, var.prob,
                        resample.preset,
                        param.all, ...)
  }

  if (model == "survival")
  {
    RLT.fit = SurvForest(x, y, censor,
                         ncat, time.grid.size,
                         obs.w, var.prob,
                         resample.preset,
                         param.all, ...)
  }

  RLT.fit$"xnames" = xnames
  RLT.fit$"xlevels" = xlevels

  if (importance)
    rownames(RLT.fit$"VarImp") = xnames

  return(RLT.fit)
}


