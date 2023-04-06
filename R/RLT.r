#' @title                 Reinforcement Learning Trees
#' @description           Fit models for regression, classification and survival 
#'                        analysis using reinforced splitting rules. The model 
#'                        fits regular random forest models by default unless the
#'                        parameter \code{reinforcement} is set to `"TRUE"`. Using 
#'                        \code{reinforcement = TRUE} activates embedded model for 
#'                        splitting variable selection and allows linear combination 
#'                        split. To specify parameters of embedded models, see 
#'                        definition of \code{param.control} for details.
#'                        
#' @param x               A `matrix` or `data.frame` of features. If \code{x} is 
#'                        a data.frame, then all factors are treated as categorical 
#'                        variables, which will go through an exhaustive search of 
#'                        splitting criteria.
#' 
#' @param y               Response variable. a `numeric`/`factor` vector.
#'                        
#' @param censor          Censoring indicator if survival model is used.
#' 
#' @param model           The model type: `"regression"`, `"classification"`, 
#'                        `"quantile"`, `"survival"` or `"graph"`.
#'                        
#' @param reinforcement   Should reinforcement splitting rule be used. Default
#'                        is `"FALSE"`, i.e., regular random forests with marginal 
#'                        search of splitting variable. When it is activated, an
#'                        embedded model is fitted to find the best splitting variable
#'                        or a linear combination of them, if \code{linear.comb} $> 1$. 
#'                        They can also be specified in \code{param.control}.
#'                        
#' @param ntrees          Number of trees, `ntrees = 100` if reinforcement is
#'                        used and `ntrees = 1000` otherwise.
#'                        
#' @param mtry            Number of randomly selected variables used at each 
#'                        internal node.
#'                        
#' @param nmin            Terminal node size. Splitting will stop when the internal 
#'                        node size is less equal to \code{nmin}.
#'                        
#' @param split.gen       How the cutting points are generated: `"random"`, 
#'                        `"rank"` or `"best"`. If minimum child node size is 
#'                        enforced (\code{alpha} $> 0$), then `"rank"` and `"best"` 
#'                        should be used.
#'
#' @param nsplit          Number of random cutting points to compare for each 
#'                        variable at an internal node.
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
#'                        or weighted log-rank test. But they will not be used for 
#'                        sampling observations. Once can use \code{resample.preset}
#'                        instead for balanced sampling, etc. This feature is 
#'                        experimental and is not implemented in all models.
#' 
#' @param var.w           Variable weights. If this is supplied, the default is to 
#'                        perform weighted sampling of \code{mtry} variables. For 
#'                        other usage, see the details of \code{split.rule} in 
#'                        \code{param.control}.
#'                        
#' @param importance      Whether to calculate variable importance measures. When
#'                        set to `"TRUE"`, the calculation follows Breiman's 
#'                        original permutation strategy. 
#'                        
#' @param param.control   A list of additional parameters. This can be used to 
#'                        specify other features in a random forest or set embedded 
#'                        model parameters for reinforcement splitting rules. 
#'                        Using \code{reinforcement = TRUE} will automatically
#'                        generate some default tuning for the embedded model. 
#'                        They are not necessarily optimized.
#'                        \itemize{
#'                        \item \code{embed.ntrees}: number of trees in the embedded model
#'                        \item \code{embed.resample.prob}: proportion of samples 
#'                              (of the internal node) in the embedded model 
#'                        \item \code{embed.mtry}: number or proportion of variables
#'                        \item \code{embed.split.gen} random cutting point search 
#'                              method (`"random"`, `"rank"` or `"best"`) 
#'                        \item \code{embed.nsplit} number of random cutting points.
#'                        }
#'                        
#'                        \code{linear.comb} is a separate feature that can be 
#'                        activated with or without using reinforcement. It creates 
#'                        linear combination of features as the splitting rule. 
#'                        Currently only available for regression. 
#'                        \itemize{
#'                        \item In reinforcement mode, a linear combination is created 
#'                              using the top continuous variables from the embedded 
#'                              model. If a categorical variable is the best, then 
#'                              a regular split will be used. The splitting point 
#'                              will be searched based on \code{split.rule} of the
#'                              model. 
#'                        \item In non-reinforcement mode, a marginal screening 
#'                              is performed and the top features are used to construct 
#'                              the linear combination. This is an experimental feature. 
#'                        }
#'                        
#'                        \code{split.rule} is used to specify the criteria used 
#'                        to compare different splittings. Here are the available 
#'                        choices. The first one is the default:
#'                        \itemize{
#'                        \item Regression: `"var"` (variance reduction); `"pca"` 
#'                              and `"sir"` can be used for linear combination splits
#'                        \item Classification: `"gini"` (gini index)
#'                        \item Survival: `"logrank"` (log-rank test), `"suplogrank"`, 
#'                              `"coxgrad"`.
#'                        \item Quantile: `"ks"` (Kolmogorov-Smirnov test)
#'                        \item Graph: `"spectral"` (spectral embedding with variance 
#'                        reduction)
#'                        }
#'                        
#'                        \code{resample.track} indicates whether to keep track 
#'                        of the observations used in each tree.
#'                        
#'                        \code{var.ready} this is a feature to allow calculating variance 
#'                        (hence confidence intervals) of the random forest prediction. 
#'                        Currently only available for regression (Xu, Zhu & Shao, 2023) 
#'                        and confidence band in survival models (Formentini, Liang & Zhu, 2023). 
#'                        Please note that this only perpares the model fitting 
#'                        so that it is ready for the calculation. To obtain the 
#'                        confidence intervals, please see the prediction function. 
#'                        Specifying \code{var.ready = TRUE} has the following effect 
#'                        if these parameters are not already provided. For details 
#'                        of their restrictions, please see the orignal paper.
#'                        \itemize{
#'                        \item \code{resample.preset} is constructed automatically
#'                        \item \code{resample.replace} is set to `FALSE`
#'                        \item \code{resample.prob} is set to \eqn{n / 2}
#'                        \item \code{resample.track} is set to `TRUE`
#'                        }
#'                        
#'                        It is recommended to use a very large \code{ntrees}, 
#'                        e.g, 10000 or larger. For \code{resample.prob} greater 
#'                        than \eqn{n / 2}, one should consider the bootstrap 
#'                        approach in Xu, Zhu & Shao (2023).
#'                        
#'                        \code{alpha} force a minimum proportion of samples 
#'                        (of the parent node) in each child node.
#'                        
#'                        \code{failcount} specifies the unique number of failure 
#'                        time points used in survival model. By default, all failure 
#'                        time points will be used. A smaller number may speed up 
#'                        the computation. The time points will be chosen uniformly 
#'                        on the quantiles of failure times, while must include the 
#'                        minimum and the maximum. 
#'                        
#' @param ncores          Number of cpu logical cores. Default is 0 (using all 
#'                        available cores).
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
#' A \code{RLT} fitted object, constructed as a list consisting
#' \itemize{
#' \item{FittedForest}{Fitted tree structures}
#' \item{VarImp}{Variable importance measures, if \code{importance = TRUE}}
#' \item{Prediction}{Out-of-bag prediction}
#' \item{Error}{Out-of-bag prediction error, adaptive to the model type}
#' \item{ObsTrack}{Provided if \code{resample.track = TRUE}, \code{var.ready = TRUE},
#'                 or if \code{resample.preset} was supplied. This is an \code{n} \eqn{\times} \code{ntrees} 
#'                 matrix that has the same meaning as \code{resample.preset}.}
#' }
#' 
#' For classification forests, these items are further provided or will replace 
#' the regression version
#' \itemize{
#' \item{NClass}{The number of classes}
#' \item{Prob}{Out-of-bag predicted probability}
#' }
#' 
#' For survival forests, these items are further provided or will replace the 
#' regression version
#' \itemize{
#' \item{timepoints}{ordered observed failure times}
#' \item{NFail}{The number of observed failure times}
#' \item{Prediction}{Out-of-bag prediciton of hazard function}
#' }
#' 
#' @references 
#' \itemize{
#'  \item Zhu, R., Zeng, D., & Kosorok, M. R. (2015) "Reinforcement Learning Trees." Journal of the American Statistical Association. 110(512), 1770-1784.
#'  \item Xu, T., Zhu, R., & Shao, X. (2023) "On Variance Estimation of Random Forests with Infinite-Order U-statistics." arXiv preprint arXiv:2202.09008.
#'  \item Formentini, S. E., Wei L., & Zhu, R. (2022) "Confidence Band Estimation for Survival Random Forests." arXiv preprint arXiv:2204.12038.
#' }
#' 
#' \donttest{}
#' 
#' @export
RLT <- function(x, y, censor = NULL, model = NULL,
        				ntrees = if (reinforcement) 100 else 500,
        				mtry = max(1, as.integer(ncol(x)/3)),
        				nmin = max(1, as.integer(log(nrow(x)))),
        				split.gen = "random",
        				nsplit = 1,
        				resample.replace = TRUE,
        				resample.prob = if(resample.replace) 1 else 0.8,
        				resample.preset = NULL,
        				obs.w = NULL,
        				var.w = NULL,
         				importance = FALSE,
        				reinforcement = FALSE,
        				param.control = list(),
        				ncores = 0,
        				verbose = 0,
        				seed = NULL,
        				...)
{
  # check model type
  if (is.null(model))
    stop("Please specify the model type")
  
  if (!match(model, c("regression", "classification", "quantile", 
                     "survival", "graph"), nomatch = 0))
    stop("model type not recognized")
  
  # check input data
  if (missing(x)) stop("x is missing")
  
  if (model != "graph")
    if (missing(y)) stop("y is missing")
  
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
  split.gen = check_splitgen(split.gen) # convert to numerical
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
  
  # check variable weights  
  if (is.null(var.w))
  {
    var.w = ARMA_EMPTY_VEC()    
    use.var.w = 0L
  }else{
    var.w = check_varw(var.w, p)   
    use.var.w = 1L
  }
  
  # 
  param = list("n" = n,
               "p" = p,
               "ntrees" = ntrees,
               "mtry" = mtry,
               "nmin" = nmin,
               "split.gen" = split.gen,
               "nsplit" = nsplit,
               "resample.replace" = resample.replace,
               "resample.prob" = resample.prob,
               "use.obs.w" = use.obs.w,
               "use.var.w" = use.var.w,
               "importance" = importance,
               "reinforcement" = reinforcement,
               "ncores" = ncores,
               "verbose" = verbose,
               "seed" = seed)
  
  # failcount for survival
  if (is.null(param.control$failcount)) {
    failcount <- 0
  } else failcount = param.control$failcount
  
  # check control parameters
  param.control = check_control(param.control, param)
  
  # reset some parameters if var.ready is needed
  if (param.control$var.ready)
  {
    if (param$resample.replace)
    {
      if (verbose) warning("resample.replace is set to FALSE due to var.ready\n")
      param$resample.replace = 0L
    }
    
    if (param$resample.prob > 0.5)
    {
      if (verbose) warning("resample.prob is set to 0.5 due to var.ready\n")
      param$resample.prob = 0.5
    }
    
    if (param$ntrees %% 2 != 0)
    {
      param$ntrees = 2*floor(param$ntrees/2)
      if (verbose) warning(paste("ntrees is set to", param$ntrees, "due to var.ready\n"))
    }
    
    if (!is.null(resample.preset))
      if (verbose) warning("resample.preset is overwritten due to var.ready\n")
  }
  
  # construct resample.preset
  if (is.null(resample.preset) & param.control$var.ready == 0)
  {
    resample.preset = ARMA_EMPTY_UMAT()
  }else{
    resample.preset = check_resamplepreset(resample.preset, param, param.control)
    param.control$resample.track = 1L
  }

  # prepare x, continuous and categorical
  if (is.data.frame(x))
  {
    # data.frame, check for categorical variables
    xlevels <- lapply(x, function(x) if (is.factor(x)) levels(x) else 0)
    ncat <- sapply(xlevels, length)
    
    ## Treat ordered factors as numeric.
    ncat <- ifelse(sapply(x, is.ordered), 1, ncat)
  }else{
    # numerical matrix for x, all continuous
    ncat <- rep(1, p)
    names(ncat) <- colnames(x)
    xlevels <- as.list(rep(0, p))
  }
  
  storage.mode(ncat) <- "integer"
  
  if (max(ncat) > 53)
    stop("cannot handle categorical predictors with more than 53 categories")
  
  xnames = colnames(x)
  x <- data.matrix(x)  
  
  # set all parameters
  param.all = append(param, param.control)

  # fit model
  
  if (model == "regression")
  {
    RLT.fit = RegForest(x, y, ncat, 
                        obs.w, var.w, 
                        resample.preset, 
                        param.all, ...)
  }

  if (model == "classification")
  {
    if (verbose > 0) cat("runing classification forest ... \n ")
    RLT.fit = ClaForest(x, y, ncat,
                        obs.w, var.w, 
                        resample.preset, 
                        param.all, ...)
  }

  if (model == "survival")
  {
    if (verbose > 0) cat("runing survival forest ... \n ")
    RLT.fit = SurvForest(x, y, censor, 
                         ncat, failcount,
                         obs.w, var.w, 
                         resample.preset, 
                         param.all, ...)
  }

  if (model == "quantile")
  {
    if (verbose > 0) cat("runing quantile forest ... \n ")
    RLT.fit = QuanForest(x, y, ncat, 
                         obs.w, var.w, 
                         resample.preset, 
                         param.all, ...)
  }
  
  if (model == "graph")
  {
    if (verbose > 0) cat("runing graph forest ... \n ")
    RLT.fit = QuanForest(x, y, ncat, 
                         obs.w, var.w, 
                         resample.preset, 
                         param.all, ...)
  }
  
  RLT.fit$"xnames" = xnames
  
  if (importance)
    rownames(RLT.fit$"VarImp") = xnames

  return(RLT.fit)
}
