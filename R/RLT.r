#' @title                 Reinforcement Learning Trees
#' @description           Fit models for regression, classification and 
#'                        survival analysis using reinforced splitting rules.
#'                        The model reduces to regular random forests if 
#'                        reinforcement is turned off.
#'                      
#' If \code{x} is a data.frame, then all factors are treated as categorical variables. 
#' 
#' To specify parameters of embedded models when \code{reinforcement = TRUE},
#' users can supply the following in the \code{param.control} list:
#' 
#' \itemize{\item \code{embed.ntrees}: number of trees in the embedded model
#' \item \code{embed.resample.prob}: proportion of samples (of the internal node)
#' in the embedded model \item \code{embed.mtry}: number or proportion of variables
#' \item \code{embed.nmin} terminal node size \item \code{embed.split.gen} random 
#' cutting point search method (`"random"`, `"rank"` or `"best"`) \item \code{embed.nsplit} 
#' number of random cutting points.}
#' 
#' For some other experimental features, please see \code{\link{check_param_RLT}}.
#'                        
#' @param x               A `matrix` or `data.frame` of features
#' 
#' @param y               Response variable. a `numeric`/`factor` vector.
#'                        
#' @param censor          The censoring indicator if survival model is used.
#' 
#' @param model           The model type: `"regression"`, `"classification"` 
#'                        or `"survival"`.
#'                        
#' @param reinforcement   Should reinforcement splitting rule be used. Default
#'                        is `"FALSE"`, i.e., regular random forests. When it
#'                        is activated, embedded model tuning parameters are 
#'                        automatically chosen. They can also be specified in 
#'                        `RLT.control`.
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
#'                        matrix and cannot contain negative values. Extremely 
#'                        large counts are not recommended, and the sum of 
#'                        each column cannot exceed \eqn{n}. If provided, then 
#'                        resample.track will be set to `TRUE`. This is an feature 
#'                        is mainly use when estimating variances of a random forest. 
#'                        Use at your own risk. 
#'                        
#' @param resample.track  Whether to keep track of the observations used in each tree.
#' 
#' @param obs.w           Observation weights
#' 
#' @param var.w           Variable weights. If this is supplied, the default is to 
#'                        perform weighted sampling of \code{mtry} variables. For 
#'                        other usage, see the details of \code{split.rule} in 
#'                        \code{\link{check_param_RLT}}.
#' 
#' @param linear.comb     When \code{linear.comb} is larger than 1, a linear combination 
#'                        split is used. When \code{reinforcement} is \code{TRUE}, the 
#'                        variables with the highest potential at an internal node is used.
#'                        When \code{reinforcement} is \code{FALSE}, a marginal screening is 
#'                        used. In both cases, SIR and SAVE are used to determine the coefficients
#'                        of the combination. When a categorical variable has the highest 
#'                        potential, then a single variable is used. Currently restricted to
#'                        less than 5 number of variables in the linear combination. 
#'                        
#' @param importance      Whether to calculate variable importance measures. The calculation 
#'                        follows Breiman's original permutation strategy. 
#' 
#' @param var.ready       Construct \code{resample.preset} automatically to allow variance 
#'                        estimations for prediction. If this is used, then \code{resample.replace} 
#'                        will be set to `FALSE` and \code{resample.prob} should be no 
#'                        larger than \eqn{n / 2}. It is recommended to use a very large
#'                        \code{ntrees}, e.g, 10000 or larger. For \code{resample.prob} greater than 
#'                        \eqn{n / 2}, one should use the \code{\link{Reg_Var_Forest}} function. 
#'                        
#' @param param.control   A list of additional parameters. This can be used to 
#'                        specify other features in a random forest and set embedded 
#'                        model parameters for reinforcement splitting rules. 
#'                        See \code{check_param_RLT} and \code{set_embed_param} for 
#'                        more details. using \code{reinforcement = TRUE} will automatically
#'                        generate some default tunings. However, they are not necessarily
#'                        good. 
#' 
#' @param ncores          Number of cores. Default is 0 (using all available cores).
#' 
#' @param verbose         Whether fitting info should be printed.
#' 
#' @param seed            Random seed number to replicate a previously fitted forest. 
#'                        Internally, the `xoshiro256++` generator is used. If not specified, 
#'                        a seed will be generated automatically. 
#'                        
#' @param ...             Additional arguments.
#' 
#' @return 
#' 
#' A \code{RLT} object, constructed as a list consisting
#' 
#' \item{FittedForest}{Fitted tree structures}
#' \item{VarImp}{Variable importance measures, if \code{importance = TRUE}}
#' \item{Prediction}{In-bag prediction values}
#' \item{OOBPrediction}{Out-of-bag prediction values}
#' \item{ObsTrack}{An \eqn{n \times} \code{ntrees} matrix that indicates which observations 
#'                 are used in each tree. Provided if \code{resample.preset}
#'                 is used or \code{resample.track = TRUE}.}
#'                 
#'  \strong{For Survival Forests}               
#'  \item{NFail}{The number of observed failure times}
#'  \item{VarImpCov}{if \code{VI.var=TRUE}, estimated covariance matrix for the variable importance}
#'  \item{cindex_tree}{Out-of-bag c-index for each tree}
#'  \item{cindex}{Out-of-bag c-index for the forest}
#'  \item{timepoints}{ordered observed failure times}
#'  \item{y.point}{order of \eqn{y} by observed failure times}               
#'                 
#' 
#' @references Zhu, R., Zeng, D., & Kosorok, M. R. (2015) "Reinforcement Learning Trees." Journal of the American Statistical Association. 110(512), 1770-1784.
#' 
#' \dontrun{}
#' 
#' @export RLT
RLT <- function(x, y, censor = NULL, model = NULL,
        				ntrees = if (reinforcement) 100 else 500,
        				mtry = max(1, as.integer(ncol(x)/3)),
        				nmin = max(1, as.integer(log(nrow(x)))),
        				split.gen = "random",
        				nsplit = 1,
        				resample.replace = TRUE,
        				resample.prob = if(resample.replace) 1 else 0.8,
        				resample.preset = NULL,
        				resample.track = FALSE,
        				obs.w = NULL,
        				var.w = NULL,
        				linear.comb = 1,
         				importance = FALSE,
        				var.ready = FALSE,
        				reinforcement = FALSE,
        				param.control = list(),
        				ncores = 0,
        				verbose = 0,
        				seed = NULL,
        				...)
{
  # check inputs
  
  if (missing(x)) stop("x is missing")
  if (missing(y)) stop("y is missing")
  
  # check model type
  model = check_input(x, y, censor, model)

  p = ncol(x)
  n = nrow(x)

  ntrees = max(ntrees, 1)
  storage.mode(ntrees) <- "integer"
  
  resample.prob = max(0, min(resample.prob, 1))
  storage.mode(resample.prob) <- "double"
  
  # check resample.preset
  if ( !is.null(resample.preset) )
  {
    if (!is.matrix(resample.preset))
      stop("resample.preset must be a matrix")
    
    if (nrow(resample.preset) != n | ncol(resample.preset) != ntrees)
      stop("Dimension of resample.preset does not match n x ntrees")
    
    if (any(resample.preset < 0))
    {
      warning("Negative entries in resample.preset will not be used as in-bag or out-of-bag samples.")
      #resample.preset[resample.preset < 0] = 0;
    }
    
    if ( any(colSums(resample.preset) > n) )
    {
      stop("Column sums in resample.preset should not be larger than n ...")
    }
    
    storage.mode(resample.preset) <- "integer"
    resample.track = TRUE
    
  }else{
    resample.preset = ARMA_EMPTY_UMAT();
  }  
  
  # set resample.preset if variance estimation is needed
  
  if (var.ready)
  {
    if (resample.replace)
      stop("Variance estimation for bootstrap samples is not avaliable")
    
    if (resample.prob > 0.5)
      stop("Variance estimation for resample.prob > 0.5 is not avaliable")
    
    if (ntrees %% 2 != 0)
      stop("Please use an even number of trees")

    resample.preset = matrix(0, n, ntrees)
    k = as.integer(resample.prob*n)
      
    for (i in 1:as.integer(ntrees/2) )
    {
      ab = sample(1:n, 2*k)
      a = ab[1:k]
      b = ab[-(1:k)]
      
      resample.preset[a, i] = 1
      resample.preset[b, i+ (ntrees/2)] = 1
    }
    
    storage.mode(resample.preset) <- "integer"
    
    resample.track = TRUE
  }
  
  
  # check observation weights  
  if (is.null(obs.w))
  {
    use.obs.w = 0L
    obs.w = ARMA_EMPTY_VEC()
  }else{
    use.obs.w = 1L
    obs.w = as.numeric(as.vector(obs.w))
    
    if (any(obs.w < 0))
      stop("observation weights cannot be negative")
    
    if (length(obs.w) != n)
      stop("length of observation weights must be n")
    
    storage.mode(obs.w) <- "double"    
    obs.w = obs.w/sum(obs.w)
  }
  
  # check variable weights  
  if (is.null(var.w))
  {
    use.var.w = 0L
    var.w = ARMA_EMPTY_VEC()
  }else{
    use.var.w = 1L
    var.w = as.numeric(as.vector(var.w))
    
    if (any(var.w < 0))
      stop("variable weights cannot be negative")
    
    if (length(var.w) != p)
      stop("length of variable weights must be p")
    
    storage.mode(var.w) <- "double"
    var.w = var.w/sum(var.w)
  }
  
  # check control parameters
  
  if (!is.null(param.control$VI.var)) {
    if(!is.logical(param.control$VI.var)){
      cat("VI.var must be TRUE or FALSE. Setting to FALSE. \n")
      param.control$VI.var <- FALSE
    }
  }else{
    param.control$VI.var <- FALSE
  }
  
  if (is.null(param.control$oob.prop) & param.control$VI.var==TRUE) {
    param.control$oob.prop <- 0.5
  }else if(param.control$VI.var==TRUE){
    if(param.control$oob.prop>0 & param.control$oob.prop<1){
    }else{
      cat("oob.prop must be in (0,1). Setting to default. \n")
      param.control$oob.prop <- 0.5
    }
  }
  
  if(importance==TRUE){
    if(param.control$VI.var & resample.prob<=0.5){
      
      k = as.integer(resample.prob*n)
      koob <- round(k*param.control$oob.prop)
      kinb <- k-koob

      
      if(dim(resample.preset)[1]==0){
        resample.preset = matrix(0, n, ntrees)
        for (i in 1:as.integer(ntrees/2) )
        {
          ab = sample(1:n, 2*k)
          a = ab[1:k]
          b = ab[-(1:k)]
          
          resample.preset[a, i] = 1
          resample.preset[b, i+ (ntrees/2)] = 1
        }
      }  
      
      for(i in 1:ntrees){
        resample.preset[,i] <- ifelse(resample.preset[,i]==0, -1, 
                                      resample.preset[,i])
        inbag <- c(1:n)[resample.preset[,i]==1]
        newoob <- sample(inbag, koob)
        resample.preset[newoob,i] <- 0
    }
    }else if(param.control$VI.var){
      cat(" Will not calculate importance variance when resamp.prob>0.5. \n ")
    }

  }

  # set all parameters
  param <- check_param_RLT(n, p, ntrees, mtry, nmin,
                           split.gen, nsplit,
                           resample.replace, resample.prob, 
                           resample.track,
                           use.obs.w, use.var.w,
                           linear.comb,
                           importance,
                           var.ready,                           
                           ncores, verbose,
                           reinforcement,
                           param.control)
  
  # random seed 
  
  if (is.null(seed) | !is.numeric(seed))
  {
    param$`seed` = runif(1) * .Machine$integer.max
  }else{
    param$`seed` = as.integer(seed)
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
    stop("Cannot handle categorical predictors with more than 53 categories")
  
  xnames = colnames(x)
  x <- data.matrix(x)
  
  # fit model
  
  if (model == "regression")
  {
    RLT.fit = RegForest(x, y, ncat, 
                        obs.w, var.w, 
                        resample.preset, 
                        param, ...)
  }

  if (model == "classification")
  {
    if (verbose > 0) cat(" run classification forest ... \n ")
    RLT.fit = ClaForest(x, y, ncat, 
                        obs.w, var.w, 
                        resample.preset, 
                        param, ...)
  }
  
  if (model == "survival")
  {
    if (verbose > 0) cat(" run survival forest ... \n ")
    RLT.fit = SurvForest(x, y, censor, ncat, 
                         obs.w, var.w, 
                         resample.preset, 
                         param, ...)
  }

  RLT.fit$"xnames" = xnames
  
  if (importance == TRUE)
    rownames(RLT.fit$"VarImp") = xnames

  return(RLT.fit)
}
