#' @title check_ntrees
#' @name check_ntrees
#' @keywords internal
check_ntrees <- function(ntrees)
{
  storage.mode(ntrees) <- "integer"
  
  if (is.na(ntrees))
    stop("ntrees should be numerical")
    
  if (ntrees < 1)
    stop("ntrees should be greater than 0")

  return(ntrees)
}

#' @title check_mtry
#' @name check_mtry
#' @keywords internal
check_mtry <- function(mtry, p)
{
  storage.mode(mtry) <- "integer"
  
  if (is.na(mtry))
    stop("mtry should be numerical")
  
  if (mtry < 1)
    stop("mtry cannot be less than 1")
  
  if (mtry > p)
    stop("mtry cannot be larger than p")

  return(mtry)
}

#' @title check_nmin
#' @name check_nmin
#' @keywords internal
check_nmin <- function(nmin)
{
  storage.mode(nmin) <- "integer"
  
  if (is.na(nmin))
    stop("nmin should be numerical")
  
  if (nmin < 1)
    stop("nmin cannot be less than 1")

  return(nmin)
}

#' @title check_splitgen
#' @name check_splitgen
#' @keywords internal
check_splitgen <- function(split.gen)
{
  split.gen.num = match(split.gen, c("random", "rank", "best"), 
                        nomatch = 0)
  
  if (split.gen.num == 0) 
    stop(paste("split.gen = ", split.gen, " is not recognized", sep = ""))
  
  storage.mode(split.gen.num) <- "integer"
  return(split.gen.num)
}

#' @title check_nsplit
#' @name check_nsplit
#' @keywords internal
check_nsplit <- function(nsplit)
{
  storage.mode(nsplit) <- "integer"
  
  if (is.na(nsplit))
    stop("nsplit should be numerical")
  
  if (nsplit < 0)
    stop("nsplit cannot be less than 1")
  
  return(nsplit)
}

#' @title check_resamplereplace
#' @name check_resamplereplace
#' @keywords internal
check_resamplereplace <- function(resample.replace)
{
  storage.mode(resample.replace) <- "integer"
  
  if (is.na(resample.replace))
    stop("resample.replace should be logical")
  
  resample.replace = ifelse(resample.replace != 0, 1L, 0L)
  return(resample.replace)
}

#' @title check_resampleprob
#' @name check_resampleprob
#' @keywords internal
check_resampleprob <- function(resample.prob)
{
  storage.mode(resample.prob) <- "double"  
  
  if (is.na(resample.prob))
    stop("resample.prob should be numerical")
  
  if (resample.prob <= 0 | resample.prob > 1)
    stop("resample.prob should be within the interval (0, 1]")
  
  return(resample.prob)
}

#' @title check_obsw
#' @name check_obsw
#' @keywords internal
check_obsw <- function(obs.w, n)
{
  obs.w = as.numeric(as.vector(obs.w))
  
  if (any(is.na(obs.w)))
    stop("observation weights (obs.w) should be numerical")
  
  if (any(obs.w < 0))
    stop("observation weights (obs.w) cannot be negative")
  
  if (length(obs.w) != n)
    stop("length of observation weights (obs.w) must be n")
  
  storage.mode(obs.w) <- "double"    
  obs.w = obs.w/sum(obs.w)
  
  return(obs.w)
}

#' @title check_varw
#' @name check_varw
#' @keywords internal
check_varw <- function(var.w, n)
{
  var.w = as.numeric(as.vector(var.w))

  if (any(is.na(var.w)))
    stop("variable weights (var.w) should be numerical")
  
  if (any(var.w < 0))
    stop("variable weights (var.w) cannot be negative")
  
  if (length(var.w) != p)
    stop("length of variable weights (var.w) must be p")
  
  storage.mode(var.w) <- "double"
  var.w = var.w/sum(var.w)
}

#' @title check_importance
#' @name check_importance
#' @keywords internal
check_importance <- function(importance)
{
  if (  match(importance, c(TRUE), nomatch = 0) )
    importance = "permute"
  
  importance.num = match(importance, c("permute", "random"), 
                         nomatch = 0)
  
  storage.mode(importance.num) <- "integer"
  return(importance.num)
}


#' @title check_reinforcement
#' @name check_reinforcement
#' @keywords internal
check_reinforcement <- function(reinforcement)
{
  storage.mode(reinforcement) <- "integer"
  
  if (is.na(reinforcement))
    stop("reinforcement should be logical")
  
  reinforcement = ifelse(reinforcement != 0, 1L, 0L)
  
  return(reinforcement)
}

#' @title check_ncores
#' @name check_ncores
#' @keywords internal
check_ncores <- function(ncores)
{
  storage.mode(ncores) <- "integer"
  
  if (is.na(ncores))
    stop("ncores should be numerical")
  
  if (ncores < 0)
    stop("ncores cannot be less than 0")
  
  return(ncores)
}

#' @title check_verbose
#' @name check_verbose
#' @keywords internal
check_verbose <- function(verbose)
{
  storage.mode(verbose) <- "integer"
  
  if (is.na(verbose))
    stop("verbose should be numerical")
  
  return(verbose)
}

#' @title check_seed
#' @name check_seed
#' @keywords internal
check_seed <- function(seed)
{
  if (is.null(seed) | !is.numeric(seed))
  {
    seed = runif(1) * .Machine$integer.max
  }else{
    seed = as.integer(seed)
  }
  
  storage.mode(seed) <- "integer"
  return(seed)
}

#' @title check_control
#' @name check_control
#' @keywords internal
check_control <- function(control, param)
{
  if (!is.list(control)) {
    stop("param.control must be a list")
  }
  
  # embedded model parameters
  
  # embed.ntrees
  if (is.null(control$embed.ntrees)) {
    embed.ntrees <- 100
  } else embed.ntrees = max(control$embed.ntrees, 1)
  storage.mode(embed.ntrees) <- "integer"
  
  # embed.mtry
  if (is.null(control$embed.mtry)) {
    embed.mtry <- 1/2
  } else embed.mtry = max(0, min(control$embed.mtry, param$p))
  storage.mode(embed.mtry) <- "double"
  
  # embed.nmin
  if (is.null(control$embed.nmin)) {
    embed.nmin <- 5
  } else embed.nmin = max(1, floor(control$embed.nmin))
  storage.mode(embed.nmin) <- "integer"
  
  # embed.split.gen
  if (is.null(control$embed.split.gen)) {
      embed.split.gen <- 1
  } else embed.split.gen = match(control$embed.split.gen, 
                                 c("random", "rank", "best"),
                                 nomatch = 0)
  storage.mode(embed.split.gen) <- "integer"
  
  # embed.nsplit
  if (is.null(control$embed.nsplit)) {
      embed.nsplit <- 1
  } else embed.nsplit = max(1, control$embed.nsplit)
  storage.mode(embed.nsplit) <- "integer"
  
  # embed.resample.replace
  if (is.null(control$embed.resample.replace)) {
    embed.resample.replace = 1
  }else embed.resample.replace = ifelse(control$embed.resample.replace!= 0, 1L, 0L)
  
  # embed.resample.prob
  if (is.null(control$embed.resample.prob)) {
      embed.resample.prob <- 0.9
  } else embed.resample.prob = max(0, min(control$embed.resample.prob, 0.99)) # must leave some oob samples
  storage.mode(embed.resample.prob) <- "double"
  
  # embed.mute
  if (is.null(control$embed.mute)) {
      embed.mute <- 0
  } else embed.mute = max(0, min(control$embed.mute, param$p))
  storage.mode(embed.mute) <- "double"
  
  # embed.protect
  if (is.null(control$embed.protect)) {
      embed.protect <- ceiling(2*log(param$n))
  } else embed.protect = max(0, min(control$embed.protect, param$p))
  storage.mode(embed.protect) <- "integer"

  # other parameters 
  
  # linear.comb
  if (is.null(control$linear.comb)) {
    linear.comb <- 1
  } else {
    if ( control$linear.comb > 5 )
      warning("very large linear.comb is not recommended")
    
    linear.comb = max(0, min(control$linear.comb, param$p))
  }
  storage.mode(linear.comb) <- "integer"  
  
  # split.rule will be checked in each model
  
  # resample.track
  if (is.null(control$resample.track)) {
    resample.track <- 0
  } else resample.track = ifelse(control$resample.track != 0, 1L, 0L)
  storage.mode(resample.track) <- "integer"
  
  # var.ready
  if (is.null(control$var.ready)) {
    var.ready <- 0
  } else var.ready = max(0, min(control$var.ready, 2))
  storage.mode(var.ready) <- "integer"

  # alpha
  if (is.null(control$alpha)) {
    alpha <- 0
  } else alpha = max(0, min(control$alpha, 0.5)) 
  storage.mode(alpha) <- "double"
  
  # return new control
  return(list(# embedded model control
              "embed.ntrees" = embed.ntrees,
              "embed.mtry" = embed.mtry,
              "embed.nmin" = embed.nmin,
              "embed.split.gen" = embed.split.gen,
              "embed.nsplit" = embed.nsplit,
              "embed.resample.replace" = embed.resample.replace,
              "embed.resample.prob" = embed.resample.prob,              
              "embed.mute" = embed.mute,
              "embed.protect" = embed.protect,
              # other parameters
              "linear.comb" = linear.comb,
              "split.rule" = control$split.rule,
              "resample.track" = resample.track,
              "var.ready" = var.ready,
              "alpha" = alpha))
}

#' @title check_resamplepreset
#' @name check_resamplepreset
#' @keywords internal
check_resamplepreset <- function(resample.preset, param, param.control)
{

  # for variance estimation
  if (param.control$var.ready)
  {
    # construct the matrix with matched sampling
    resample.preset = matrix(0, param$n, param$ntrees)
    k = as.integer(param$resample.prob*param$n)

    for (i in 1:as.integer(param$ntrees/2) )
    {
      ab = sample(1:param$n, 2*k)
      a = ab[1:k]
      b = ab[-(1:k)]
      
      resample.preset[a, i] = 1
      resample.preset[b, i+ (param$ntrees/2)] = 1
    }
  }else{
    
    # check resample.preset
    if (!is.matrix(resample.preset))
      stop("resample.preset must be a matrix")
    
    if (nrow(resample.preset) != param$n | ncol(resample.preset) != param$ntrees)
      stop("dimension of resample.preset does not match n x ntrees")
      
    if ( any(colSums(resample.preset*(resample.preset>0)) > param$n) )
      stop("column sums in resample.preset should not be larger than n")
    
  }
  
  storage.mode(resample.preset) <- "integer"
  return(resample.preset)
  
}