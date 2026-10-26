check_ntrees <- function(ntrees)
{
  storage.mode(ntrees) <- "integer"
  
  if (is.na(ntrees))
    stop("ntrees should be numerical")
    
  if (ntrees < 1)
    stop("ntrees should be greater than 0")

  return(ntrees)
}

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

check_nmin <- function(nmin)
{
  storage.mode(nmin) <- "integer"
  
  if (is.na(nmin))
    stop("nmin should be numerical")
  
  if (nmin < 1)
    stop("nmin cannot be less than 1")

  return(nmin)
}


check_nsplit <- function(nsplit)
{
  storage.mode(nsplit) <- "integer"
  
  if (is.na(nsplit))
    stop("nsplit should be numerical")
  
  if (nsplit < 0)
    stop("nsplit cannot be less than 0")
  
  return(nsplit)
}

check_resamplereplace <- function(resample.replace)
{
  storage.mode(resample.replace) <- "integer"
  
  if (is.na(resample.replace))
    stop("resample.replace should be logical")
  
  resample.replace = ifelse(resample.replace != 0, 1L, 0L)
  return(resample.replace)
}

check_resampleprob <- function(resample.prob)
{
  storage.mode(resample.prob) <- "double"  
  
  if (is.na(resample.prob))
    stop("resample.prob should be numerical")
  
  if (resample.prob <= 0 | resample.prob > 1)
    stop("resample.prob should be within the interval (0, 1]")
  
  return(resample.prob)
}

check_obsw <- function(obs.w, n)
{
  obs.w = as.numeric(as.vector(obs.w))
  
  if (any(is.na(obs.w)))
    stop("observation weights (obs.w) should be numerical")

  if (any(!is.finite(obs.w)))
    stop("observation weights (obs.w) must be finite")
  
  if (any(obs.w < 0))
    stop("observation weights (obs.w) cannot be negative")
  
  if (length(obs.w) != n)
    stop("length of observation weights (obs.w) must be n")

  if (!length(obs.w) || !any(obs.w > 0))
    stop("observation weights (obs.w) must have positive total mass")
  
  storage.mode(obs.w) <- "double"    
  obs.w = obs.w/max(obs.w)
  obs.w = obs.w/sum(obs.w)
  
  return(obs.w)
}

check_varprob <- function(var.prob, p)
{
  var.prob = as.numeric(as.vector(var.prob))

  if (any(is.na(var.prob)))
    stop("variable probabilities (var.prob) should be numerical")

  if (any(!is.finite(var.prob)))
    stop("variable probabilities (var.prob) must be finite")
  
  if (any(var.prob < 0))
    stop("variable probabilities (var.prob) cannot be negative")
  
  if (length(var.prob) != p)
    stop("length of variable probabilities (var.prob) must be p")

  if (!length(var.prob) || !any(var.prob > 0))
    stop("variable probabilities (var.prob) must have positive total mass")
  
  storage.mode(var.prob) <- "double"
  var.prob = var.prob/max(var.prob)
  var.prob = var.prob/sum(var.prob)
  return(var.prob)
}

check_importance <- function(importance)
{
  importance.num = -1
  
  if (  importance == 0 )
    importance.num = 0
  
  if (  match(importance, c(TRUE, "permute"), nomatch = 0) )
    importance.num = 1
  
  if (  match(importance, c("distribute"), nomatch = 0) )
    importance.num = 2
  
  if (importance.num == -1)
    stop("variable importance mode is not recognized")
    
  storage.mode(importance.num) <- "integer"
  return(importance.num)
}

check_importance_settings <- function(importance, n, resample.replace, resample.prob)
{
  if (importance == 0) return(invisible(NULL))
  
  # expected OOB size
  oob_size <- n - as.integer(n * resample.prob)
  
  if (resample.replace)
  {
    # bootstrap always has OOB; just check n is not tiny
    if (n < 3)
      warning(sprintf(
        "n = %d is very small; importance estimates may be unreliable.",
        n), call. = FALSE)
    return(invisible(NULL))
  }
  
  # no replacement: OOB size is deterministic
  if (importance == 1 && oob_size < 2)
  {
    warning(sprintf(
      "Permutation importance requires at least 2 OOB observations per tree, but current settings yield only %d (n = %d, resample.prob = %.4f, resample.replace = FALSE). Importance calculation will be skipped. Consider increasing OOB by lowering resample.prob or using resample.replace = TRUE.",
      oob_size, n, resample.prob), call. = FALSE)
  }
  
  if (importance == 2 && oob_size < 1)
  {
    warning(sprintf(
      "Distributed importance requires at least 1 OOB observation per tree, but current settings yield 0 (n = %d, resample.prob = %.4f, resample.replace = FALSE). Importance calculation will be skipped. Consider lowering resample.prob.",
      n, resample.prob), call. = FALSE)
  }
  
  if (importance == 2 && oob_size >= 1 && oob_size < 5)
  {
    warning(sprintf(
      "Distributed importance with only %d OOB observation(s) per tree (n = %d, resample.prob = %.4f) may produce noisy estimates. Consider using more OOB samples (e.g., resample.prob = 0.632 with resample.replace = FALSE).",
      oob_size, n, resample.prob), call. = FALSE)
  }
  
  invisible(NULL)
}


check_reinforcement <- function(reinforcement)
{
  storage.mode(reinforcement) <- "integer"
  
  if (is.na(reinforcement))
    stop("reinforcement should be logical")
  
  reinforcement = ifelse(reinforcement != 0, 1L, 0L)
  
  return(reinforcement)
}

check_ncores <- function(ncores)
{
  storage.mode(ncores) <- "integer"
  
  if (is.na(ncores))
    stop("ncores should be numerical")
  
  if (ncores < 0)
    stop("ncores cannot be less than 0")
  
  return(ncores)
}

check_verbose <- function(verbose)
{
  storage.mode(verbose) <- "integer"
  
  if (is.na(verbose))
    stop("verbose should be numerical")
  
  return(verbose)
}

check_seed <- function(seed)
{
  if (is.null(seed) | !is.numeric(seed))
  {
    seed = stats::runif(1) * .Machine$integer.max
  }else{
    seed = as.integer(seed)
  }
  
  storage.mode(seed) <- "integer"
  return(seed)
}

check_control <- function(control, param)
{
  if (!is.list(control)) {
    stop("param.control must be a list")
  }
  
  # embedded model parameters
  
  # embed.ntrees
  if (is.null(control$embed.ntrees)) {
    embed.ntrees <- 50
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
  
  # embed.nsplit (determines split strategy: 0=best, >0=random)
  if (is.null(control$embed.nsplit)) {
      embed.nsplit <- 3
  } else embed.nsplit = max(0, floor(control$embed.nsplit))
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
      embed.protect <- ceiling(log(param$n))
  } else embed.protect = max(0, min(control$embed.protect, param$p))
  storage.mode(embed.protect) <- "integer"

  # embed.threshold
  if (is.null(control$embed.threshold)) {
    embed.threshold <- 0.25
  } else embed.threshold = max(0, min(control$embed.threshold, 1))
  storage.mode(embed.threshold) <- "double"
  
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
  
  # linear.comb.method will be checked in each model
  
  # split.rule will be checked in each model (for classification/survival)
  
  # resample.track
  if (is.null(control$resample.track)) {
    resample.track <- 0
  } else resample.track = ifelse(control$resample.track != 0, 1L, 0L)
  storage.mode(resample.track) <- "integer"
  
  # var.mode
  if (is.null(control$var.mode)) {
    var.mode <- "none"
  } else if (is.logical(control$var.mode)) {
    var.mode <- ifelse(control$var.mode, "matched", "none")
  } else {
    var.mode <- match.arg(control$var.mode, c("none", "matched", "IJ", "jack"))
  }
  
  # return new control
  return(list(# embedded model control
              "embed.ntrees" = embed.ntrees,
              "embed.mtry" = embed.mtry,
              "embed.nmin" = embed.nmin,
              "embed.nsplit" = embed.nsplit,
              "embed.resample.replace" = embed.resample.replace,
              "embed.resample.prob" = embed.resample.prob,              
              "embed.mute" = embed.mute,
              "embed.protect" = embed.protect,
              "embed.threshold" = embed.threshold,
              # other parameters
              "linear.comb" = linear.comb,
              "linear.comb.method" = control$linear.comb.method,
              "split.rule" = control$split.rule,
              "resample.track" = resample.track,
              "var.mode" = var.mode))
}

check_resamplepreset <- function(resample.preset, param, param.control)
{

  # for variance estimation
  if (param.control$var.mode != "none")
  {
    k = as.integer(param$resample.prob*param$n)
    resample.preset = gen_ms_obs_track_mat_cpp(param$n, k, param$ntrees, param$seed + 1)
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
