# Run process-aborting baseline cases outside the test runner. This also
# supports the audit's direct source-DLL loader before package installation.
native_fit_child <- function(code) {
  script <- tempfile(fileext = ".R")
  output <- tempfile(fileext = ".log")
  on.exit(unlink(c(script, output)), add = TRUE)
  dll <- getLoadedDLLs()[["RLT"]][["path"]]
  source_root <- file.path(dirname(dll), "..")
  libpaths <- paste(capture.output(dput(.libPaths())), collapse = "\n")
  init <- paste0(".libPaths(", libpaths, ")")
  if (file.exists(file.path(source_root, "R", "RLT.r"))) {
    dll <- encodeString(normalizePath(dll, winslash = "/"), quote = '"')
    rdir <- encodeString(normalizePath(file.path(source_root, "R"), winslash = "/"), quote = '"')
    init <- c(init, "loadNamespace('Rcpp')", paste0("dll <- dyn.load(", dll, ")"),
      "calls <- getDLLRegisteredRoutines(dll)[['.Call']]",
      "for (nm in names(calls)) assign(nm, calls[[nm]], envir=globalenv())",
      paste0("for (f in list.files(", rdir,
             ", full.names=TRUE, pattern='\\\\.[rR]$')) sys.source(f, globalenv())"))
  } else {
    init <- c(init, "library(RLT)")
  }
  writeLines(c(init, code), script)
  status <- suppressWarnings(system2(file.path(R.home("bin"), "Rscript"),
    c("--vanilla", shQuote(script)), stdout = output, stderr = output, timeout = 60))
  text <- if (file.exists(output)) readLines(output, warn = FALSE) else character()
  expect_identical(as.integer(status), 0L, info = paste(text, collapse = "\n"))
}

test_that("the original survival reinforcement crash is contained and corrected", {
  for (cores in 1:2) native_fit_child(c(
    "set.seed(176)",
    "n <- 100; p <- 5; x <- matrix(rnorm(n*p), n, p)",
    "y <- x[,1] + 2*x[,2] + rnorm(n, sd=.2)",
    "ys <- rexp(n, rate=exp(x[,1]*.4)); delta <- rbinom(n, 1, .7)",
    sprintf("fit <- RLT(x, ys, censor=delta, model='survival', ntrees=8, ncores=%d, seed=35, reinforcement=TRUE)", cores),
    "stopifnot(length(fit$FittedForest$NodeHaz)==8, all(is.finite(unlist(fit$FittedForest$NodeHaz))))"
  ))
})

test_that("entirely unavailable embedded survival screening terminates the node", {
  for (cores in 1:2) native_fit_child(c(
    "n <- 12; x <- cbind(seq_len(n),rev(seq_len(n)))",
    sprintf("cores <- %dL", cores),
    "fit <- RLT(x,as.numeric(seq_len(n)),censor=rep(1L,n),model='survival',ntrees=2,nmin=2,reinforcement=TRUE,resample.replace=FALSE,resample.prob=1,resample.preset=matrix(1L,n,2),param.control=list(embed.ntrees=2,embed.nmin=2,embed.resample.replace=FALSE,embed.resample.prob=.99),ncores=cores,seed=3)",
    "stopifnot(all(vapply(fit$FittedForest$SplitVar,function(v) length(v)==1L && v[1]==-1L,logical(1))),all(is.finite(unlist(fit$FittedForest$NodeHaz))))"
  ))
})

test_that("survival importance distinguishes unavailable comparisons from zero", {
  for (cores in 1:2) native_fit_child(c(
    "n <- 24; x <- cbind(seq_len(n), rep(0,n)); y <- as.numeric(seq_len(n))",
    "censor <- c(rep(0L,4), rep(1L,n-4))",
    "a <- rep(1L,n); a[c(5,6,15,16)] <- 0L",
    "b <- rep(1L,n); b[1:4] <- 0L",
    "root <- rep(0L,n); root[1:3] <- 1L",
    sprintf("cores <- %dL", cores),
    "for (lc in 1:2) for (imp in c('permute','distribute')) {",
    "  make <- function(track, nmin=3L) RLT(x,y,censor=censor,model='survival',ntrees=ncol(track),mtry=2,nmin=nmin,linear.comb=lc,importance=imp,resample.preset=track,ncores=cores,seed=17)",
    "  one <- rep(1L,n); one[1] <- 0L",
    "  missing <- make(cbind(b,one,rep(1L,n),b),nmin=30L)",
    "  stopifnot(all(is.na(missing$VarImp)))",
    "  defined <- make(matrix(a,n,4))",
    "  mixed <- make(cbind(a,b,root,rep(1L,n)))",
    "  plain <- suppressWarnings(RLT(x,y,censor=censor,model='survival',ntrees=4,mtry=2,nmin=3,linear.comb=lc,importance=FALSE,resample.preset=cbind(a,b,root,rep(1L,n)),ncores=cores,seed=17))",
    "  stopifnot(defined$VarImp[1]>0, defined$VarImp[2]==0)",
    # Distributed importance has no permutation Monte Carlo noise. The valid
    # root-only tree contributes zero and counts in the denominator.
    "  if (imp=='distribute') stopifnot(isTRUE(all.equal(mixed$VarImp,defined$VarImp/2,tolerance=1e-12)))",
    "  stopifnot(identical(mixed$FittedForest,plain$FittedForest), isTRUE(all.equal(mixed$Prediction,plain$Prediction,tolerance=1e-12)))",
    "  no_oob <- suppressWarnings(RLT(x,y,censor=censor,model='survival',ntrees=2,nmin=30,linear.comb=lc,importance=imp,resample.replace=FALSE,resample.prob=1,ncores=cores,seed=1))",
    "  stopifnot(all(is.na(no_oob$VarImp)))",
    "}",
    "for (lc in 1:2) {",
    "  make <- function(censor) suppressMessages(RLT(x,y,censor=censor,model='survival',ntrees=4,nmin=30,linear.comb=lc,var.mode='matched',ncores=cores,seed=1))",
    "  one_event <- make(c(1L,rep(0L,n-1)))",
    "  all_events <- make(rep(1L,n))",
    "  stopifnot(all(one_event$VarImp==0),all(is.na(one_event$VarVI)),all(all_events$VarImp==0),all(all_events$VarVI==0))",
    "}"
  ))
})

test_that("zero-mass outer and embedded samples produce catchable fit errors", {
  for (cores in 1:2) native_fit_child(c(
    "set.seed(940); n <- 12; x <- matrix(rnorm(n*3),n,3)",
    sprintf("cores <- %dL", cores),
    "for (model in c('regression','classification','survival')) for (lc in 1:2) {",
    "  y <- if(model=='classification') factor(rep(1:2,6)) else as.numeric(seq_len(n))",
    "  track <- matrix(1L,n,2); track[1,] <- 0L",
    "  args <- list(x=x,y=y,model=model,linear.comb=lc,ntrees=2,nmin=20,obs.w=c(1,rep(0,n-1)),resample.preset=track,ncores=cores,seed=1)",
    "  if(model=='survival') args$censor <- rep(1L,n)",
    "  err <- tryCatch(do.call(RLT,args),error=identity)",
    "  stopifnot(inherits(err,'error'),grepl('sampled tree has zero total observation weight',conditionMessage(err),fixed=TRUE))",
    "}",
    "for (model in c('regression','classification','survival')) for(lc in 1:2) {",
    "  if(model=='survival' && lc==2) next",
    "  y <- if(model=='classification') factor(rep(1:2,6)) else as.numeric(seq_len(n))",
    "  args <- list(x=x,y=y,model=model,linear.comb=lc,reinforcement=(lc==1),ntrees=2,nmin=1,obs.w=c(1,1,rep(0,n-2)),resample.preset=matrix(1L,n,2),ncores=cores,seed=1,param.control=list(embed.ntrees=50,embed.nmin=2,embed.resample.replace=FALSE,embed.resample.prob=.25))",
    "  if(model=='survival') args$censor <- rep(1L,n)",
    "  err <- tryCatch(do.call(RLT,args),error=identity)",
    "  stopifnot(inherits(err,'error'),grepl('sampled tree has zero total observation weight',conditionMessage(err),fixed=TRUE))",
    "}"
  ))
})

test_that("positive-mass nodes retain zero-weight records without undefined leaves", {
  for (cores in 1:2) native_fit_child(c(
    "n <- 12; x <- cbind(seq_len(n),rev(seq_len(n)))",
    sprintf("cores <- %dL", cores),
    "for (model in c('regression','classification','survival')) for(lc in 1:2) for(nsplit in c(0L,3L)) {",
    "  y <- if(model=='classification') factor(rep(1:2,6)) else as.numeric(seq_len(n))",
    "  args <- list(x=x,y=y,model=model,linear.comb=lc,ntrees=2,nmin=1,nsplit=nsplit,obs.w=c(1,1,rep(0,n-2)),resample.preset=matrix(1L,n,2),resample.replace=FALSE,resample.prob=1,ncores=cores,seed=2)",
    "  if(lc>1) args$linear.comb.method <- 'naive'",
    "  if(model=='survival') args$censor <- rep(1L,n)",
    "  fit <- do.call(RLT,args)",
    "  f <- fit$FittedForest",
    "  stopifnot(all(unlist(f$NodeWeight)>0),all(is.finite(unlist(f[c('NodeAve','NodeProb','NodeHaz')]))))",
    "}",
    "x <- data.frame(group=factor(rep(LETTERS[1:4],each=3)))",
    "for(model in c('regression','classification')) for(nsplit in c(0L,5L)) {",
    "  y <- if(model=='classification') factor(rep(1:2,each=6)) else as.numeric(seq_len(n))",
    "  fit <- RLT(x,y,model=model,ntrees=2,nmin=1,nsplit=nsplit,obs.w=c(rep(0,3),1,0,0,1,rep(0,5)),resample.preset=matrix(1L,n,2),resample.replace=FALSE,resample.prob=1,ncores=cores,seed=2)",
    "  f <- fit$FittedForest",
    "  stopifnot(all(unlist(f$NodeWeight)>0),all(is.finite(unlist(f[c('NodeAve','NodeProb')]))))",
    "}"
  ))
})

test_that("subnormal positive survival node weights do not become zero divisors", {
  for (cores in 1:2) native_fit_child(c(
    "n <- 12; x <- cbind(seq_len(n),rev(seq_len(n))); track <- matrix(1L,n,2); track[1,] <- 0L",
    sprintf("cores <- %dL", cores),
    "for(lc in 1:2) for(nmin in c(2L,20L)) for(rule in c('logrank','suplogrank','coxgrad')) {",
    "  w <- c(1,1e-323,rep(0,n-2))",
    "  stopifnot(sum(w[-1])>0, sum(w[-1])/(n-1)==0)",
    "  args <- list(x=x,y=as.numeric(rep(1:6,each=2)),censor=rep(1L,n),model='survival',ntrees=2,nmin=nmin,linear.comb=lc,split.rule=rule,obs.w=w,resample.preset=track,ncores=cores,seed=4)",
    "  if(lc>1) args$linear.comb.method <- 'naive'",
    "  fit <- do.call(RLT,args)",
    "  stopifnot(all(unlist(fit$FittedForest$NodeWeight)>0),all(is.finite(unlist(fit$FittedForest$NodeHaz))))",
    "}"
  ))
})

test_that("undefined concordance does not create artificial finite evidence", {
  expect_true(is.na(cindex(1, 1L, 0)))
  expect_true(is.na(cindex(c(1,2), c(0L,0L), c(1,2))))
  expect_true(is.na(cindex(c(1,2), c(1L,1L), c(NA_real_,0))))
  expect_equal(cindex(c(1,2), c(1L,1L), c(2,1)), 1)
  expect_equal(cindex(c(1,1), c(1L,1L), c(2,2)), 1)
})

test_that("partial weighted candidate sampling terminates without changing the full pool", {
  for (cores in 1:2) native_fit_child(c(
    "set.seed(183); n <- 64L; x <- matrix(rnorm(n*5),n,5)",
    sprintf("cores <- %dL", cores),
    "for(model in c('regression','classification','survival')) for(lc in if(model=='survival') 1:2 else 1L) {",
    "  y <- switch(model,regression=x[,1]+x[,2],classification=factor(x[,1]>0),survival=as.numeric(seq_len(n)))",
    "  args <- list(x=x,y=y,model=model,linear.comb=lc,ntrees=2L,nmin=63L,nsplit=0L,resample.replace=FALSE,resample.prob=1,ncores=cores,seed=7L,param.control=list(embed.nmin=100L))",
    "  if(lc>1L) args$linear.comb.method <- 'naive'",
    "  if(model=='survival') args$censor <- rep(1L,n)",
    "  args$mtry <- 4L; args$var.prob <- c(1,0,0,0,0)",
    "  err <- tryCatch(do.call(RLT,args),error=identity)",
    "  stopifnot(inherits(err,'error'),grepl('fewer positive weights',conditionMessage(err),fixed=TRUE))",
    "  args$mtry <- 2L; args$var.prob <- c(1,1e-300,1e-300,0,0)",
    "  err <- tryCatch(do.call(RLT,args),error=identity)",
    "  stopifnot(inherits(err,'error'),grepl('remaining weight',conditionMessage(err),fixed=TRUE))",
    "  args$mtry <- 4L; args$var.prob <- c(.1,.2,.3,.4,0)",
    "  fit <- do.call(RLT,args)",
    "  stopifnot(all(is.finite(unlist(fit$FittedForest$NodeWeight))))",
    "  args$mtry <- 5L; args$var.prob <- c(1,0,0,0,0)",
    "  full <- do.call(RLT,args); args$var.prob <- NULL; plain <- do.call(RLT,args)",
    "  stopifnot(identical(full$FittedForest,plain$FittedForest))",
    "}"
  ))
})

test_that("weighted SIR skips empty response slices and retains a real direction", {
  for (cores in 1:2) native_fit_child(c(
    "set.seed(5); n <- 64L; x <- matrix(rnorm(n*4),n,4); y <- as.numeric(0:63)",
    "x[,1] <- y/64 + .1*x[,1]",
    sprintf("cores <- %dL", cores),
    "fit <- RLT(x,y,ntrees=2L,mtry=4L,nmin=63L,nsplit=0L,linear.comb=2L,linear.comb.method='sir',obs.w=c(rep(0,16),rep(1,48)),resample.replace=FALSE,resample.prob=1,ncores=cores,seed=7L,param.control=list(embed.nmin=100L))",
    "for(load in fit$FittedForest$SplitLoad) {",
    "  stopifnot(sum(abs(load[1,])>1e-12)==2L,all(is.finite(load)),abs(sum(load[1,]^2)-1)<1e-12,!isTRUE(all.equal(abs(load[1,]),c(1,1),tolerance=1e-8)))",
    "}",
    "stopifnot(all(is.finite(unlist(fit$FittedForest$NodeAve))))",
    "one <- RLT(x,y,ntrees=2L,mtry=4L,nmin=63L,nsplit=0L,linear.comb=2L,linear.comb.method='sir',obs.w=c(rep(0,56),rep(1,8)),resample.replace=FALSE,resample.prob=1,ncores=cores,seed=7L,param.control=list(embed.nmin=100L))",
    "for(tree in seq_along(one$FittedForest$SplitLoad)) {",
    "  vars <- one$FittedForest$SplitVar[[tree]][1,]+1L; load <- one$FittedForest$SplitLoad[[tree]][1,]",
    "  stopifnot(identical(as.vector(load),as.vector(sign(cor(x[,vars,drop=FALSE],y)))),sum(load!=0)==2L)",
    "}"
  ))
})

test_that("unweighted LC suplogrank uses finite variance at screening and cutpoints", {
  for (cores in 1:2) native_fit_child(c(
    "set.seed(45682); n <- 64L; x <- matrix(rnorm(n*5),n,5)",
    "signal <- 1.5*x[,1]-.7*x[,2]+.25*x[,3]*x[,4]",
    "time <- pmax(1,round(rexp(n,rate=exp(.4*signal))*4)); event <- as.integer(seq_len(n)%%5!=1)",
    sprintf("cores <- %dL", cores),
    "score <- function(u,cut) {",
    "  times <- sort(unique(time[event==1L])); left <- u<=cut",
    "  risk <- vapply(times,function(t) sum(time>=t),numeric(1)); fail <- vapply(times,function(t) sum(time==t & event==1L),numeric(1))",
    "  lr <- vapply(times,function(t) sum(left & time>=t),numeric(1)); lf <- vapply(times,function(t) sum(left & time==t & event==1L),numeric(1)); rr <- risk-lr",
    "  w <- lr*rr/risk; delta <- rep(0,length(times)); ok <- lr>0 & rr>0; delta[ok] <- lf[ok]/lr[ok]-(fail[ok]-lf[ok])/rr[ok]",
    "  variance <- rep(0,length(times)); ok <- risk>1; variance[ok] <- (risk[ok]-fail[ok])/(risk[ok]-1)*fail[ok]/risk[ok]",
    "  denominator <- sum(variance*w); if(denominator>0) max(cumsum(w*delta)^2)/denominator else -1",
    "}",
    "for(nsplit in c(0L,3L)) {",
    "  fit <- RLT(x,time,censor=event,model='survival',ntrees=2L,mtry=5L,nmin=63L,nsplit=nsplit,alpha=.1,linear.comb=2L,linear.comb.method='naive',split.rule='suplogrank',resample.replace=FALSE,resample.prob=1,ncores=cores,seed=7L)",
    "  f <- fit$FittedForest",
    "  for(tree in seq_along(f$SplitLoad)) {",
    "    load <- f$SplitLoad[[tree]][1,]; stopifnot(sum(abs(load)>1e-12)==2L,all(is.finite(load)))",
    "    u <- as.vector(x[,f$SplitVar[[tree]][1,]+1L,drop=FALSE] %*% load); chosen <- score(u,f$SplitValue[[tree]][1])",
    "    stopifnot(is.finite(chosen),chosen>0)",
    "    if(nsplit==0L) { ranked <- sort(u); k <- 6:58; cuts <- (ranked[k]+ranked[k+1L])/2; stopifnot(abs(chosen-max(vapply(cuts,function(cut) score(u,cut),numeric(1))))<1e-10) }",
    "  }",
    "  stopifnot(all(is.finite(unlist(f$NodeHaz))))",
    "}",
    "fit <- RLT(x,rep(1,n),censor=rep(1L,n),model='survival',ntrees=2L,mtry=5L,nmin=63L,linear.comb=2L,split.rule='suplogrank',resample.replace=FALSE,resample.prob=1,ncores=cores,seed=7L)",
    "stopifnot(all(vapply(fit$FittedForest$SplitVar,function(v) v[1,1]==-1L,logical(1))),all(is.finite(unlist(fit$FittedForest$NodeHaz))))"
  ))
})
