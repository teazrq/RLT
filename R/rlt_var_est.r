#' @title                 Regression random forest with variance estimation
#' @description           Fit random forests with variance estimation at the 
#'                        given testing point. Only sampling without
#'                        replacement is available. 
#'                        
#' @param x               A `matrix` or `data.frame` of features
#' 
#' @param y               Response variable. a `numeric`/`factor` vector.
#'                        
#' @param testx           A `matrix` or `data.frame` of testing data
#'                        
#' @param ntrees          Number of trees, `ntrees = 100` if reinforcement is
#'                        used and `ntrees = 1000` otherwise.
#'                        
#' @param mtry            Number of randomly selected variables used at each 
#'                        internal node.
#'                        
#' @param nmin            Terminal node size. Splitting will stop when the 
#'                        internal node size is less than twice of `nmin`. This
#'                        is equivalent to setting `nodesize` = 2*`nmin` in the
#'                        `randomForest` package.
#'                        
#' @param alpha           Minimum number of observations required for each 
#'                        child node as a portion of the parent node. Must be 
#'                        within `[0, 0.5)`. When `alpha` $> 0$ and `split.gen`
#'                        is `rank` or `best`, this will force each child node 
#'                        to contain at least \eqn{\max(\texttt{nmin}, \alpha \times N_A)}
#'                        number of number of observations, where $N_A$ is the 
#'                        sample size at the current internal node. This is 
#'                        mainly for theoritical concern.  
#'                                              
#' @param k               Subsampling size.
#'                        
#' @param split.gen       How the cutting points are generated: `"random"`, 
#'                        `"rank"` or `"best"`. `"random"` performs random 
#'                        cutting point and does not take `alpha` into 
#'                        consideration. `"rank"` could be more effective when 
#'                        there are a large number of ties. It can also be used 
#'                        to guarantee child node size if `alpha` > 0. `"best"` 
#'                        finds the best cutting point, and can be cominbed with 
#'                        `alpha` too.
#' 
#' @param nsplit          Number of random cutting points to compare for each 
#'                        variable at an internal node.
#'                        
#' @param resample.prob   Proportion of in-bag samples.
#'                        
#' @param seed            Random seed using the `Xoshiro256+` generator.
#' 
#' @param ncores          Number of cores. Default is 1.
#' 
#' @param verbose         Whether fitting should be printed.
#' 
#' @param ...             Additional arguments.
#' 
#' @export
#' 
#' @return 
#' 
#' Prediction and variance estimation

rlt_var_est <- function(x, y, testx,
              			    ntrees = if (reinforcement) 100 else 500,
              			    mtry = max(1, as.integer(ncol(x)/3)),
              			    nmin = max(1, as.integer(log(nrow(x)))),
              			    alpha = 0,
              			    k = nrow(x) / 2, 
              			    split.gen = "random",
              			    nsplit = 1,
              			    seed = NaN,
              			    ncores = 1,
              			    verbose = 0,
              			    ...)
{
    if (!is.matrix(testx) & !is.data.frame(testx)) stop("testx must be a matrix or a data.frame")
    if (any(is.na(testx))) stop("NA not permitted in testx")
  
    RLT.fit = RLT(x, y, ntrees = ntrees*10, mtry = mtry, nmin = nmin, alpha = alpha,
                  split.gen = split.gen, replacement = TRUE, resample.prob = k/n, 
                  ncores = ncores)
    
    RLT.pred = predict(RLT.fit, testx, ncores = ncores, keep.all = TRUE)
    
    tree.var = apply(RLT.pred$PredictionAll, 1, var)
    
    # count how many pairs of trees match C
    
    RLT.fit = RLT(x, y, ntrees = ntrees, mtry = mtry, nmin = nmin, alpha = alpha,
                  split.gen = "best", replacement = FALSE, resample.prob = k/n,
                  ncores = ncores, track.obs = TRUE)
    
    RLT.pred = predict(RLT.fit, testx, ncores = ncores, keep.all = TRUE)
    
    
    C_min = qhyper(0.01, k, n - k, k)
    C_max = qhyper(0.99, k, n - k, k)
    
    C = seq(C_min, C_max)
    storage.mode(C) <- "integer"
    
    two.sample.var = EofVar(RLT.fit$ObsTrack, RLT.pred$PredictionAll, C, ncores, verbose)
    
    
    return(list("pred" = RLT.pred$Prediction,
                "tree.var" = tree.var,
                "allc" = two.sample.var$allcounts,
                "estimation" = two.sample.var$estimation,
                "var" = tree.var - rowSums(sweep(two.sample.var$estimation, 2, two.sample.var$allc, FUN = "*"))/sum(two.sample.var$allc)))

}
