#' @title Print a single tree 
#' @description Print a single fitted tree from a forest object
#' @param x A fitted RLT object
#' @param tree the tree number, starting from 1 to \code{ntrees}.
#' @param ... ...
#' @return A data.frame with columns: Node (depth, BFS), NodeType (Split=1, Leaf=-1), SplitVar, SplitValue, LeftNode, RightNode, N (sample count). Model-specific columns include YAvg (regression), Prob (classification), or Hazard/SurvProb (survival).
#' @export
#' @examples
#' \donttest{
#'   set.seed(42)
#'   x <- matrix(rnorm(300 * 5), ncol = 5)
#'   y <- rowSums(x[, 1:2]) + rnorm(300)
#'   fit <- RLT(x, y, ntrees = 50)
#'   get.one.tree(fit, tree = 1)
#' }

get.one.tree <- function(x, tree = 1, ...)
{
  if (!inherits(x, "RLT") || !inherits(x, "fit"))
    stop("Not an RLT fitted forest.")
  
  if (tree > length(x$FittedForest$SplitVar) || tree < 1)
    stop(paste("There is no tree", tree, "in the fitted forest"))
  
  # Variable names
  if (is.null(x$xnames))
  {
    vnames = paste0("V", 1:x$parameters$p)
  }else{
    vnames = x$xnames
  }
  
  # Append (F) for categorical variables
  vnames = paste0(vnames, ifelse(x$ncat > 1, "(F)", ""))
  
  model_type <- class(x)[[3]]
  ncomb <- x$parameters$linear.comb
  is_lc <- (ncomb > 1)
  
  # Extract tree structure (0-indexed from C++)
  LeftNode  <- x$FittedForest$LeftNode[[tree]]
  RightNode <- x$FittedForest$RightNode[[tree]]
  NodeWeight <- x$FittedForest$NodeWeight[[tree]]
  n_nodes <- length(NodeWeight)
  
  # Compute depth for each node via BFS
  # Note: terminal nodes have LeftNode/RightNode = 0 (not -1),
  # so we must skip them using the terminal flag
  depth <- integer(n_nodes)
  depth[1] <- 0
  
  # Need terminal flag early for BFS
  if (is_lc)
  {
    terminal_bfs <- (x$FittedForest$SplitVar[[tree]][, 1] == -1)
  }else{
    terminal_bfs <- (x$FittedForest$SplitVar[[tree]] == -1)
  }
  
  if (n_nodes > 1)
  {
    queue <- 1
    while (length(queue) > 0)
    {
      node <- queue[1]
      queue <- queue[-1]
      
      if (!terminal_bfs[node])
      {
        lnk <- LeftNode[node] + 1
        rnk <- RightNode[node] + 1
        
        if (lnk >= 1 && lnk <= n_nodes)
        {
          depth[lnk] <- depth[node] + 1
          queue <- c(queue, lnk)
        }
        if (rnk >= 1 && rnk <= n_nodes)
        {
          depth[rnk] <- depth[node] + 1
          queue <- c(queue, rnk)
        }
      }
    }
  }
  
  # Terminal nodes (already computed above as terminal_bfs)
  terminal <- terminal_bfs
  
  # Build Split column
  SplitStr <- character(n_nodes)
  
  if (is_lc)
  {
    SplitLoad <- x$FittedForest$SplitLoad[[tree]]
    SplitVar  <- x$FittedForest$SplitVar[[tree]]
    
    for (i in 1:n_nodes)
    {
      if (terminal[i])
      {
        SplitStr[i] <- "*"
        next
      }
      
      loads <- SplitLoad[i, ]
      vars  <- SplitVar[i, ]
      nonzero <- (loads != 0)
      
      if (sum(nonzero) == 0)
      {
        SplitStr[i] <- "*"
        next
      }
      
      active_vars <- vars[nonzero]
      active_loads <- loads[nonzero]
      n_active <- sum(nonzero)
      
      if (n_active <= 3)
      {
        # Show full linear combination with coefficients
        terms <- character(n_active)
        for (k in seq_len(n_active))
        {
          coef <- active_loads[k]
          vname <- vnames[active_vars[k] + 1]
          
          if (k == 1)
          {
            if (abs(abs(coef) - 1) < 1e-10)
            {
              terms[k] <- if (coef >= 0) vname else paste0("-", vname)
            } else {
              terms[k] <- paste0(sprintf("%.3f", abs(coef)), "\u00b7", vname)
              if (coef < 0) terms[k] <- paste0("-", terms[k])
            }
          } else {
            if (abs(abs(coef) - 1) < 1e-10)
            {
              terms[k] <- paste0(if (coef >= 0) " + " else " - ", vname)
            } else {
              terms[k] <- paste0(if (coef >= 0) " + " else " - ",
                                  sprintf("%.3f", abs(coef)), "\u00b7", vname)
            }
          }
        }
        SplitStr[i] <- paste0(terms, collapse = "")
      } else {
        # >3 active variables: compact variable names only
        SplitStr[i] <- paste(vnames[active_vars + 1], collapse = ", ")
      }
    }
  }else{
    SplitVar <- x$FittedForest$SplitVar[[tree]]
    for (i in 1:n_nodes)
    {
      if (terminal[i])
      {
        SplitStr[i] <- "*"
      }else{
        SplitStr[i] <- vnames[SplitVar[i] + 1]
      }
    }
  }
  
  # Build Value column (NA for leaves; show threshold for LC too)
  SplitValue <- x$FittedForest$SplitValue[[tree]]
  ValueStr <- sprintf("%.4f", SplitValue)
  ValueStr[terminal] <- NA_character_
  
  # Build output columns
  NodeID <- 1:n_nodes
  Depth  <- depth
  n      <- NodeWeight
  
  # Print header
  model_label <- switch(model_type,
                        "reg"  = "Regression",
                        "cla"  = "Classification",
                        "surv" = "Survival",
                        model_type)
  
  if (is_lc)
    cat(sprintf("Tree #%d  [%s, Linear Combination]\n\n", tree, model_label))
  else
    cat(sprintf("Tree #%d  [%s]\n\n", tree, model_label))
  
  # Print based on model type
  if (model_type == "reg")
  {
    NodeAve <- x$FittedForest$NodeAve[[tree]]
    
    cat(sprintf("%-5s %5s  %-30s  %10s  %5s  %10s\n",
                "Node", "Depth", "Split", "Value", "n", "NodeAve"))
    cat(strrep("-", 78), "\n")
    
    for (i in 1:n_nodes)
    {
      val_str <- if (is.na(ValueStr[i])) "     -" else ValueStr[i]
      
      cat(sprintf("%5d %5d  %-30s  %10s  %5d  %10.4f\n",
                  NodeID[i], Depth[i], SplitStr[i], val_str, n[i], NodeAve[i]))
    }
  }else if (model_type == "cla"){
    
    probmat <- x$FittedForest$NodeProb[[tree]]
    K <- ncol(probmat)
    
    # Format class probs as {0.50, 0.50}
    prob_strs <- character(n_nodes)
    for (i in 1:n_nodes)
    {
      probs <- sprintf("%.2f", probmat[i, ])
      prob_strs[i] <- paste0("{", paste(probs, collapse = ", "), "}")
    }
    
    cat(sprintf("%-5s %5s  %-30s  %10s  %5s  %s\n",
                "Node", "Depth", "Split", "Value", "n", "ClassProbs"))
    cat(strrep("-", 78), "\n")
    
    for (i in 1:n_nodes)
    {
      val_str <- if (is.na(ValueStr[i])) "     -" else ValueStr[i]
      
      cat(sprintf("%5d %5d  %-30s  %10s  %5d  %s\n",
                  NodeID[i], Depth[i], SplitStr[i], val_str, n[i], prob_strs[i]))
    }
  }else if (model_type == "surv"){
    
    cat(sprintf("%-5s %5s  %-30s  %10s  %5s\n",
                "Node", "Depth", "Split", "Value", "n"))
    cat(strrep("-", 62), "\n")
    
    for (i in 1:n_nodes)
    {
      val_str <- if (is.na(ValueStr[i])) "     -" else ValueStr[i]
      
      cat(sprintf("%5d %5d  %-30s  %10s  %5d\n",
                  NodeID[i], Depth[i], SplitStr[i], val_str, n[i]))
    }
  }
  
  invisible(data.frame(
    Node = NodeID,
    Depth = Depth,
    Split = SplitStr,
    Value = SplitValue,
    n = n,
    stringsAsFactors = FALSE
  ))
}
