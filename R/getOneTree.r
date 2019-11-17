#' @title Print a single tree 
#' @description Print a single tree 
#' @param x A fitted RLT object
#' @param tree the tree number
#' @param ... ...
#' @examples

getOneTree<- function(x, tree = 1, ...)
{
  
  if ( all(class(x)[2:3] == c("fit", "reg")) )
  {
    if (tree > length(x$FittedForest$NodeType) || tree < 1)
      stop(paste("There is no tree", tree, "in the fitted forest"))

    cat(paste("Tree #", tree, " in the fitted regression forest: \n\n", sep = ""))
    
    if (is.null(x$xnames))
      newnames = paste("var ", 1:x$parameters$p)
    else
      newnames = x$xnames
    
    newnames = gsub("\\s", " ", format(newnames, width=max(nchar(newnames))))
    
    OneTree = data.frame( "NodeType" = x$FittedForest$NodeType[[tree]],
                          "SplitVar" = newnames[x$FittedForest$SplitVar[[tree]] + 1],
                          "SplitValue" = x$FittedForest$SplitValue[[tree]],
                          "LeftNode" = x$FittedForest$LeftNode[[tree]] + 1,
                          "RightNode" = x$FittedForest$RightNode[[tree]] + 1,
                          "NodeSize" = x$FittedForest$NodeSize[[tree]],
                          "NodeAve" = x$FittedForest$NodeAve[[tree]])
    
    # OneTree[OneTree$NodeType == 2, ] = NA
    OneTree[OneTree$NodeType == 3, c(2,3,4,5)] = NA

    return(OneTree)
  }
  
  if ( all(class(x)[2:3] == c("fit", "surv")) )
  {
    if (tree > length(x$FittedForest$NodeType) || tree < 1)
      stop(paste("There is no tree", tree, "in the fitted forest"))
    
    cat(paste("Tree #", tree, " in the fitted regression forest: \n\n", sep = ""))
    
    
    if (is.null(x$xnames)){
      newnames = paste("var", 1:x$parameters$p)
    }else{
      newnames = x$xnames
    }
    
    newnames = gsub("\\s", " ", format(newnames, width=max(nchar(newnames))))
    
    OneTree = data.frame( "NodeType" = x$FittedForest$NodeType[[tree]],
                          "SplitVar" = newnames[x$FittedForest$SplitVar[[tree]] + 1],
                          "SplitValue" = x$FittedForest$SplitValue[[tree]],
                          "LeftNode" = x$FittedForest$LeftNode[[tree]] + 1,
                          "RightNode" = x$FittedForest$RightNode[[tree]] + 1,
                          "NodeSize" = x$FittedForest$NodeSize[[tree]])
    
    OneTree[OneTree$NodeType == 3, c(2,3,4,5)] = NA
    
    return(OneTree)
  }
  
}
