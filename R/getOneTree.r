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
    
    OneTree = data.frame( "NodeType" = x$FittedForest$NodeType[[tree]],
                          "SplitVar" = x$FittedForest$SplitVar[[tree]] + 1,
                          "SplitValue" = x$FittedForest$SplitValue[[tree]],
                          "LeftNode" = x$FittedForest$LeftNode[[tree]] + 1,
                          "RightNode" = x$FittedForest$RightNode[[tree]] + 1,
                          "NodeAve" = x$FittedForest$NodeAve[[tree]],
                          "NodeSize" = x$FittedForest$NodeSize[[tree]])
    
    OneTree[OneTree$NodeType == 2, 6] = NA
    OneTree[OneTree$NodeType == 3, c(2,3,4,5)] = NA

    return(OneTree)
  }
}
