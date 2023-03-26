#' @title Print a single tree 
#' @description Print a single fitted tree from a forest object
#' @param x A fitted RLT object
#' @param tree the tree number, starting from 1 to \code{ntrees}.
#' @param ... ...
#' @export

get.one.tree <- function(x, tree = 1, ...)
{
  if (any(class(x)[1:2] != c("RLT", "fit")))
    stop("Not an RLT fitted forest.")
  
  if (tree > length(x$FittedForest$SplitVar) || tree < 1)
    stop(paste("There is no tree", tree, "in the fitted forest"))
  
  if (is.null(x$xnames))
  {
    newnames = paste("var ", 1:x$parameters$p)
  }else
    newnames = x$xnames
  
  # newnames = gsub("\\s", " ", format(newnames, width=max(nchar(newnames))))
  newnames = paste(newnames, ifelse(x$ncat > 1, "(F)", "(C)"))
  newnames = c(newnames, " ")
  
  p = x$parameters$p
  
  SplitVar = x$FittedForest$SplitVar[[tree]]
  SplitValue = x$FittedForest$SplitValue[[tree]]
  LeftNode = x$FittedForest$LeftNode[[tree]] + 1
  RightNode = x$FittedForest$RightNode[[tree]] + 1
  NodeWeight = x$FittedForest$NodeWeight[[tree]]
  
  terminal = (SplitVar == -1)

  # correct other columns
  SplitVar[terminal] = NA  
  SplitValue[terminal] = NA
  LeftNode[terminal] = NA
  RightNode[terminal] = NA
  
  # old code when node size is not saved
  # SplitVar[terminal] = p  
  # # the node size is saved at LeftNode
  # NodeSize = LeftNode - 1
  # NodeSize[!terminal] = NA
  # 
  # # correct other columns
  # SplitValue[terminal] = NA
  # LeftNode[terminal] = NA
  # RightNode[terminal] = NA
  # 
  # while (sum(is.na(NodeSize)) > 0)
  # {
  #   uppernodes = (LeftNode %in% which(!is.na(NodeSize))) & 
  #                (RightNode %in% which(!is.na(NodeSize))) & 
  #                is.na(NodeSize)
  # 
  #   NodeSize[uppernodes] = NodeSize[LeftNode[uppernodes]] + 
  #                          NodeSize[RightNode[uppernodes]]
  # }  
  
  if ( class(x)[3] == "reg" )
  {
    cat(paste("Tree #", tree, " in the fitted regression forest: \n\n", sep = ""))

    OneTree = data.frame("SplitVar (Type)" = newnames[SplitVar + 1],
                         "SplitValue" = SplitValue,
                         "LeftNode" = LeftNode,
                         "RightNode" = RightNode,
                         "NodeWeight" = NodeWeight,
                         "NodeAve" = x$FittedForest$NodeAve[[tree]])
  }
  
  
  if ( class(x)[3] == "surv" )
  {
    cat(paste("Tree #", tree, " in the fitted survival forest: \n\n", sep = ""))
    
    OneTree = data.frame("SplitVar" = newnames[SplitVar + 1],
                         "SplitValue" = SplitValue,
                         "LeftNode" = LeftNode,
                         "RightNode" = RightNode,
                         "NodeWeight" = NodeWeight)
    
  }
  
  return(OneTree)
}
