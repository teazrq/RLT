# Align and encode new features using the fitted training dictionaries.
encode_RLT_features <- function(object, x, input = "testx")
{
  if (!is.matrix(x) && !is.data.frame(x))
    stop(input, " must be a matrix or a data.frame")
  if (is.matrix(x) && !is.numeric(x) && !is.logical(x))
    stop(input, " must be a numeric matrix; character matrices are not supported")

  p <- object$parameters$p
  if (ncol(x) < p)
    stop(input, " dimension does not match training data")

  if (!is.null(object$xnames) && !is.null(colnames(x)))
  {
    if (!identical(colnames(x), object$xnames))
    {
      matching_names <- colnames(x)[colnames(x) %in% object$xnames]
      if (anyDuplicated(object$xnames) || anyDuplicated(matching_names))
        stop(input, " has ambiguous duplicate variable names; supply columns in their original training order")
      varmatch <- match(object$xnames, colnames(x))
      if (anyNA(varmatch))
        stop(input, " is missing some variables from the original training data")
      warning(input, " variable names do not match training data; matching by name", call. = FALSE)
      x <- x[, varmatch, drop = FALSE]
    }
  }else if (ncol(x) != p){
    stop(input, " dimension does not match training data; matching variable names are not supplied")
  }

  xlevels <- object$xlevels
  legacy <- is.null(xlevels)
  if (!legacy && (!is.list(xlevels) || length(xlevels) != p))
    stop("The fitted object has invalid training feature mappings; refit the forest")

  for (j in seq_len(p))
  {
    value <- if (is.data.frame(x)) x[[j]] else x[, j]
    column <- if (!is.null(object$xnames)) object$xnames[j] else as.character(j)
    if (anyNA(value))
      stop(input, " column '", column, "' contains missing values")

    labels <- if (!legacy) xlevels[[j]] else NULL
    if (is.factor(value) || is.character(value))
    {
      if (legacy)
        stop(input, " label inputs require the original training mapping; restore that mapping or refit the forest")
      if (is.null(labels))
        stop(input, " column '", column, "' was numerical during training; supply numerical values")
      value <- match(as.character(value), labels)
      if (anyNA(value))
        stop(input, " column '", column, "' contains an unknown label")
    }else{
      # Legacy nominal factors retain their numeric-code interface. Their labels
      # cannot be reconstructed from ncat, nor can old ordered/character maps.
      nlevels <- if (!is.null(labels)) length(labels) else if (legacy && object$ncat[j] > 1) object$ncat[j] else NULL
      if (!is.null(nlevels))
      {
        if (!is.numeric(value) || is.complex(value) || any(!is.finite(value)) ||
            any(value != floor(value)) || any(value < 1 | value > nlevels))
          stop(input, " column '", column, "' must contain integer training codes in 1:", nlevels)
      }
    }

    if (is.data.frame(x)) x[[j]] <- value else x[, j] <- value
  }

  data.matrix(x)
}
