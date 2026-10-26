encoding_split_fixture <- function(type = "factor") {
  group <- rep(c("A", "B"), each = 10)
  feature <- switch(type,
    factor = factor(group, levels = c("A", "B")),
    ordered = ordered(group, levels = c("A", "B")),
    character = group)
  track <- matrix(1L, 20, 2)
  track[c(10, 20), ] <- 0L
  RLT(data.frame(group = feature), rep(c(0, 10), each = 10),
      ntrees = 2, nmin = 2, resample.preset = track, seed = 4, ncores = 1)
}

test_that("factor labels survive releveling in predictions and both kernel inputs", {
  fit <- encoding_split_fixture()
  expect_true(all(vapply(fit$FittedForest$SplitVar, function(x) x[1] != -1, logical(1))))
  expect_identical(fit$xlevels, list(group = c("A", "B")))
  query <- data.frame(group = factor(c("A", "B"), levels = c("A", "B")))
  reversed <- data.frame(group = factor(c("A", "B"), levels = c("B", "A")))
  unused <- data.frame(group = factor(c("A", "B"), levels = c("unused", "B", "A")))

  expect_equal(as.numeric(predict(fit, query)$Prediction), c(0, 10))
  expect_equal(predict(fit, reversed), predict(fit, query))
  expect_equal(predict(fit, unused), predict(fit, query))
  expect_equal(as.numeric(predict(fit, data.frame(group = factor("B")))$Prediction), 10)

  expected <- diag(2, 2)
  expect_equal(forest.kernel(fit, reversed, query)$Kernel, expected)
  expect_equal(forest.kernel(fit, query, reversed)$Kernel, expected)
  expect_equal(forest.kernel(fit, unused, reversed)$Kernel, expected)
  bad <- data.frame(group = "C")
  expect_error(predict(fit, bad), "unknown label")
  expect_error(forest.kernel(fit, bad, query), "unknown label")
  expect_error(forest.kernel(fit, query, bad), "unknown label")
})

test_that("ordered and character features retain their training rank codes", {
  for (type in c("ordered", "character")) {
    fit <- encoding_split_fixture(type)
    expect_equal(unname(fit$ncat), 1L)
    expect_identical(fit$xlevels, list(group = c("A", "B")))
    query <- data.frame(group = if (type == "ordered") ordered("B") else "B")
    expect_equal(as.numeric(predict(fit, query)$Prediction), 10)
    expect_equal(predict(fit, query), predict(fit, matrix(2, 1, 1)))
  }
})

test_that("one-feature alignment retains a numeric matrix for all entry points", {
  fit <- encoding_split_fixture()
  query <- data.frame(group = factor(c("A", "B")))
  extra <- data.frame(extra = 1:2, group = factor(c("A", "B")))
  expect_warning(pred <- predict(fit, extra), "matching by name")
  expect_equal(pred, predict(fit, query))
  expect_warning(left <- forest.kernel(fit, extra, query), "matching by name")
  expect_warning(right <- forest.kernel(fit, query, extra), "matching by name")
  expect_equal(left$Kernel, diag(2, 2))
  expect_equal(right$Kernel, diag(2, 2))
})

test_that("the encoder aligns mixed features without coding numerical values as labels", {
  object <- list(parameters = list(p = 4),
    xnames = c("number", "nominal", "rank", "text"), ncat = c(1L, 2L, 1L, 1L),
    xlevels = list(NULL, c("B", "A"), c("low", "middle", "high"), c("A", "Z")))
  query <- data.frame(text = c("Z", "A"), extra = 99,
    nominal = factor(c("A", "B")), number = c(10, 20), rank = ordered(c("low", "high")))
  expect_warning(encoded <- RLT:::encode_RLT_features(object, query), "matching by name")
  expected <- cbind(number = c(10, 20), nominal = c(2, 1), rank = c(1, 3), text = c(2, 1))
  expect_equal(encoded, expected)
  numerical <- list(parameters = list(p = 1), xnames = "number", ncat = 1L, xlevels = list(NULL))
  expect_error(RLT:::encode_RLT_features(numerical, data.frame(number = c("10", "20"))), "was numerical")
  expect_error(RLT:::encode_RLT_features(numerical, data.frame(number = factor(c(10, 20)))), "was numerical")
  expect_equal(RLT:::encode_RLT_features(numerical, matrix(c(10, 20), 2, 1)), matrix(c(10, 20), 2, 1))
  expect_error(RLT:::encode_RLT_features(object, query[, -3, drop = FALSE]), "missing some variables")
})

test_that("numeric category codes are supported and validated", {
  fit <- encoding_split_fixture()
  expect_equal(as.numeric(predict(fit, matrix(1:2, 2, 1))$Prediction), c(0, 10))
  for (code in c(0, 3, 1.5, Inf, -Inf))
    expect_error(predict(fit, matrix(code, 1, 1)), "integer training codes")
  for (code in c(NA_real_, NaN))
    expect_error(predict(fit, matrix(code, 1, 1)), "missing values")
  expect_error(predict(fit, data.frame(group = factor(NA_character_))), "missing values")
  expect_error(predict(fit, matrix("B", 1, 1)), "character matrices")
  expect_error(RLT(matrix(c("A", "B"), 2, 1), c(0, 10), ntrees = 2), "character matrices")
})

test_that("already aligned duplicate names preserve distinct feature columns", {
  object <- list(parameters = list(p = 2), xnames = c("same", "same"),
    ncat = c(1L, 1L), xlevels = list(NULL, NULL))
  query <- cbind(same = 1:3, same = 11:13)
  expect_identical(RLT:::encode_RLT_features(object, query), query)
  extra <- cbind(extra = 21:23, query)
  expect_error(RLT:::encode_RLT_features(object, extra), "ambiguous duplicate variable names")
  object$xnames <- c("first", "second")
  ambiguous <- cbind(first = 1:3, first = 11:13, second = 21:23)
  expect_error(RLT:::encode_RLT_features(object, ambiguous), "ambiguous duplicate variable names")
})

test_that("legacy numerical encodings remain usable without guessing missing labels", {
  for (type in c("factor", "ordered", "character")) {
    fit <- encoding_split_fixture(type)
    legacy <- fit
    legacy$xlevels <- NULL
    codes <- matrix(1:2, 2, 1)
    expect_equal(predict(legacy, codes), predict(fit, codes))
    expect_equal(forest.kernel(legacy, codes, codes), forest.kernel(fit, codes, codes))
    expect_error(predict(legacy, data.frame(group = c("A", "B"))), "original training mapping.*refit")
    expect_error(forest.kernel(legacy, codes, data.frame(group = factor(c("A", "B")))), "original training mapping.*refit")
  }
  numerical <- list(parameters = list(p = 1), xnames = NULL, ncat = 1L)
  expect_equal(RLT:::encode_RLT_features(numerical, matrix(c(10, 20), 2, 1)), matrix(c(10, 20), 2, 1))
})

test_that("training mappings survive RDS serialization", {
  fit <- encoding_split_fixture()
  path <- tempfile(fileext = ".rds")
  saveRDS(fit, path)
  restored <- readRDS(path)
  unlink(path)
  query <- data.frame(group = factor(c("A", "B"), levels = c("B", "A")))
  expect_identical(restored$xlevels, fit$xlevels)
  expect_equal(predict(restored, query), predict(fit, query))
})

test_that("feature encoding reaches every model and split dispatch", {
  group <- factor(rep(c("A", "B"), each = 6))
  x <- data.frame(group = group, value = seq_len(12))
  track <- matrix(1L, 12, 2)
  track[c(6, 12), ] <- 0L
  query <- x[c(1, 7), , drop = FALSE]
  changed <- query
  changed$group <- factor(as.character(changed$group), levels = c("B", "A"))
  for (model in c("regression", "classification", "survival")) for (lc in c(1, 2)) {
    args <- list(x = x, y = if (model == "classification") group else rep(1:6, each = 2),
      model = model, ntrees = 2, nmin = 20, linear.comb = lc,
      resample.preset = track, seed = 4, ncores = 1)
    if (model == "survival") args$censor <- rep(1, 12)
    fit <- do.call(RLT, args)
    expect_identical(fit$xlevels, list(group = c("A", "B"), value = NULL))
    expect_equal(predict(fit, changed), predict(fit, query))
    expect_equal(forest.kernel(fit, changed, query)$Kernel, matrix(2, 2, 2))
  }
})

test_that("prediction grid size must be a finite nonnegative integer scalar", {
  fit <- encoding_split_fixture()
  for (size in list(NULL, numeric(), c(1, 2), NA_real_, NaN, Inf, -1, 1.5, "2", TRUE, 1 + 0i))
    expect_error(predict(fit, band.grid.size = size), "band.grid.size must be a finite nonnegative integer scalar")
  for (size in list(0, 1L, 2))
    expect_identical(predict(fit, band.grid.size = size), fit$Prediction)
})
