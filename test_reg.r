## Speed and Accuracy Test

library(RLT)
library(randomForest)
library(randomForestSRC)
library(ranger)
library(parallel)

set.seed(1)

trainn = 10000
testn = 1000
n = trainn + testn
p = 40
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = matrix(as.integer(runif(n*p/2)*3), n, p/2)

X = data.frame(X1, X2)
for (j in (p/2 + 1):p) X[,j] = as.factor(X[,j])
y = 1 + X[, 2] + 2 * (X[, p/2+1] %in% c(1, 3)) + rnorm(n)
#y = 1 + rowSums(X[, 1:(p/4)]) + rowSums(data.matrix(X[, (p/2) : (p/1.5)])) + rnorm(n)
#y = 1 + X[, 1] + rnorm(n)

ntrees = 100
ncores = detectCores() - 1
nmin = 60
mtry = p/2
sampleprob = 0.85
rule = "best"
nsplit = ifelse(rule == "best", 0, 3)
importance = TRUE 

trainX = X[1:trainn, ]
trainY = y[1:trainn]

testX = X[1:testn + trainn, ]
testY = y[1:testn + trainn]

metric = data.frame(matrix(NA, 4, 5))
rownames(metric) = c("rlt", "rsf", "rf", "ranger")
colnames(metric) = c("fit.time", "pred.time", "pred.error", 
                     "obj.size", "tree.size")

start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = sampleprob,
              importance = importance, param.control = list("alpha" = 0),
              verbose = TRUE)
metric[1, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX, ncores = ncores)
metric[1, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[1, 3] = mean((RLTPred$Prediction - testY)^2)
metric[1, 4] = object.size(RLTfit)
metric[1, 5] = mean(unlist(lapply(RLTfit$FittedForest$SplitVar, length)))

options(rf.cores = ncores)
start_time <- Sys.time()
rsffit <- rfsrc(y ~ ., data = data.frame(trainX, "y"= trainY), 
                ntree = ntrees, nodesize = nmin/2, mtry = mtry, 
                nsplit = nsplit, sampsize = trainn*sampleprob, 
                importance = ifelse(importance, "permute", "none"))
metric[2, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rsfpred = predict(rsffit, data.frame(testX))
metric[2, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[2, 3] = mean((rsfpred$predicted - testY)^2)
metric[2, 4] = object.size(rsffit)
metric[2, 5] = rsffit$forest$totalNodeCount / rsffit$forest$ntree

start_time <- Sys.time()
rf.fit <- randomForest(trainX, trainY, ntree = ntrees, 
                       mtry = mtry, nodesize = nmin, 
                       sampsize = trainn*sampleprob, 
                       importance = importance)
metric[3, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rf.pred <- predict(rf.fit, testX)
metric[3, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[3, 3] = mean((rf.pred - testY)^2)
metric[3, 4] = object.size(rf.fit)
metric[3, 5] = mean(colSums(rf.fit$forest$nodestatus != 0))

start_time <- Sys.time()
rangerfit <- ranger(trainY ~ ., data = data.frame(trainX), 
                    num.trees = ntrees, min.node.size = nmin, 
                    mtry = mtry, num.threads = ncores, 
                    sample.fraction = sampleprob, 
                    importance = "permutation",
                    respect.unordered.factors = "partition")
metric[4, 1] = difftime(Sys.time(), start_time, units = "secs")
rangerpred = predict(rangerfit, data.frame(testX))
metric[4, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[4, 3] = mean((rangerpred$predictions - testY)^2)
metric[4, 4] = object.size(rangerfit)
metric[4, 5] = mean(unlist(lapply(rangerfit$forest$split.varIDs, length)))

metric
mean((RLTfit$OOBPrediction - trainY)^2)

par(mfrow=c(2,2))
par(mar = c(1, 2, 2, 2))

barplot(as.vector(RLTfit$VarImp), main = "RLT")
barplot(as.vector(rsffit$importance), main = "rsf")
barplot(rf.fit$importance[, 1], main = "rf")
barplot(as.vector(rangerfit$variable.importance), main = "ranger")

# multivariate split 

set.seed(1)

n = 300
p = 100
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = matrix(as.integer(runif(n*p/2)*3), n, p/2)

X = data.frame(X1, X2)
for (j in (p/2 + 1):p) X[,j] = as.factor(X[,j])
y = 1 + X[, 1] + X[, 2] + (X[, p/2+1] %in% c(1, 3)) + rnorm(n)

trainX = X[1:(n/2), ]
trainY = y[1:(n/2)]

testX = X[-(1:(n/2)), ]
testY = y[-(1:(n/2))]

RLTfit <- RLT(X, y, ntrees = 100, ncores = 15, nmin = 3, 
              mtry = 50, linear.comb = 4, 
              resample.replace = TRUE, resample.prob = 0.6,
              split.gen = "rank", nsplit = 1, 
              param.control = list("split.rule" = "save"))



# RLT split 

set.seed(1)

n = 1000
p = 1000
X = matrix(rnorm(n*p), n, p)
y = 1 + X[, 1] + X[, 9] + X[, 3] + rnorm(n)

testX = matrix(rnorm(n*p), n, p)
testy = 1 + testX[, 1] + testX[, 9] + testX[, 3]  + rnorm(n)

start_time <- Sys.time()
RLTfit <- RLT(X, y, ntrees = 100, ncores = 15, nmin = 10,
              split.gen = "random", nsplit = 1, linear.comb = 1, 
              resample.prob = 0.85, resample.replace = FALSE,
              reinforcement = TRUE, importance = TRUE, 
              param.control = list("embed.ntrees" = 100,
                                   "embed.mtry" = 1/3,
                                   "embed.nmin" = 10,
                                   "embed.split.gen" = "random",
                                   "embed.nsplit" = 1,
                                   "embed.resample.prob" = 0.75,
                                   "embed.mute" = 0.75,
                                   "embed.protect" = 2))
difftime(Sys.time(), start_time, units = "secs")

barplot(as.vector(RLTfit$VarImp[1:50]), main = "RLT")

get.one.tree(RLTfit, 1)

mean((RLTfit$OOBPrediction - y)^2, na.rm = TRUE)
pred = predict(RLTfit, testX)
mean((pred$Prediction - testy)^2)


RLTfit <- RLT(X, y, ntrees = 1000, ncores = 6, nmin = 10,
              mtry = p, resample.prob = 0.85, 
              importance = TRUE, resample.track = TRUE)
mean((RLTfit$OOBPrediction - y)^2, na.rm = TRUE)
pred = predict(RLTfit, testX)
mean((pred$Prediction - testy)^2)
barplot(as.vector(RLTfit$VarImp[1:50]), main = "RLT")

# rf kernel
K = forest.kernel(RLTfit, X1 = X[1, , drop = FALSE], X2 = X)
