library(RLT)
library(randomForest)
library(randomForestSRC)

# survival analysis

set.seed(1)

trainn = 500
testn = 200
n = trainn + testn
p = 2000
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = data.frame(matrix(as.integer(runif(n*p/2)*5), n, p/2))
for (j in 1:ncol(X2)) X2[,j] = as.factor(X2[,j])
X = cbind(X1, X2)
y = 10 + X[, 1] + 2*(X[, p/2+1] %in% c(1, 3)) + rnorm(n)
censor = rbinom(n, 1, 0.5)

ntrees = 10
ncores = 10
nmin = 20
mtry = ncol(X)
sampleprob = 0.8
rule = "best"
nsplit = ifelse(rule == "best", 0, 3)
importance = FALSE


trainX = X[1:trainn, ]
testX = X[1:trainn + testn, ]
trainY = y[1:trainn]
testY = y[1:trainn + testn]
trainCensor = censor[1:trainn]
testCensor = censor[1:trainn + testn]


metric = data.frame(matrix(NA, 2, 4))
rownames(metric) = c("rlt", "rsf")
colnames(metric) = c("fit.time", "pred.time", "pred.error", "obj.size")

start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, nmin = nmin/2, mtry = mtry, replacement = FALSE, 
              split.gen = rule, nsplit = nsplit, resample.prob = sampleprob, importance = importance, kernel.ready = TRUE, track.obs = TRUE)
metric[1, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX, ncores = ncores)
metric[1, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[1, 3] = 1- cindex(testY, testCensor, colSums(apply(RLTPred$hazard, 1, cumsum)))
metric[1, 4] = object.size(RLTfit)

object.size(trainX)

options(rf.cores = ncores)
start_time <- Sys.time()
rsffit <- rfsrc(Surv(trainY, trainCensor) ~ ., data = data.frame(trainX, trainY, trainCensor), ntree = ntrees, nodesize = nmin, mtry = mtry,
                nsplit = nsplit, sampsize = trainn*sampleprob, importance = importance, samptype = "swor")
metric[2, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rsfpred = predict(rsffit, data.frame(testX))
metric[2, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[2, 3] = 1- cindex(testY, testCensor, rowSums(rsfpred$chf))
metric[2, 4] = object.size(rsffit)


metric
RLTfit$parameters$kernel.ready
RLTfit$obs.w

KW = getKernelWeight(RLTfit, testX, ncores = ncores)


RLTfit$cindex

barplot(t(RLTfit$VarImp))


getOneTree(RLTfit, 1)

