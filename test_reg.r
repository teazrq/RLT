library(RLT)
library(randomForest)
library(randomForestSRC)

set.seed(1)
n = 500
p = 20
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = matrix(as.integer(runif(n*p/2)*5), n, p/2)
for (j in ncol(X2)) X2[,j] = as.factor(X2[,j])

X = cbind(X1, X2)
y = 1 + X[, 1] + X[, p/2+1] %in% c(1, 3) + rnorm(n)

ntrees = 200
ncores = 5
nmin = 10
mtry = p
sampleprob = 0.85
rule = "best"
nsplit = ifelse(rule == "best", 0, 3)

trainn = n/2
testn = n - trainn

trainX = X[1:trainn, ]
testX = X[1:trainn + testn, ]
trainY = y[1:trainn]
testY = y[1:trainn + testn]

metric = data.frame(matrix(NA, 3, 4))
rownames(metric) = c("rlt", "rsf", "rf")
colnames(metric) = c("fit.time", "pred.time", "pred.error", "obj.size")

start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, nmin = nmin, mtry = mtry,
              split.gen = rule, nsplit = nsplit, resample.prob = sampleprob)
metric[1, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX, ncores = ncores)
metric[1, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[1, 3] = mean((RLTPred$Prediction - testY)^2)
metric[1, 4] = object.size(RLTfit)

options(rf.cores = ncores)
start_time <- Sys.time()
rsffit <- rfsrc(y ~ ., data = data.frame(trainX, "y"= trainY), ntree = ntrees, nodesize = nmin, mtry = mtry, 
                nsplit = nsplit, sampsize = trainn*sampleprob, importance = FALSE)
metric[2, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rsfpred = predict(rsffit, data.frame(testX))
metric[2, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[2, 3] = mean((rsfpred$predicted - testY)^2)
metric[2, 4] = object.size(rsffit)

start_time <- Sys.time()
rf.fit <- randomForest(trainX, trainY, ntree = ntrees, mtry = mtry, nodesize = nmin, sampsize = trainn*sampleprob)
metric[3, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rf.pred <- predict(rf.fit, testX)
metric[3, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[3, 3] = mean((rf.pred - testY)^2)
metric[3, 4] = object.size(rf.fit)

metric

################# other features of RLT ##########################

ntrees = 10
MySamples = matrix(sample(c(0, 1), ntrees*nrow(trainX), replace = TRUE), ncol = ntrees)

RLTfit <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, nmin = nmin, mtry = mtry,
              split.gen = rule, nsplit = nsplit, resample.prob = sampleprob, 
              ObsTrack = MySamples)

RLTPred <- predict(RLTfit, testX, ncores = ncores)
mean((RLTPred$Prediction - testY)^2)

# oob predictions 

mean((RLTfit$OOBPrediction - y[1:trainn])^2)
mean((RLTfit$Prediction - y[1:trainn])^2)

# kernel weights

y = 1 + X[, 1] + X[, 2] + rnorm(n)
trainY = y[1:trainn]

RLTfit <- RLT(trainX, trainY, kernel.ready = TRUE)
RLTkernel = getKernelWeight(RLTfit, X[trainn + 1:2, ])
# heatmap(RLTkernel$Kernel[[1]], Rowv = NA, Colv = NA)

plot(trainX[, 1], trainX[, 2] + rnorm(trainn, sd = 0.1), pch = 19,
     cex = rowMeans(RLTkernel$Kernel[[1]])*15, xlab = "x1", ylab = "x2")

# peek a tree
getOneTree(RLTfit, 1)




