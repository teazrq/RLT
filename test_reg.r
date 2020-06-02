library(RLT)
library(randomForest)
library(randomForestSRC)
library(ranger)

set.seed(1)
trainn = 500
testn = 1000
n = trainn + testn
p = 100
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = matrix(as.integer(runif(n*p/2)*3), n, p/2)
for (j in 1:ncol(X2)) X2[,j] = as.factor(X2[,j])

X = cbind(X1, X2)
#y = 1 + X[, 1] + 2 * (X[, p/2+1] %in% c(1, 3)) + rnorm(n)
y = 1 + X[, 1] + rnorm(n)

ntrees = 200
ncores = 10
nmin = 20
mtry = p
sampleprob = 0.85
rule = "best"
nsplit = ifelse(rule == "best", 0, 3)
importance = TRUE 

trainX = X[1:trainn, ]
trainY = y[1:trainn]

testX = X[1:testn + trainn, ]
testY = y[1:testn + trainn]

xorder = order(testX[, 1])
testX = testX[xorder, ]
testY = testY[xorder]

metric = data.frame(matrix(NA, 4, 4))
rownames(metric) = c("rlt", "rsf", "rf", "ranger")
colnames(metric) = c("fit.time", "pred.time", "pred.error", "obj.size")

start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, nmin = nmin/2, mtry = mtry,
              split.gen = rule, nsplit = nsplit, resample.prob = sampleprob, importance = importance)
metric[1, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX, ncores = ncores)
metric[1, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[1, 3] = mean((RLTPred$Prediction - testY)^2)
metric[1, 4] = object.size(RLTfit)

options(rf.cores = ncores)
start_time <- Sys.time()
rsffit <- rfsrc(y ~ ., data = data.frame(trainX, "y"= trainY), ntree = ntrees, nodesize = nmin, mtry = mtry, 
                nsplit = nsplit, sampsize = trainn*sampleprob, importance = importance)
metric[2, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rsfpred = predict(rsffit, data.frame(testX))
metric[2, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[2, 3] = mean((rsfpred$predicted - testY)^2)
metric[2, 4] = object.size(rsffit)

start_time <- Sys.time()
rf.fit <- randomForest(trainX, trainY, ntree = ntrees, mtry = mtry, nodesize = nmin, sampsize = trainn*sampleprob, importance = importance)
metric[3, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rf.pred <- predict(rf.fit, testX)
metric[3, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[3, 3] = mean((rf.pred - testY)^2)
metric[3, 4] = object.size(rf.fit)

start_time <- Sys.time()
rangerfit <- ranger(trainY ~ ., data = data.frame(trainX), num.trees = ntrees, 
                    min.node.size = nmin, mtry = mtry, num.threads = ncores, 
                    sample.fraction = sampleprob, importance = "permutation")
metric[4, 1] = difftime(Sys.time(), start_time, units = "secs")
rangerpred = predict(rangerfit, data.frame(testX))
metric[4, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[4, 3] = mean((rangerpred$predictions - testY)^2)
metric[4, 4] = object.size(rangerfit)


metric

par(mfrow=c(2,2))
par(mar = c(0.5, 0.5, 2, 2))

barplot(as.vector(RLTfit$VarImp), main = "RLT")
barplot(as.vector(rsffit$importance), main = "rsf")
barplot(rf.fit$importance[, 1], main = "rf")
barplot(as.vector(rangerfit$variable.importance), main = "ranger")


# getOneTree(RLTfit, 1)

A = forest.kernel(RLTfit, testX)
heatmap(A$Kernel, Rowv = NA, Colv = NA, symm = TRUE)


# predict on a subset of trees 

RLTPred_sub <- predict(RLTfit, testX, treeindex = c(1:10), keep.all = TRUE, ncores = ncores)


################# RLT with Embedded Model ##########################


RLTfit <- RLT(trainX, trainY, ntrees = 1, ncores = 1, nmin = nmin/2, mtry = mtry,
              split.gen = rule, nsplit = nsplit, resample.prob = sampleprob, importance = importance, 
              reinforcement = TRUE, verbose = TRUE)

getOneTree(RLTfit, 1)






################# other features of RLT ##########################


ntrees = 2000
mtry = p/2
rule = "random"
nsplit = 1

K = floor(trainn * 0.85)

NP = K
NQ = trainn - K

ALLC = 0:K

Den = dhyper(ALLC, NP, NQ, K, log = FALSE)
plot(ALLC, Den)
C = ALLC[which.max(Den)]
C

MySamples1 = rbind(matrix(1, C, ntrees), 
                   matrix(1, K - C, ntrees), 
                   matrix(0, trainn - K, ntrees))

MySamples2 = rbind(matrix(1, C, ntrees), 
                   matrix(0, trainn - K, ntrees),
                   matrix(1, K - C, ntrees))

for (j in 1:ntrees)
{
    shuffle = sample(1:trainn)
    
    MySamples1[, j] = MySamples1[shuffle, j]
    MySamples2[, j] = MySamples2[shuffle, j]
}

table(colSums(MySamples1 * MySamples2))



RLTfit1 <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, nmin = nmin, mtry = mtry,
              split.gen = rule, nsplit = nsplit, track.obs = 2, ObsTrack = MySamples1)



RLTPred1 <- predict(RLTfit1, testX, keep.all = TRUE, ncores = ncores)
mean((RLTPred1$Prediction - testY)^2)

# fit another random forests with the same subsample index
RLTfit2 <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, nmin = nmin, mtry = mtry,
              split.gen = rule, nsplit = nsplit, ObsTrack = MySamples2)

RLTPred2 <- predict(RLTfit2, testX, keep.all = TRUE, ncores = ncores)
mean((RLTPred2$Prediction - testY)^2)


subj = 2

# Var(Trees)

var(c(RLTPred1$PredictionAll[subj, ], RLTPred2$PredictionAll[subj, ]))
var(RLTPred2$PredictionAll[subj, ])


# E[Var[Trees | shared]]
mean((RLTPred1$PredictionAll[subj, ] - RLTPred2$PredictionAll[subj, ])^2/2)



RLTfit_bs <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, nmin = nmin, mtry = mtry,
               split.gen = rule, nsplit = nsplit, 
               resample.prob = 0.85, replacement = TRUE)
RLTPred_bs <- predict(RLTfit_bs, testX, keep.all = TRUE, ncores = ncores)
var(RLTPred_bs$PredictionAll[subj, ])



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



