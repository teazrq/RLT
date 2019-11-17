library(RLT)
library(randomForest)
library(randomForestSRC)
library(survival)

# survival analysis

# generate data 
set.seed(1)

trainn = 600
testn = 1000
n = trainn + testn
p = 20
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = data.frame(matrix(as.integer(runif(n*p/2)*5), n, p/2))
for (j in 1:ncol(X2)) X2[,j] = as.factor(X2[,j])
X = cbind(X1, X2)
FT = rexp(n, rate = exp(X[, 1] + 3*X[, p/2 + 1] %in% c(1, 3)) )
CT = rexp(n, rate = exp( mean(X[, 1] - 2*X[, p/2 + 1] %in% c(1, 3)) ) )

y = pmin(FT, CT)
censor = as.numeric(FT <= CT)
mean(censor)

colnames(X) = NULL

ntrees = 1000
ncores = 10
nmin = 20
mtry = ncol(X)
sampleprob = 0.75
rule = "random"
nsplit = ifelse(rule == "best", 0, 1)
importance = TRUE


trainX = X[1:trainn, ]
testX = X[1:testn + trainn, ]
trainY = y[1:trainn]
testY = y[1:testn + trainn]
trainCensor = censor[1:trainn]
testCensor = censor[1:testn + trainn]

# get true survival function 
timepoints = sort(unique(trainY[trainCensor==1]))
SurvMat = matrix(NA, testn, length(timepoints))

for (j in 1:length(timepoints))
    SurvMat[, j] = 1 - pexp(timepoints[j], rate = exp(testX[, 1] + 3*as.numeric(testX[, p/2 + 1] %in% c(1, 3)) ) )

# fit models 

metric = data.frame(matrix(NA, 2, 5))
rownames(metric) = c("rlt", "rsf")
colnames(metric) = c("fit.time", "pred.time", "pred.error", "L1", "obj.size")

start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, nmin = nmin/2, mtry = mtry, replacement = FALSE, 
              split.gen = rule, nsplit = nsplit, resample.prob = sampleprob, importance = importance)
metric[1, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()

RLTPred <- predict(RLTfit, testX, ncores = ncores)
metric[1, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[1, 3] = 1- cindex(testY, testCensor, colSums(apply(RLTPred$hazard, 1, cumsum)))
metric[1, 4] = mean(colMeans(abs(RLTPred$Survival - SurvMat)))
metric[1, 5] = object.size(RLTfit)


options(rf.cores = ncores)
start_time <- Sys.time()
rsffit <- rfsrc(Surv(trainY, trainCensor) ~ ., data = data.frame(trainX, trainY, trainCensor), ntree = ntrees, nodesize = nmin, mtry = mtry,
                nsplit = nsplit, sampsize = trainn*sampleprob, importance = "permute", samptype = "swor")
metric[2, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rsfpred = predict(rsffit, data.frame(testX))
metric[2, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[2, 3] = 1- cindex(testY, testCensor, rowSums(rsfpred$chf))
metric[2, 4] = mean(colMeans(abs(rsfpred$survival - SurvMat)))
metric[2, 5] = object.size(rsffit)

metric



par(mfrow=c(2,2))
par(mar = c(2, 2, 2, 2))
barplot(as.vector(RLTfit$VarImp))
barplot(as.vector(rsffit$importance))
matplot(x = timepoints, y = t(RLTPred$Survival - SurvMat), type = "l")
matplot(x = timepoints, y = t(rsfpred$survival - SurvMat), type = "l")




RLTfit$parameters$kernel.ready
RLTfit$obs.w




KW = getKernelWeight(RLTfit, testX, ncores = ncores)


RLTfit$cindex

barplot(t(RLTfit$VarImp))


getOneTree(RLTfit, 1)


getOneTree(RLTfit, 1)
barplot(t(RLTfit$VarImp))
RLTfit$cindex