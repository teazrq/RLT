## Speed and Accuracy Test

library(RLT)
library(randomForest)
library(randomForestSRC)
library(ranger)
library(survival)

set.seed(1)

trainn = 250
testn = 5
n = trainn + testn
p = 400
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = matrix(as.integer(runif(n*p/2)*3), n, p/2)
C = rexp(n, 1)

X = data.frame(X1, X2)
xlink <- function(x) exp(x[, 7] + x[, 16] + x[, 25] + x[, p]) 
FT = rexp(n, rate = 1/xlink(X) )
CT = rexp(n, rate = 1)

y = pmin(FT, CT)
Censor = as.numeric(FT <= CT)
mean(Censor)

ntrees = 200
ncores = 10
nmin = 25
mtry = p/2
sampleprob = 0.85
rule = "best"
nsplit = ifelse(rule == "best", 0, 3)
importance = TRUE

trainX = X[1:trainn, ]
trainY = y[1:trainn]
trainCensor = Censor[1:trainn]

testX = X[1:testn + trainn, ]
testY = y[1:testn + trainn]
testCensor = Censor[1:testn + trainn]

# get true survival function 
timepoints = sort(unique(trainY[trainCensor==1]))
yloc = rep(NA, length(timepoints))
for (i in 1:length(timepoints)) yloc[i] = sum( timepoints[i] >= trainY )

SurvMat = matrix(NA, testn, length(timepoints))

for (j in 1:length(timepoints))
{
  SurvMat[, j] = 1 - pexp(timepoints[j], rate = 1/xlink(testX) )
}

metric = data.frame(matrix(NA, 4, 6))
rownames(metric) = c("rlt", "rsf", "rf", "ranger")
colnames(metric) = c("fit.time", "pred.time", "pred.error", "L1", 
                     "obj.size", "tree.size")

start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = sampleprob,
              importance = importance, param.control = list("alpha" = 0), 
              verbose = TRUE, resample.replace=FALSE)
metric[1, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX, ncores = ncores)
metric[1, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[1, 3] = 1- cindex(testY, testCensor, colSums(apply(RLTPred$hazard, 1, cumsum)))
metric[1, 4] = mean(colMeans(abs(RLTPred$Survival - SurvMat)))
metric[1, 5] = object.size(RLTfit)
metric[1, 6] = mean(unlist(lapply(RLTfit$FittedForest$SplitVar, length)))

options(rf.cores = ncores)
start_time <- Sys.time()
rsffit <- rfsrc(Surv(trainY, trainCensor) ~ ., data = data.frame(trainX, trainY, trainCensor), ntree = ntrees, nodesize = nmin, mtry = mtry,
                nsplit = nsplit, sampsize = trainn*sampleprob, 
                importance = ifelse(importance==TRUE,"random", "none"), samptype = "swor",
                block.size = 1)
metric[2, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rsfpred = predict(rsffit, data.frame(testX))
metric[2, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[2, 3] = 1- cindex(testY, testCensor, rowSums(rsfpred$chf))
metric[2, 4] = mean(colMeans(abs(rsfpred$survival - SurvMat)))
metric[2, 5] = object.size(rsffit)
metric[2, 6] = sum(is.na(rsffit$forest$nativeArray[,4]))/ntrees

start_time <- Sys.time()
rangerfit <- ranger(Surv(trainY, trainCensor) ~ ., data = data.frame(trainX, trainY, trainCensor), num.trees = ntrees, 
                    min.node.size = nmin, mtry = mtry, splitrule = "logrank", num.threads = ncores, 
                    sample.fraction = sampleprob, importance = "none")
metric[4, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rangerpred = predict(rangerfit, data.frame(testX))
metric[4, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[4, 3] = 1- cindex(testY, testCensor, rowSums(rangerpred$chf))
metric[4, 4] = mean(colMeans(abs(rangerpred$survival[, yloc] - SurvMat)))
metric[4, 5] = object.size(rangerfit)

metric

group <- factor(ifelse(c(1:(p/2)) %in% c(7, 16, 25, p), "Imp", "Not Imp"))
plot(c(1:(p/2)),RLTfit$VarImp[1:200,1], pch=19, col=group, xlab="X",ylab="Avg. Diff in C-index Error")
legend("bottomleft",
       legend = levels(factor(group)),
       pch = 19,
       col = factor(levels(factor(group))))

plot(rsffit$importance[1:200],RLTfit$VarImp[1:200,1], pch=19, col=group, 
     xlab="RSF Fit",ylab="RLT Fit")
legend("topleft",
       legend = levels(factor(group)),
       pch = 19,
       col = factor(levels(factor(group))))

start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = 0.5,
              importance = importance, param.control = list("alpha" = 0), 
              verbose = TRUE, resample.replace=FALSE, var.ready = TRUE)
print(difftime(Sys.time(), start_time, units = "secs"))
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX, ncores = ncores, var.est = TRUE, 
                   keep.all = TRUE, calc.cv = TRUE)
print(difftime(Sys.time(), start_time, units = "secs"))

plot(diag(RLTPred$Cov[,,5]))
matplot(RLTfit$timepoints, t(RLTPred$MarginalVar), type="l")
matplot(RLTfit$timepoints, t(RLTPred$MarginalVarSmooth), type="l")
plot(RLTfit$timepoints, RLTPred$CumHazard[1,]+RLTPred$CVprojSmooth[1]*sqrt(RLTPred$MarginalVarSmooth[1,]), type="l", 
     ylim = c(0,max(RLTPred$CumHazard[1,]+RLTPred$CVprojSmooth[1]*sqrt(RLTPred$MarginalVarSmooth[1,]))),
     ylab="Cumulative Hazard", xlab="Time")
lines(RLTfit$timepoints, RLTPred$CumHazard[1,]-RLTPred$CVprojSmooth[1]*sqrt(RLTPred$MarginalVarSmooth[1,]), type="l")
lines(RLTfit$timepoints, RLTPred$CumHazard[1,], type="l")

RLTfit <- Surv_Cov_Forest(trainX, trainY, trainCensor, testx = testX,
                          ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = 0.5,
              importance = importance, param.control = list("alpha" = 0), 
              verbose = TRUE)
plot(diag(RLTfit$Cov[,,5]))
RLTPred <- predict(RLTfit$Fit, testX, ncores = ncores, var.est = TRUE, keep.all = TRUE)
plot(diag(RLTfit$Cov[,,5]), diag(RLTPred$Cov[,,5]))
