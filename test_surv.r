## Speed and Accuracy Test

library(RLT)
library(randomForest)
library(randomForestSRC)
library(ranger)
library(survival)

# set.seed(1)

trainn = 2000
testn = 1000
n = trainn + testn
p = 200
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = matrix(as.integer(runif(n*p/2)*3), n, p/2)
C = rexp(n, 1)

X = data.frame(X1, X2)
xlink <- function(x) exp(x[, 7] + x[, 16] + x[, 25] + x[, p]) 
FT = rexp(n, rate = xlink(X) )
CT = rexp(n, rate = 0.5)

y = pmin(FT, CT)
Censor = as.numeric(FT <= CT)
mean(Censor)

ntrees = 200
ncores = 10
nmin = 25
mtry = p/3
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

metric = data.frame(matrix(NA, 6, 6))
rownames(metric) = c("rlt", "rltsup", "rltcox", "rltcoxpen", "rsf", "ranger")
colnames(metric) = c("fit.time", "pred.time", "pred.error", "L1", 
                     "obj.size", "tree.size")

start_time <- Sys.time()

RLTfit <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = sampleprob,
              importance = importance, 
              param.control = list(split.rule = "logrank", "alpha" = 0.2), 
              verbose = TRUE, resample.replace=FALSE)
metric[1, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX, ncores = ncores)
metric[1, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[1, 3] = 1- cindex(testY, testCensor, colSums(apply(RLTPred$hazard, 1, cumsum)))
metric[1, 4] = mean(colMeans(abs(RLTPred$Survival - SurvMat)))
metric[1, 5] = object.size(RLTfit)
metric[1, 6] = mean(unlist(lapply(RLTfit$FittedForest$SplitVar, length)))

start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = sampleprob,
              importance = importance, 
              param.control = list(split.rule = "suplogrank", "alpha" = 0), 
              verbose = TRUE, resample.replace=FALSE)
metric[2, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX, ncores = ncores)
metric[2, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[2, 3] = 1- cindex(testY, testCensor, colSums(apply(RLTPred$hazard, 1, cumsum)))
metric[2, 4] = mean(colMeans(abs(RLTPred$Survival - SurvMat)))
metric[2, 5] = object.size(RLTfit)
metric[2, 6] = mean(unlist(lapply(RLTfit$FittedForest$SplitVar, length)))

start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = sampleprob,
              importance = importance,
              param.control = list(split.rule = "coxgrad", "alpha" = 0), 
              verbose = TRUE, resample.replace=FALSE)
metric[3, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX, ncores = ncores)
metric[3, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[3, 3] = 1- cindex(testY, testCensor, colSums(apply(RLTPred$hazard, 1, cumsum)))
metric[3, 4] = mean(colMeans(abs(RLTPred$Survival - SurvMat)))
metric[3, 5] = object.size(RLTfit)
metric[3, 6] = mean(unlist(lapply(RLTfit$FittedForest$SplitVar, length)))

start_time <- Sys.time()
RLTfitp <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = sampleprob,
              importance = FALSE, var.w = ifelse(c(1:(p)) %in% c(7, 16, 25, p), 1, 0.5),
              #var.w = pmax(RLTfit$VarImp[,1], min(abs(RLTfit$VarImp[,1][RLTfit$VarImp[,1]>0]))),
              param.control = list(split.rule = "coxgrad", "alpha" = 0), 
              verbose = TRUE, resample.replace=FALSE)
metric[4, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPredp <- predict(RLTfitp, testX, ncores = ncores)
metric[4, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[4, 3] = 1- cindex(testY, testCensor, colSums(apply(RLTPredp$hazard, 1, cumsum)))
metric[4, 4] = mean(colMeans(abs(RLTPredp$Survival - SurvMat)))
metric[4, 5] = object.size(RLTfitp)
metric[4, 6] = mean(unlist(lapply(RLTfitp$FittedForest$SplitVar, length)))


options(rf.cores = ncores)
start_time <- Sys.time()
rsffit <- rfsrc(Surv(trainY, trainCensor) ~ ., data = data.frame(trainX, trainY, trainCensor), ntree = ntrees, nodesize = nmin, mtry = mtry,
                nsplit = nsplit, sampsize = trainn*sampleprob, 
                importance = ifelse(importance==TRUE,"random", "none"), samptype = "swor",
                block.size = 1, ntime = NULL)
metric[5, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rsfpred = predict(rsffit, data.frame(testX))
metric[5, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[5, 3] = 1- cindex(testY, testCensor, rowSums(rsfpred$chf))
metric[5, 4] = mean(colMeans(abs(rsfpred$survival - SurvMat)))
metric[5, 5] = object.size(rsffit)
metric[5, 6] = rsffit$forest$totalNodeCount / rsffit$forest$ntree

start_time <- Sys.time()
rangerfit <- ranger(Surv(trainY, trainCensor) ~ ., data = data.frame(trainX, trainY, trainCensor), num.trees = ntrees, 
                    min.node.size = nmin, mtry = mtry, splitrule = "logrank", num.threads = ncores, 
                    sample.fraction = sampleprob, importance = "none")
metric[6, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rangerpred = predict(rangerfit, data.frame(testX))
metric[6, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[6, 3] = 1- cindex(testY, testCensor, rowSums(rangerpred$chf))
metric[6, 4] = mean(colMeans(abs(rangerpred$survival[, yloc] - SurvMat)))
metric[6, 5] = object.size(rangerfit)
metric[6, 6] = mean(unlist(lapply(rangerfit$forest$split.varIDs, length)))

metric

group <- factor(ifelse(c(1:p) %in% c(7, 16, 25, p), "Imp", "Not Imp"))
plot(c(1:p),RLTfit$VarImp[1:200,1], pch=19, col=group, xlab="X",ylab="Avg. Diff in C-index Error")
legend("topright",
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
RLTPred <- predict(RLTfit, testX[1:5,], ncores = ncores, var.est = TRUE, 
                   keep.all = TRUE)
print(difftime(Sys.time(), start_time, units = "secs"))

plot(diag(RLTPred$Cov[,,5]))
matplot(RLTfit$timepoints, t(RLTPred$MarginalVar), type="l")
matplot(RLTfit$timepoints, t(RLTPred$MarginalVarSmooth), type="l")
plot(RLTfit$timepoints, RLTPred$CumHazard[1,]+RLTPred$CVprojSmooth[1,10]*sqrt(RLTPred$MarginalVarSmooth[1,]), type="l", 
     ylim = c(0,max(RLTPred$CumHazard[1,]+RLTPred$CVprojSmooth[1,10]*sqrt(RLTPred$MarginalVarSmooth[1,]))),
     ylab="Cumulative Hazard", xlab="Time")
lines(RLTfit$timepoints, RLTPred$CumHazard[1,]-RLTPred$CVprojSmooth[1,10]*sqrt(RLTPred$MarginalVarSmooth[1,]), type="l")
lines(RLTfit$timepoints, RLTPred$CumHazard[1,], type="l")

RLTfit <- Surv_Cov_Forest(trainX, trainY, trainCensor, testx = testX[1:5,],
                          ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = 0.5,
              importance = importance, param.control = list("alpha" = 0), 
              verbose = TRUE)
plot(diag(RLTfit$Cov[,,5]))
RLTPred <- predict(RLTfit$Fit, testX[1:5,], ncores = ncores, var.est = TRUE, keep.all = TRUE)
plot(diag(RLTfit$Cov[,,5]), diag(RLTPred$Cov[,,5]))


RLTfit <- RLT(trainX[,1:25], trainY, trainCensor, ntrees = ntrees*10, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = 0.5,
              importance = importance,
              param.control = list(split.rule = "logrank", "alpha" = 0,
                                   VI.var=TRUE), 
              verbose = TRUE, resample.replace=FALSE)

rsffit <- rfsrc(Surv(trainY, trainCensor) ~ ., 
                data = data.frame(trainX, trainY, trainCensor), 
                ntree = ntrees*10, nodesize = nmin, mtry = mtry,
                nsplit = nsplit, sampsize = trainn*sampleprob, 
                importance = ifelse(importance==TRUE,"random", "none"), 
                samptype = "swor",
                block.size = 1, ntime = NULL)

rsffitsamp <- subsample(rsffit, B=100)
plot(rsffitsamp)

library(ggplot2)
proj <- nearPD(RLTfit$VarImpCov, base.matrix = TRUE)$mat
VarImpMat <- data.frame(Importance=RLTfit$VarImp[,1], Var=diag(RLTfit$VarImpCov[,1]), 
                        sd=sqrt(diag(RLTfit$VarImpCov[,1])),
                        x=c(1:p), ImpVar=c(1:p)%in%c(7, 16, 25, p),
                        sd_proj=sqrt(diag(proj)),Var_proj=diag(proj))
ggplot(VarImpMat[1:25,], aes(x=x, y=Importance, color=ImpVar)) + 
  geom_point() +
  geom_errorbar(aes(ymin=Importance-1.96*sd_proj, ymax=Importance+1.96*sd_proj), width=.2,
                position=position_dodge(.9))

k <- trainn/2
koob <- k*0.2
kinb <- k-koob
sampvec <- c(rep(-1, k), rep(1, kinb), rep(0, koob))
resamp1 <- sapply(c(1:(ntrees/2)), function(i) sample(sampvec, trainn))
resamp2 <- apply(resamp1, 2, function(col){
  newinds <- c(1:trainn)[col==(-1)]
  inbag <- sample(newinds, kinb)
  col2 <- rep(0, trainn)
  col2[inbag] <- 1
  col2[-newinds] <- -1
  return(col2)
} )
obstrack <- cbind(resamp1, resamp2)

RLTfit <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = sampleprob,
              importance = importance, 
              resample.preset = obstrack,
              param.control = list(split.rule = "coxgrad", "alpha" = 0), 
              verbose = TRUE, resample.replace=FALSE)

RLTfitp <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, 
               nmin = nmin, mtry = mtry, nsplit = nsplit,
               split.gen = rule, resample.prob = sampleprob,
               importance = FALSE, var.w = ifelse(c(1:(p)) %in% c(7, 16, 25, p), 1, 0.5),
               resample.preset = obstrack,
               #var.w = pmax(RLTfit$VarImp[,1], min(abs(RLTfit$VarImp[,1][RLTfit$VarImp[,1]>0]))),
               param.control = list(split.rule = "coxgrad", "alpha" = 0), 
               verbose = TRUE, resample.replace=FALSE)

cindex_diff <- (RLTfit$cindex_tree - (RLTfitp$cindex_tree))[,1]
tree_var <- sum((cindex_diff[1:(ntrees/2)] - cindex_diff[(ntrees/2+1):ntrees])^2)/(ntrees)
vars <- var(cindex_diff)
VarUforest <- tree_var - vars*(1 - 1/2/ntrees)
plot(1-RLTfit$cindex_tree, 1-RLTfitp$cindex_tree)
lines(x=c(0,1), y=c(0,1))
plot(cindex_diff)
lines(x=c(0,500), y=c(mean(cindex_diff),mean(cindex_diff)))
mean(cindex_diff)+c(-1,1)*1.96*sqrt(VarUforest)
#Positive implies penalized forest is better as original has higher error

RLTfit <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = sampleprob,
              importance = importance, 
              resample.preset = obstrack,
              param.control = list(split.rule = "coxgrad", "alpha" = 0), 
              verbose = TRUE, resample.replace=FALSE)

RLTfitp <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, 
               nmin = nmin, mtry = mtry, nsplit = nsplit,
               split.gen = rule, resample.prob = sampleprob,
               importance = FALSE, var.w = pmax(RLTfit$VarImp[,1],min(RLTfit$VarImp[,1][RLTfit$VarImp[,1]>0])),
               resample.preset = obstrack,
               #var.w = pmax(RLTfit$VarImp[,1], min(abs(RLTfit$VarImp[,1][RLTfit$VarImp[,1]>0]))),
               param.control = list(split.rule = "coxgrad", "alpha" = 0), 
               verbose = TRUE, resample.replace=FALSE)

cindex_diff <- (RLTfit$cindex_tree - (RLTfitp$cindex_tree))[,1]
tree_var <- sum((cindex_diff[1:(ntrees/2)] - cindex_diff[(ntrees/2+1):ntrees])^2)/(ntrees)
vars <- var(cindex_diff)
VarUforest <- tree_var - vars*(1 - 1/2/ntrees)
plot(1-RLTfit$cindex_tree, 1-RLTfitp$cindex_tree)
lines(x=c(0,1), y=c(0,1))
plot(cindex_diff)
lines(x=c(0,500), y=c(mean(cindex_diff),mean(cindex_diff)))
mean(cindex_diff)+c(-1,1)*1.96*sqrt(VarUforest)
#Positive implies penalized forest is better as original has higher error
