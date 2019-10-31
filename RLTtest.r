set.seed(1)
n = 500
p = 100
X = cbind(matrix(rnorm(n*p), n, p), matrix(as.integer(runif(n*p)*10), n, p))
X = as.data.frame(X)
for (j in (1:p + p)) X[,j] = as.factor(X[,j])

# y = 2 + rowSums(data.matrix(X[, 1:5])) * 2 + rowSums(data.matrix(X[,1:5 + p]))*0.5 + rnorm(n)
# y = 2 + rowSums(data.matrix(X[, 1:5])) * 2 + rnorm(n)
 y = 2 + X[, 1] * 1 + X[, 2] * 1 + X[, 3] * 1 + rnorm(n)

trainn = n/2
testn = n - trainn
ntrees = 100
ncores = 3
nmin = 2
mtry = p
sampleprob = 0.85
rule = "best"
nsplit = ifelse(rule == "best", 0, 3)

start_time <- Sys.time()
RLTfit <- RLT(X[1:trainn, ], y[1:trainn], ntrees = ntrees, ncores = ncores, nmin = nmin, mtry = mtry, # obs.w = runif(trainn), 
           split.gen = rule, nsplit = nsplit, replacement = FALSE, resample.prob = sampleprob, kernel.ready = TRUE)
Sys.time() - start_time


RLTkernel = getKernelWeight(RLTfit, X[1:trainn + testn, ])
heatmap(RLTkernel$Kernel[[1]], Rowv = NA, Colv = NA)
sum(RLTkernel$Kernel[[1]])/ntrees

plot(RLTfit$Prediction, y[1:trainn])
plot(RLTfit$OOBPrediction, y[1:trainn])
mean((RLTfit$OOBPrediction - y[1:trainn])^2)

start_time <- Sys.time()
RLTpred = predict(RLTfit, X[1:trainn + testn, ], ncores = ncores)
Sys.time() - start_time

plot(RLTpred$Prediction, y[1:trainn + testn])
mean((RLTpred$Prediction - y[1:trainn + testn])^2)


start_time <- Sys.time()
RLTpred = predict(RLTfit, X[1:trainn + testn, ], kernel = TRUE, ncores = ncores)
Sys.time() - start_time
plot(RLTpred$Prediction, y[1:trainn + testn])
mean((RLTpred$Prediction - y[1:trainn + testn])^2)



for (i in 1:ntrees)
{
  if ( any(sort(unlist(RLTfit$NodeRegi[[1]])) != 1:trainn -1) )
    cat("NodeRegi does not match \n")
}

getOneTree(RLTfit, 1)
getOneTree(RLTfit, 100)$NodeType


getKernelWeight(RLTfit, X[1:2 + testn, ])









# survival analysis

set.seed(1)
n = 500
p = 100
X = cbind(matrix(rnorm(n*p), n, p), matrix(as.integer(runif(n*p)*10), n, p))
X = as.data.frame(X)
for (j in (1:p + p)) X[,j] = as.factor(X[,j])

# y = 2 + rowSums(data.matrix(X[, 1:5])) * 2 + rowSums(data.matrix(X[,1:5 + p]))*0.5 + rnorm(n)
# y = 2 + X[, 2] * 3 + rnorm(n) + 10

censor = rbinom(n, 1, 0.5)
y = 10 + rnorm(n)


trainn = n/2
testn = n - trainn
ntrees = 300
ncores = 5
nmin = 5
mtry = ncol(X)
sampleprob = 0.8
rule = "best"
nsplit = ifelse(rule == "best", 0, 3)
importance = TRUE

start_time <- Sys.time()
RLTfit <- RLT(X[1:trainn, ], y[1:trainn], censor[1:trainn], ntrees = ntrees, ncores = ncores, nmin = nmin, mtry = mtry, # obs.w = runif(trainn), 
              split.gen = rule, nsplit = nsplit, replacement = FALSE, resample.prob = sampleprob, kernel.ready = TRUE, importance = importance)
Sys.time() - start_time

library(randomForestSRC)
library(survival)

rsffit <- rfsrc(Surv(Y, Censor) ~ ., data = data.frame("x" = X[1:trainn, ], "Y" = y[1:trainn], "Censor" = censor[1:trainn]), 
                ntree = ntrees, nodesize = nmin, mtry = mtry, nsplit = nsplit, sampsize = trainn*sampleprob)

RLTpred = predict(RLTfit, X[1:trainn + testn, ], kernel = FALSE, ncores = ncores)
rsfpred = predict(rsffit, data = data.frame("x" = X[1:trainn + testn, ]))
apply(abs(rsfpred$survival - RLTpred$Survival), 2, median)
apply(abs(rsfpred$survival - RLTpred$Survival), 2, mean)

for (i in 1:15)
{
    plot(rsfpred$time.interest, rsfpred$survival[i,], type = "s", col = "red", xlim = c(min(y), max(y)), ylim = c(0, 1))
    points(RLTfit$timepoints, RLTpred$Survival[i,], type = "s", col = "blue")
    abline(h = 0.5)
}


############################

set.seed(1)
n = 500
p = 100

#X <- data.frame(as.factor(sample(10,n,replace = TRUE)))
#for(i in 2:p) X <- cbind(X,as.factor(sample(10,n,replace = TRUE)))
#X <- cbind(X, matrix(rnorm(n*p,c(1:p),5/c(1:p)),ncol=p))


#X = cbind(matrix(rnorm(n*p), n, p), matrix(as.integer(runif(n*p)*10), n, p))
#X = as.data.frame(X)
#for (j in (1:p + p)) X[,j] = as.factor(X[,j])

X = cbind( matrix(as.integer(runif(n*p)*3), n, p), matrix(rnorm(n*p), n, p))
X = as.data.frame(X)
for (j in (1:p)) X[,j] = as.factor(X[,j])


# y = 2 + rowSums(data.matrix(X[, 1:5])) * 2 + rowSums(data.matrix(X[,1:5 + p]))*0.5 + rnorm(n)

censor = sample(c(0,1),n,replace = TRUE,prob = c(0.5,0.5))
y = 10 + rnorm(n)

trainn = n/2
testn = n - trainn
ntrees = 500
ncores = 5
nmin = 10
mtry = ncol(X)
sampleprob = 0.85
rule = "random"
nsplit = ifelse(rule == "best", 0, 3)

start_time <- Sys.time()
RLTfit <- RLT(X[1:trainn, ], y[1:trainn], censor[1:trainn], ntrees = ntrees, ncores = ncores, nmin = nmin, mtry = mtry, # obs.w = runif(trainn), 
              split.gen = rule, nsplit = nsplit, replacement = FALSE, resample.prob = sampleprob, kernel.ready = TRUE, importance = importance)
Sys.time() - start_time

rsffit <- rfsrc(Surv(Y, Censor) ~ ., data = data.frame("x" = X[1:trainn, ], "Y" = y[1:trainn], "Censor" = censor[1:trainn]), 
                ntree = ntrees, nodesize = nmin, mtry = mtry, nsplit = nsplit, sampsize = trainn*sampleprob)


RLTpred = predict(RLTfit, X[1:trainn + testn, ], kernel = FALSE, ncores = ncores)
rsfpred = predict(rsffit, data = data.frame("x" = X[1:trainn + testn, ]))
apply(abs(rsfpred$survival - RLTpred$Survival), 2, median)
apply(abs(rsfpred$survival - RLTpred$Survival), 2, mean)

RLTfit$cindex
max(rsffit$err.rate, na.rm = TRUE)

for (i in 101:115)
{
    plot(rsfpred$time.interest, rsfpred$survival[i,], type = "s", col = "red", xlim = c(min(y), max(y)), ylim = c(0, 1))
    points(RLTfit$timepoints, RLTpred$Survival[i,], type = "s", col = "blue")
    abline(h = 0.5)
    abline(v = 10)
}
















barplot(t(RLTfit$VarImp))


getOneTree(RLTfit, 1)


start_time <- Sys.time()
RLTpred = predict(RLTfit, X[1:trainn + testn, ], kernel = FALSE, ncores = ncores)
Sys.time() - start_time

# plot(RLTpred$Survival[, 60], y[1:trainn + testn])


rsfpred = predict(rsffit, data = data.frame("x" = X[1:trainn + testn, ]))


RLTpred = predict(RLTfit, X[1:trainn + testn, ], kernel = FALSE, ncores = ncores)
rsfpred = predict(rsffit, data = data.frame("x" = X[1:trainn + testn, ]))
apply(abs(rsfpred$survival - RLTpred$Survival), 2, median)
apply(abs(rsfpred$survival - RLTpred$Survival), 2, mean)


rsffit$forest$nativeArray[rsffit$forest$nativeArray$treeID == 1, ]
getOneTree(RLTfit, 1)


par(mfrow=c(5, 3))
par(mar = c(0,0,0,0))

for (i in 101:115)
{
    plot(rsfpred$time.interest, rsfpred$survival[i,], type = "s", col = "red", xlim = c(min(y), max(y)), ylim = c(0, 1))
    points(RLTfit$timepoints, RLTpred$Survival[i,], type = "s", col = "blue")
    abline(h = 0.5)
}


abline(v = 2 + X[trainn + i, 2] * 3 + 10)




plot(Surv( y[1:trainn], censor[1:trainn]))





# compare with other models 


library(randomForest)
library(randomForestSRC)

set.seed(1)
n = 2000
p = 100
X = cbind(matrix(rnorm(n*p), n, p), matrix(as.integer(runif(n*p)*3), n, p))
for (j in (1:p + p)) X[,j] = as.factor(X[,j])
X = as.data.frame(X)

#y = 2 + as.numeric(rowSums(X[, 1:10])) * 1 + as.numeric(rowSums(X[,1:10 + p]))*0.25 + rnorm(n, 0, 1)
 y = 2 + X[, 1] * 2 + X[, 2] * 2 + rnorm(n)
# y = 2 + as.numeric(rowSums(X[, 1:10])) * 1 + rnorm(n, 0, 1)

trainn = n/2
testn = n - trainn

trainX = X[1:trainn, ]
testX = X[1:trainn + testn, ]
trainY = y[1:trainn]
testY = y[1:trainn + testn]

ntrees = 1000
ncores = 1
nmin = 1
mtry = 2
sampleprob = 1
rule = "best"
nsplit = ifelse(rule == "best", 0, 3)

metric = data.frame(matrix(NA, 3, 4))
rownames(metric) = c("rlt", "rsf", "rf")
colnames(metric) = c("fit.time", "pred.time", "accuracy", "obj.size")


start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, nmin = nmin, mtry = mtry, # obs.w = runif(trainn), 
              split.gen = rule, nsplit = nsplit, replacement = FALSE, resample.prob = sampleprob, kernel.ready = FALSE)
metric[1, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX, ncores = ncores)
metric[1, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[1, 3] = mean((RLTPred$Prediction - testY)^2)
metric[1, 4] = object.size(RLTfit)


options(rf.cores = ncores)
start_time <- Sys.time()
rsffit <- rfsrc(y ~ ., data = data.frame(trainX, "y"= trainY), ntree = ntrees, nodesize = nmin, mtry = mtry, nsplit = nsplit, sampsize = trainn*sampleprob)
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




