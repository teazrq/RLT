## Seed Test

library(RLT)

# setup data 

set.seed(1)

trainn = 1000
testn = 1000
n = trainn + testn
p = 20
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = matrix(as.integer(runif(n*p/2)*3), n, p/2)

X = data.frame(X1, X2)
for (j in (p/2 + 1):p) X[,j] = as.factor(X[,j])
y = 1 + X[, 1] + rnorm(n)

trainX = X[1:trainn, ]
trainY = y[1:trainn]

testX = X[1:testn + trainn, ]
testY = y[1:testn + trainn]

xorder = order(testX[, 1])
testX = testX[xorder, ]
testY = testY[xorder]

ntrees = 100
ncores = 10
nmin = 50
mtry = 1
sampleprob = 0.85
rule = "random"
nsplit = ifelse(rule == "best", 0, 3)
importance = TRUE


# fit model with or without pre-seeding

result_metric = data.frame(matrix(NA, 3, 5))
rownames(result_metric) = c("Noseed", "seed", "seed_rep")
colnames(result_metric) = c("fit.time", "pred.time", "pred.error", "obj.size", "ave.tree.size")

for (i in 1:3) {
  
  # the first one uses different seed
  # the second and third are using the same seed
  
  if (i > 1) set.seed(13)
  
  start_time <- Sys.time()
  RLTfit <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, nmin = nmin/2, mtry = mtry,
                split.gen = rule, nsplit = nsplit, resample.prob = sampleprob, resample.replace = TRUE,
                importance = importance)
  result_metric[i, 1] = difftime(Sys.time(), start_time, units = "secs")
  start_time <- Sys.time()
  
  
  RLTPred <- predict(RLTfit, testX, ncores = ncores)
  result_metric[i, 2] = difftime(Sys.time(), start_time, units = "secs")
  result_metric[i, 3] = mean((RLTPred$Prediction - testY)^2)
  result_metric[i, 4] = object.size(RLTfit)
  result_metric[i, 5] = mean(unlist(lapply(RLTfit$FittedForest$SplitVar, length)))

}

result_metric

## check seed and reproducibility on variance estimation
# still need to check this. 

Var.Est = Reg_Var_Forest(trainX, trainY, testX, ncores = 10, nmin = 30,
                         mtry = p, split.gen = "random", nsplit = 3, 
                         ntrees = 5000, resample.prob = 0.5)


Var.Est2 = Reg_Var_Forest(trainX, trainY, testX, ncores = 10, nmin = 30,
                         mtry = p, split.gen = "random", nsplit = 3, 
                         ntrees = 5000, resample.prob = 0.5, 
                         seed = Var.Est$Fit$parameters$seed)

all(Var.Est$var == Var.Est2$var)


