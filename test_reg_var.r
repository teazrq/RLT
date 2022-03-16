# Variance estimation of regression forest

library(RLT)

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

## Variance Estimation Example

RLTfit <- RLT(trainX, trainY, ntrees = 20000, ncores = 10, nmin = 8, 
              mtry = p, split.gen = "random", nsplit = 3, resample.prob = 0.5, 
              resample.replace = FALSE, var.ready = TRUE, resample.track = TRUE)

RLTPred <- predict(RLTfit, testX, var.est = TRUE, ncores = 10, keep.all = TRUE)

mean(RLTPred$Variance < 0)

cover = (1 + testX$X1 > RLTPred$Prediction - 1.96*sqrt(RLTPred$Variance)) & 
  (1 + testX$X1 < RLTPred$Prediction + 1.96*sqrt(RLTPred$Variance))

mean(cover, na.rm = TRUE)

par(mfrow=c(1,1))
par(mar = rep(2, 4))
plot(RLTPred$Prediction, 1 + testX$X1,  pch = 19, cex = ifelse(is.na(cover), 1, 0.3), 
     col = ifelse(is.na(cover), "red", ifelse(cover, "green", "black")))
abline(0, 1, col = "red", lwd = 2)

## Variance Estimation Example (for k > n/2)


Var.Est = Reg_Var_Forest(trainX, trainY, testX, ncores = 10, nmin = 8,
                         mtry = p, split.gen = "random", nsplit = 3, 
                         ntrees = 50000, resample.prob = 0.75)

mean(Var.Est$var < 0)
alphalvl = 0.05

cover = (1 + testX$X1 > Var.Est$Prediction - qnorm(1-alphalvl/2)*sqrt(Var.Est$var)) & 
  (1 + testX$X1 < Var.Est$Prediction + qnorm(1-alphalvl/2)*sqrt(Var.Est$var))

mean(cover, na.rm = TRUE)

par(mfrow=c(1,1))
par(mar = rep(2, 4))
plot(Var.Est$Prediction, 1 + testX$X1,  pch = 19, cex = ifelse(is.na(cover), 1, 0.3), 
     col = ifelse(is.na(cover), "red", ifelse(cover, "green", "black")))
abline(0, 1, col = "red", lwd = 2)



########### test


library(RLT)

set.seed(1)

trainn = 1000
testn = 1000
n = trainn + testn
p = 20

X = matrix(runif( (trainn + testn)*p ), trainn + testn, p)
y = 2*X[, 1] + 3*X[, 2] - 5*X[, 3] - 1*X[, 4] + 1 + rnorm(n)

trainX = X[1:trainn, ]
trainY = y[1:trainn]

testX = X[1:testn + trainn, ]
testY = y[1:testn + trainn]

RLTfit <- RLT(trainX, trainY, ntrees = 2000, ncores = 10, nmin = 30, 
              mtry = p/3, split.gen = "random", nsplit = 3, resample.prob = 0.75, 
              resample.replace = FALSE)

RLTPred <- predict(RLTfit, testX, ncores = 10)

mean( (RLTPred$Prediction - testY)^2)

#########coverage simulation 

nsim = 200
trainn = 200
testn = 50
p = 6

set.seed(2)
testX = matrix(runif(testn*p), testn, p)

rfpred = matrix(NA, nsim, testn)
est_var = matrix(NA, nsim, testn)
est_sd = matrix(NA, nsim, testn)

for (i in 1:nsim)
{
  cat(paste("\n\n---run", i, "...\n"))
  
  X = matrix(runif( trainn*p ), trainn, p)
  y = 2*X[, 1] + 3*X[, 2] - 5*X[, 3] - 1*X[, 4] + 1 + rnorm(trainn)
  
  Var.Est = Reg_Var_Forest(X, y, testX, ncores = 12, nmin = 5,
                           mtry = p/2, split.gen = "best", # nsplit = 4, 
                           ntrees = 20000, resample.prob = 0.80)
  
  rfpred[i, ] = Var.Est$Prediction
  est_var[i, ] = Var.Est$var
  cat(paste("negative rate", mean(est_var < 0, na.rm = TRUE)), "\n")
  
  sdi = sqrt(Var.Est$var)
  sdi[is.na(sdi)] = 1e-20

  est_sd[i, ] = sdi
  
  rfmeans = colMeans(rfpred, na.rm = TRUE)
  rfmeanmat = matrix(rep(rfmeans, each=nsim), nrow=nsim)
  
  cover95 = (rfpred - 1.96*est_sd < rfmeanmat) & (rfpred + 1.96*est_sd > rfmeanmat)
  cover90 = (rfpred - 1.64*est_sd < rfmeanmat) & (rfpred + 1.64*est_sd > rfmeanmat)
  
  cat(paste("95% coverage: ", mean(cover95, na.rm = TRUE), "\n"))
  cat(paste("90% coverage: ", mean(cover90, na.rm = TRUE), "\n"))
  
  rfsd = apply(rfpred, 2, sd, na.rm = TRUE)
  rfsdmat = matrix(rep(rfsd, each=nsim), nrow=nsim)
  cat(paste("relative bias", round(mean(est_sd / rfsd, na.rm = TRUE) - 1, 4), 
            "; relative sd = ", round(sd(est_sd / rfsd, na.rm = TRUE), 4)))
}


######### error debug


trainn = 200
testn = 50
p = 6

set.seed(914)

X = matrix(runif( trainn*p ), trainn, p)
y = 2*X[, 1] + 3*X[, 2] - 5*X[, 3] - 1*X[, 4] + 1 + rnorm(trainn)
testX = matrix(runif(testn*p), testn, p)  

Var.Est = Reg_Var_Forest(X, y, testX, ncores = 1, nmin = 20,
                         mtry = p/2, split.gen = "best", # nsplit = 4, 
                         ntrees = 100, resample.prob = 0.80, verbose = TRUE)
  



