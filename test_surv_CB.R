## Speed and Accuracy Test

library(RLT)
library(randomForest)
library(randomForestSRC)
library(ranger)
library(survival)

set.seed(1)

trainn = 250
testn = 1000
n = trainn + testn
p = 400
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

ntrees = 500
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

start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, trainCensor, ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = 0.5,
              importance = importance, param.control = list("alpha" = 0), 
              verbose = TRUE, resample.replace=FALSE, var.ready = TRUE)
print(difftime(Sys.time(), start_time, units = "secs"))
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX[1:5,], ncores = ncores, var.est = TRUE, 
                   keep.all = TRUE, calc.cv = TRUE)
print(difftime(Sys.time(), start_time, units = "secs"))

plot(RLTfit$timepoints, RLTPred$CumHazard[1,]+RLTPred$CVprojSmooth[1]*sqrt(RLTPred$MarginalVarSmooth[1,]), type="l", 
     ylim = c(0,max(RLTPred$CumHazard[1,]+RLTPred$CVprojSmooth[1]*sqrt(RLTPred$MarginalVarSmooth[1,]))),
     ylab="Cumulative Hazard", xlab="Time")
lines(RLTfit$timepoints, RLTPred$CumHazard[1,]-RLTPred$CVprojSmooth[1]*sqrt(RLTPred$MarginalVarSmooth[1,]), type="l")
lines(RLTfit$timepoints, RLTPred$CumHazard[1,], type="l", col="red")
