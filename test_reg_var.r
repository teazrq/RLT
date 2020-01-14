library(RLT)
library(randomForest)
library(randomForestSRC)
library(ranger)

set.seed(1)
n = 1000

p = 6
X = matrix(runif(n*p), n, p)
Y = X[, 1] + X[, 2] + X[, 3] + X[, 4] - 2 + rnorm(n)

testx = rbind(rep(0.5, p),
              c(0.4, 0.6, 0.4, 0.6, 0.4, 0.6),
              c(0.25, 0.75, 0.25, 0.75, 0.25, 0.75),
              rep(0.25, p),
              rep(0.75, p),
              rep(1, p))


myfit = rlt_var_est(X, Y, testx, ntrees = 1000, mtry = 3, nmin = 20, k = 600,
                    split.gen = "best", nsplit = 1, ncores = 12)

myfit$var

plot(myfit$estimation[1,])

rowSums(sweep(myfit$estimation, 2, myfit$allc, FUN = "*"))/sum(myfit$allc)

myfit$var

myfit$allc


myfit$sd



n = 1000
k = 500
x = seq(1:k)

dhyper(x, k, n - k, k)

qhyper(0.05, k, n - k, k)
qhyper(0.95, k, n - k, k)


