# Testing other features 

library(RLT)

set.seed(1)

trainn = 1000
testn = 500
n = trainn + testn
p = 30
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


RLTfit <- RLT(trainX, trainY, ntrees = 1000, ncores = 10, nmin = 50, mtry = p/3,
              split.gen = "random", nsplit = 3, resample.prob = 0.6, 
              importance = TRUE, resample.track = TRUE)

# Obtain the tree structure of one tree

get.one.tree(RLTfit, 1000)

# Forest Kernel
# since testing data is ordered by x1, closer subjects should 
# have larger kernel weights
A = forest.kernel(RLTfit, testX)
heatmap(A$Kernel, Rowv = NA, Colv = NA, symm = TRUE)

# cross kernels
A = forest.kernel(RLTfit, X1 = testX,
                  X2 = testX[1:(testn/2), ])
heatmap(A$Kernel, Rowv = NA, Colv = NA, symm = FALSE)

# vs.train 
A = forest.kernel(RLTfit, X1 = testX,
                  X2 = trainX[order(trainX[, 1]), ], 
                  vs.train = TRUE)
heatmap(A$Kernel, Rowv = NA, Colv = NA, symm = FALSE)
