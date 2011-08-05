#Tests further functionalitiy of gradient descent and its variations
#including gradient.descent.update
#Benchmark batch gradient descent and stochastic gradient descent
#Suppose Y.new comes from a different population (different linear
#combinations of the features), can gradient descent still find the
#best B value?
#This test case uses a larger dataset than test1.R

require(LargeRegression)

set.seed(100)

n = 2000
p = 100
numY = 50
X=matrix(rnorm(n*p),n,p)
B.true.1 = matrix(rnorm(p*numY), p, numY)
Y = X %*% B.true.1
#B.guess = lm(Y[1:(n/10),] ~ X[1:(n/10),])$coef
B.guess = matrix(0, p+1, numY)
alpha=1/n
convergence.threshold=0.0001
penalty = 0
min.iters=2
max.iters=200

gradient.descent(Y,X,B.guess,alpha,
  convergence.threshold,penalty,
  min.iters,max.iters, intercept=T)
lm(Y~X)$coef

X.new = matrix(rnorm(n*p),n,p)
B.true.2 = matrix(rnorm(p*numY), p, numY)
Y.new = X.new %*% B.true.2 + matrix(rnorm(n*numY), n, numY)
Y.augmented = rbind(Y,Y.new)
X.augmented = rbind(X,X.new)

alpha = 1/(2*n)

system.time(test1 <- lm(Y.augmented ~ X.augmented)$coef)
system.time(test2 <- gradient.descent(Y.augmented, X.augmented, B.guess, alpha, 
  convergence.threshold,penalty, min.iters, max.iters, intercept=T))
system.time(test2.gpu <- gradient.descent(Y.augmented, X.augmented, B.guess, alpha,
  convergence.threshold, min.iters, max.iters, intercept=T, gpu=T))

print("Results of lm:")
print(test1)
print("Results of gradient descent on the entire datasets (2 shards)")
print(test2)
print("Results of gradient descent with GPU acceleration")
print(test2.gpu)
