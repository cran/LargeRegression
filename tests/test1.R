#Tests basic functionality of gradient.descent
#Make sure gradient.descent converged to something reasonably close
#to the lm command

require(LargeRegression)

set.seed(100)

n = 100
p = 5
numY = 2
X=matrix(rnorm(n*p),n,p)
B.true = matrix(rnorm(p*numY), p, numY)
Y = X %*% B.true + rnorm(n)
B.guess = matrix(0,p+1, numY)
alpha=1/n
convergence.threshold=0.0001
penalty=0
min.iters=2
max.iters=200

gradient.descent(Y,X,B.guess,alpha,
  convergence.threshold, penalty,
  min.iters,max.iters, intercept=T)
lm(Y~X)$coef
