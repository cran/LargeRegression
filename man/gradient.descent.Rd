\name{gradient.descent}
\alias{gradient.descent}
\title{
Least Squares using Gradient Descent
}
\description{
Solve the least squares problem using gradient descent
}
\usage{
gradient.descent(Y, X, B = matrix(0, ncol(as.matrix(X))+1, ncol(as.matrix(Y))), 
  alpha = 1/nrow(Y), convergence.threshold = 1e-04, penalty = 0, min.iters = 0, 
  max.iters = 200, gpu = F, intercept = F, verbose = T)
}
\arguments{
  \item{Y}{
The response vector or matrix
}
  \item{X}{
The data matrix
}
  \item{B}{
Initial guess of the coefficient matrix for Y = XB
}
  \item{alpha}{
The learning rate.  Decrease this for greater numerical stability
}
  \item{convergence.threshold}{
Optional.  Determines when the algorithm has converged.  If 
sum((B.update - B)^2) is less than convergence.threshold, then the
algorithm has converged
}
  \item{penalty}{
An L2 penalty to be used.  A value of 0 is equivalent to the standard OLS 
regression problem
}
  \item{min.iters}{
The minimum number of iterations before the algorithm is allowed to converge
}
  \item{max.iters}{
The maximum number of iterations before the algorithm is aborted
}
  \item{gpu}{
If TRUE, use the R package cudaMatrixOps to use GPU acceleration
}
  \item{intercept}{
Logical.  If true, append a column vector of 1's to the data matrix.  This
represents the intercept term in the regression
}
  \item{verbose}{
Logical.  If true, report the algorithm's progress
}
}
\details{
An iterative approach to solving the least squares problem.  Minimizes
the sum of squared errors using gradient descent.  Capable of ridge regression
where an L2 penalty is applied to the coefficient matrix.  

This function is particularly useful if there is a guess of what the coefficient
matrix should be, either from physical properties, intuition, or from results 
from previous regressions.  Generally, this function performs faster than R's 
lm function.

GPU acceleration can be used by installing the cudaMatrixOps package and setting
gpu to TRUE.  For multivariate regressions with many response variables, gpu
acceleration greatly increases the performance.
}
\value{
The estimated B matrix
}
\references{
}
\author{
Jeffrey Wong
}
\note{
}

\section{Warnings }{It is possible that the regression results from this
function are numerically unstable.  This depends on the learning rate of the
algorithm, which can be decreased for greater stability}  

\seealso{
}
\examples{
}
\keyword{ lm }
