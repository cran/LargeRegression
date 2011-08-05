\name{LargeRegression-package}
\alias{LargeRegression-package}
\alias{LargeRegression}
\docType{package}
\title{
Large Regressions
}
\description{
Solve the least squares regression problem using gradient descent
}
\details{
\tabular{ll}{
Package: \tab LargeRegression\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2011-07-21\cr
License: \tab GPL\cr
LazyLoad: \tab yes\cr
Depends: Matrix
Suggests: cudaMatrixOps
}
Uses gradient descent to minimize the sum of squared residuals for the
regression problem.  Can include an L2 penalty on the coefficient matrix.
This function is very useful when there is an initial guess of what B should be
in Y = XB.  This guess may come from physical properties, intuition, or results
from previous regressions.  When new data is received, it can also be useful to
use results from an old regression as the B argument to update the coefficient 
matrix, rather than estimate a new regression from scratch.

In general, this functions performs faster than R's lm function, even when the
guess for B is bad.  Using GPU acceleration, this function can perform extremely
fast for multivariate regression problems with many response variables.  
}
\author{
Jeffrey Wong <jeff.ct.wong@stanford.edu>
}
\references{

}
\keyword{ lm }
\seealso{
fastVAR
}
\examples{

}
