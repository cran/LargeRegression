#Last updated Aug 5

gradient.descent = function(Y, X, B=matrix(0, ncol(as.matrix(X))+1, 
  ncol(as.matrix(Y))), alpha=1/nrow(Y), convergence.threshold=0.0001, penalty=0, 
  min.iters=0, max.iters=200, gpu=F, intercept = F, verbose=T) {

  if(!is.matrix(Y)) Y = as.matrix(Y)
  if(!is.matrix(X)) X = as.matrix(X)
  if(intercept) {
    design = cbind(1,X)
  } else {
    design = X
  }

  n = nrow(Y)
  p = ncol(Y)

  if(gpu & require(cudaMatrixOps)) {
    XTX = gpuCrossprod(design)
    XTY = gpuCrossprod(design,Y)
    for(iter in 1:max.iters) {
      update.step = XTY - gpuMatMult(XTX,B) + penalty * B
      B.new = B + alpha * update.step

      if(sum((B.new - B)^2) < (convergence.threshold * p) &
        (iter > min.iters)) {
        if(verbose) print(paste("Converged on iteration",iter))
        return(B.new)
      } else {
        if(verbose) print(paste("End iteration",iter))
        B = B.new
      }
    }
  } else {
    XTX = crossprod(design)
    XTY = crossprod(design,Y)
    for(iter in 1:max.iters) {
      update.step = XTY - XTX%*%B + penalty * B
      B.new = B + alpha * update.step

      if(sum((B.new - B)^2) < (convergence.threshold * p) & 
        (iter > min.iters)) {
        if(verbose) print(paste("Converged on iteration",iter))
        return(B.new)
      } else {
        if(verbose) print(paste("End iteration",iter))
        B = B.new
      }
    }
  }
  if(verbose) print("Did not converge")
  return (B)
}
