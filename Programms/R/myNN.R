# inputsize = 784 (28*28)
Perceptron <- function(pInputSize = 784, pOutputSize = 10){
  envi <- environment()
  # data fields (private)
  inputSize <- pInputSize
  outputSize <- pOutputSize
  weights <- array(data = runif(inputSize * outputSize), dim = c(outputSize, inputSize))
  
  # functions
  me <- list(
    envi = envi,
    
    getEnvironment = function(){
      return(get("envi",envi))
    },
    
    getWeights = function(){
      return(get("weights",envi))
    },
    
    setWeights = function(value){
      if(!is.matrix(input)) stop("input must be a matrix")
      if(!identical( dim(value) , as.integer(c(outputSize, inputSize)))) 
        stop(cat("Dimension mismatch in setWeights! Dimension of weights needs to be", outputSize, "x", inputSize))
      return(assign("weights",value,envi))
    },
    
    getInputSize = function(){
      return(get("inputSize",envi))
    },
    
    getOutputSize = function(){
      return(get("outputSize",envi))
    },
    
    evaluate = function(input) {
      if(!is.matrix(input)) stop("input must be a matrix")
      if(!identical( dim(input) , as.integer(c(inputSize, 1)))) 
        stop(cat("Dimension mismatch in evaluate! Dimension of input needs to be", inputSize, "x 1"))
      output <- weights %*% input
      return(output)
    },
    
    classify = function(input) {
      if(!is.matrix(input)) stop("input must be a matrix")
      if(!identical( dim(input) , as.integer(c(inputSize, 1)))) 
        stop(cat("Dimension mismatch in classify! Dimension of input needs to be", inputSize, "x 1"))
      result <- me$evaluate(input)
      return(which.max(result))
    }
  )
    
  assign('this', me, envir = envi)
  class(me) <- append(class(me), 'Perceptron')
  return(me)
}



