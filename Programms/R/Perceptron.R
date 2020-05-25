Perceptron <- function(pInputSize = 784, pOutputSize = 10, learningfactor = 0.01){
  envi <- environment()
  
  me <- NN()
  
  # data fields (private)
  inputSize <- pInputSize
  outputSize <- pOutputSize
  weights <- array(data = runif(inputSize * outputSize, min = -1), dim = c(outputSize, inputSize))
  
  # functions (the equals sign is mandatory)
    me$envi = envi
    
    me$getEnvironment = function(){
      return(get("envi",envi))
    }
    
    me$getInputSize = function(){
      return(get("inputSize",envi))
    }
    
    me$getOutputSize = function(){
      return(get("outputSize",envi))
    }
    
    me$evaluate = function(input) {
      if(!is.array(input)) stop("input must be an array")
      if(!identical( dim(input) , as.integer(c(inputSize)))) 
        stop(cat("Dimension mismatch in evaluate! Dimension of input needs to be", inputSize, "x 1"))
      output <- weights %*% input
      return(array(output))
    }
    
    me$train = function(input, output) {
      if(!is.array(input)) stop("input must be an array")
      if(!is.array(output)) stop("output must be an array")
      if(!identical( dim(input) , as.integer(c(inputSize)))) 
        stop(cat("Dimension mismatch in classify! Dimension of input needs to be", inputSize, "x 1"))
      if(!identical( dim(output) , as.integer(c(outputSize)))) 
        stop(cat("Dimension mismatch in classify! Dimension of input needs to be", inputSize, "x 1"))
      result <- me$evaluate(input)
      
      # Deltaregel
      d <- output - result
      gradient <- d %*% t(input) # t() transposes a matrix
      weights <- weights + learningfactor * gradient
    }
    
  assign('this', me, envir = envi)
  class(me) <- append(class(me), 'Perceptron')
  return(me)
}