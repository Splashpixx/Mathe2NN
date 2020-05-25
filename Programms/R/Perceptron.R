Perceptron <- function(pInputSize = 784, pOutputSize = 10, learningfactor = 0.01){
  envi <- environment()
  
  # data fields (private)
  inputSize <- pInputSize
  outputSize <- pOutputSize
  weights <- array(data = runif(inputSize * outputSize, min = -1), dim = c(outputSize, inputSize))
  
  # functions (the equals sign is mandatory)
  me <- list(
    envi = envi,
    
    getEnvironment = function(){
      return(get("envi",envi))
    },
    
    getInputSize = function(){
      return(get("inputSize",envi))
    },
    
    getOutputSize = function(){
      return(get("outputSize",envi))
    },
    
    evaluate = function(input) {
      if(!is.array(input)) stop("input must be an array")
      if(!identical( dim(input) , as.integer(c(inputSize)))) 
        stop(cat("Dimension mismatch in evaluate! Dimension of input needs to be", inputSize, "x 1"))
      output <- weights %*% input
      return(output)
    },
    
    classify = function(input) {
      if(!is.array(input)) stop("input must be an array")
      if(!identical( dim(input) , as.integer(c(inputSize)))) 
        stop(cat("Dimension mismatch in classify! Dimension of input needs to be", inputSize, "x 1"))
      result <- me$evaluate(input)
      return(which.max(result))
    },
    
    train = function(input, output) {
      if(!is.array(inputData)) stop("inputData must be an array")
      if(!is.array(outputData)) stop("outputData must be an array")
      if(!identical( dim(inputData) , as.integer(c(inputSize)))) 
        stop(cat("Dimension mismatch in classify! Dimension of inputData needs to be", inputSize, "x 1"))
      if(!identical( dim(outputData) , as.integer(c(outputSize)))) 
        stop(cat("Dimension mismatch in classify! Dimension of inputData needs to be", inputSize, "x 1"))
      result <- me$evaluate(inputData)
      
      # Deltaregel
      gradient <- array(dim = c(outputSize, inputSize))
      d <- output - result
      for(outputNeuron in 1:nrow(gradient)){
        for(inputNeuron in 1:ncol(gradient)){
          # \Delta w_ik = e * d_i * a_k
          # e is out of loop
          gradient[outputNeuron, inputNeuron] <- d[outputNeuron] * input[inputNeuron]
        }
      }
      weights <- weights + learningfactor * gradient
    }
    
  )
    
  assign('this', me, envir = envi)
  class(me) <- append(class(me), 'Perceptron')
  return(me)
}