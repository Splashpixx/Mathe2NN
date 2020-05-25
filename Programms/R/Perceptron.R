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
      if(!identical( dim(input) , as.integer(c(inputSize)))) 
        stop(cat("Dimension mismatch in evaluate! Dimension of input needs to be", inputSize, "x 1"))
      output <- weights %*% input
      return(output)
    },
    
    classify = function(input) {
      if(!is.matrix(input)) stop("input must be a matrix")
      if(!identical( dim(input) , as.integer(c(inputSize)))) 
        stop(cat("Dimension mismatch in classify! Dimension of input needs to be", inputSize, "x 1"))
      result <- me$evaluate(input)
      return(which.max(result))
    },
    
    train = function(input, output) {
      if(!is.matrix(inputData)) stop("inputData must be a matrix")
      if(!is.matrix(outputData)) stop("outputData must be a matrix")
      if(!identical( dim(inputData) , as.integer(c(inputSize)))) 
        stop(cat("Dimension mismatch in classify! Dimension of inputData needs to be", inputSize, "x 1"))
      if(!identical( dim(outputData) , as.integer(c(outputSize)))) 
        stop(cat("Dimension mismatch in classify! Dimension of inputData needs to be", inputSize, "x 1"))
      result <- me$evaluate(inputData)
      
      # Deltaregel
      gradient <- array(dim = c(outputSize, inputSize))
      for(outputNeuron in 1:nrow(gradient)){
        for(inputNeuron in 1:ncol(gradient)){
          # \Delta w_ik = e * d_i * a_k
          # e is out of loop
          d <- output[outputNeuron] - result[outputNeuron]
          gradient[outputNeuron, inputNeuron] <- d * input[inputNeuron]
        }
      }
      weights <- weights + learnfactor * gradient
    }
    
  )
    
  assign('this', me, envir = envi)
  class(me) <- append(class(me), 'Perceptron')
  return(me)
}