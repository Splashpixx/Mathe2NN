library(e1071) # contains sigmoid. Please don't ask why it's called that, I don't know

FeedForward <- function(pInputSize = 784, pOutputSize = 10, numOfHiddenLayers = 1,
                       hiddenLayerSize = 100, activation = sigmoid, learningfactor = 0.01){
  envi <- environment()
  
  # data fields (private)
  inputSize <- pInputSize
  outputSize <- pOutputSize
  hiddenLayerWeights <- array(data = runif(hiddenLayerSize^2 * (numOfHiddenLayers - 1), min = -1),
                        dim = c(numOfHiddenLayers -1, hiddenLayerSize, hiddenLayerSize))
  intoHiddenWeights <- array(data = runif(inputSize * hiddenLayerSize, min = -1), dim = c(hiddenLayerSize, inputSize))
  outOfHiddenWeights <- array(data = runif(hiddenLayerSize * outputSize, min = -1), dim = c(outputSize, hiddenLayerSize))
  
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
      output <- intoHiddenWeights %*% input
      output <- activation(output)
      if(numOfHiddenLayers > 1){
        for(layer in 1:numOfHiddenLayers) {
          output <- hiddenLayerWeights[layer,,] %*% output
          output <- activation(output)
        }
      }
      output <- outOfHiddenWeights %*% output
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
      d_output <- array(dim = outputSize)
      d <- array(dim = c(numOfHiddenLayers, hiddenLayerSize))
      output_gradient <- array(dim = c(outputSize, hiddenLayerSize))
      input_gradient <- array(dim = c(hiddenLayerSize, inputSize))
      hidden_gradient <- array(dim = c(numOfHiddenLayers -1, hiddenLayerSize, hiddenLayerSize))
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