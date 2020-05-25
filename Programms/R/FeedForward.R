library(e1071) # contains sigmoid. Please don't ask why it's called that, I don't know

FeedForward <- function(pInputSize = 784, pOutputSize = 10, numOfHiddenLayers = 1,
                       hiddenLayerSize = 100, activation = sigmoid, activation_derivative = dsigmoid, learningfactor = 0.01){
  if(hiddenLayerSize < 1){
    stop("hiddenLayerSize has to be 1 or greater")
  }
  envi <- environment()
  
  # data fields (private)
  inputSize <- pInputSize
  outputSize <- pOutputSize
  hiddenActivation <- array(dim = c(numOfHiddenLayers - 1, hiddenLayerSize))
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
      hiddenActivation[1,] <- activation(intoHiddenWeights %*% input)
      if(numOfHiddenLayers > 1){
        for(layer in 1:numOfHiddenLayers-1) {
          hiddenActivation[layer,] <- activation(hiddenLayerWeights[layer,,] %*% hiddenActivation[layer-1,])
        }
      }
      output <- activation(outOfHiddenWeights %*% hiddenActivation[numOfHiddenLayers,])
      hiddenActivation <- activation_derivative(hiddenActivation)
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
      
      d_output <- array(dim = outputSize)
      d <- array(dim = c(numOfHiddenLayers, hiddenLayerSize))
      outOfHiddenGradient <- array(dim = c(outputSize, hiddenLayerSize))
      intoHiddenGradient <- array(dim = c(hiddenLayerSize, inputSize))
      hiddenGradient <- array(dim = c(numOfHiddenLayers -1, hiddenLayerSize, hiddenLayerSize))
      
      # find d for the outputneurons
      d_output <- activation_derivative(output - result)

      # find d for the last hidden layer
      for(hiddenNeuron in 1:hiddenLayerSize){
        d[numOfHiddenLayers - 1, hiddenNeuron] <- outOfHiddenWeights %*% d_output
      }
      
      #find d for the other layers
      if(numOfHiddenLayers > 1){
        for(layer in numOfHiddenLayers-2:1){
          for(neuron in 1:hiddenLayerSize){
            d[layer, neuron] <- hiddenLayerWeights[layer + 1,,] %*% d[layer + 1,]
          }
        }
      }
      
      outOfHiddenGradient <- d_output %*% t(hiddenActivation[numOfHiddenLayers,])
      outOfHiddenWeights <- outOfHiddenWeights + learningfactor * outOfHiddenGradient
      
      if(numOfHiddenLayers > 1){
        for(layer in numOfHiddenLayers-1:2){
          hiddenGradient[layer,] <- d[layer,] %*% t(hiddenActivation[layer -1,])
        }
      }
      hiddenLayerWeights <- hiddenLayerWeights + learningfactor * hiddenGradient
      
      intoHiddenGradient <- d[1,] %*% t(input)
      intoHiddenWeights <- intoHiddenWeights + learningfactor * intoHiddenGradient
    }
    
  )
  
  assign('this', me, envir = envi)
  class(me) <- append(class(me), 'Perceptron')
  return(me)
}