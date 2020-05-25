NN <- function(){
  envi <- environment()
  
  me <- list(
    
    evaluate = function(input){
      stop("Function not implemented")
    },
    
    train = function(input,output){
      stop("Function not implemented")
    },
    
    getInputSize = function(){
      stop("Function not implemented")
    },
    
    batchTrain = function(input,output){
      for(i in 1:dim(input)[1]){
        me$train(input[i,,],output[i,,])
      }
    },
    
    trainClassifyFromCSV = function(trainpath, testpath){
      train <- read.csv(trainpath)
      train_labels <- train[,1]
      train_array <- array(data = 0,dim = c(nrows(train_labels),10))
      for(i in 1:nrows(train_labels)){
        train_array[i,train_labels[i]] <- 1
      }
      train_inputs <- train[,2:ncol(train)]
      test <- read.csv(testpath)
      test_labels <- test[,1]
      test_array <- array(data = 0,dim = c(nrows(test_labels),10))
      for(i in 1:nrows(test_labels)){
        test_array[i,test_labels[i]] <- 1
      }
      test_inputs <- test[,2:ncol(test)]
      me$batchTrain(input = train_inputs, output = test_array)
      me$test(test_inputs, test_array)
    },
    
    classify = function(input) {
      if(!is.array(input)) stop("input must be an array")
      if(!identical( dim(input) , as.integer(c(inputSize)))) 
        stop(cat("Dimension mismatch in classify! Dimension of input needs to be", me$getInputSize(), "x 1"))
      result <- me$evaluate(input)
      return(which.max(result))
    },
    
    
    
    test = function(input, output){
      results <- array(dim = nrow(input))
      for(i in 1:nrow(input)){
        results[i] <- me$classify(input[i]) == output[i]
      }
      print(cat("accuracy is ", sum(results)/length(results)))
      return(sum(results)/length(results))
    },
    
  )
  assign('this', me, envir = envi)
  class(me) <- append(class(me), 'NN')
  return(me)
}