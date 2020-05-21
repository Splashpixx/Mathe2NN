# inputsize = 784 (28*28)

weights <- array(data = runif(784*10), dim = c(10,784))

evaluate <- function(input) {
  if(!is.matrix(input)) stop("input must be a matrix")
  if(length(input) != 784) stop("input must have a length of 784")
  output <- weights %*% input
  return(output)
}

classify <- function(input) {
  if(!is.matrix(input)) stop("input must be a matrix")
  if(length(input) != 784) stop("input must have a length of 784")
  result <- evaluate(input)
  return(which.max(result))
}