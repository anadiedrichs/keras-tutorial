#' ## GLOSARY
#' 
#' *Mini-batch or batch* A small set of samples (typically between 8 and 128) 
#' that are processed simultaneously by the model. The number of samples is
#' often a power of 2, to facilitate memory allocation on GPU. When training, a
#' mini-batch is used to compute a single gradient-descent update applied to
#' the weights of the model.
#' 
#' ## Generator yielding timeseries samples and their targets
#' *data* - The original array of floating-point data, which you normalized before
#' *lookback* How many timesteps back the input data should go.
#' *delay*  How many timesteps in the future the target should be.
#' *min_index* and *max_index* —Indices in the data array that delimit which time-steps to draw from. This is useful for keeping a segment of the data for validation and another for testing.
#' *shuffle* —Whether to shuffle the samples or draw them in chronological order.
#' *batch_size* —The number of samples per batch.
#' *step* —The period, in timesteps, at which you sample data. You’ll set it to 6 in order to draw one data point every hour.



generator <- function(data, lookback, delay, min_index, max_index, shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index)) max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size, max_index)) # <--- POR QUE??
      i <<- i + length(rows)
    }
    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,predictor.target]
    }
    list(samples, targets)
  }
}

get_test_set <- function(data, lookback, delay, min_index, max_index, shuffle = FALSE, batch_size = 128, step = 6) {
  
  if (is.null(max_index)) max_index <- nrow(data) - delay - 1
  
  m <- min_index + lookback
  

  if (m + batch_size >= max_index)
    m <- min_index + lookback
  
  rows <- c(m:max_index)
  
    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,predictor.target]
    }
    return(list(samples=samples, target=targets))
  
}