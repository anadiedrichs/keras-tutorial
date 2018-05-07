#' # Temperature forecasting with GRU
#' First steps, create an directory, download the dataset, unzip it.
#' Let's look the data 
library(tibble)
library(readr)
peach_four_sensors <- read_csv("~/phd-repos/RNN-keras-prediction/peach_four_sensors.csv")
#View(peach_four_sensors)
data <- peach_four_sensors
glimpse(data)
#' Plot the temperature °C
#' 
library(ggplot2)
ggplot(data, aes(x = 1:nrow(data), y = `00-17-0d-00-00-30-60-ef`)) + geom_line()
#' Converting the data into a floating-point matrix
#' Quito columna Date
data <- data.matrix(data[,-1])
#' Normalizing the data
train_data <- data[1:500,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)
#' ## GLOSARY
#' 
#' *Mini-batch or batch* A small set of samples (typically between 8 and 128) 
#' that are processed simultaneously by the model. The number of samples is
#' often a power of 2, to facilitate memory allocation on GPU. When training, a
#' mini-batch is used to compute a single gradient-descent update applied to
#' the weights of the model.
#' 
#' ## Generator yielding timeseries samples and their targets
#' 
#' *data* - The original array of floating-point data, which you normalized in listing 6.32.
#' 
#' *lookback* How many timesteps back the input data should go.
#' 
#' *delay*  How many timesteps in the future the target should be.
#' 
#' *min_index* and *max_index* —Indices in the data array that delimit which time-steps to draw from. This is useful for keeping a segment of the data for validation and another for testing.
#' 
#' *shuffle* —Whether to shuffle the samples or draw them in chronological order.
#' 
#' *batch_size* —The number of samples per batch.
#' 
#' *step* —The period, in timesteps, at which you sample data. You’ll set it to 6 in order to draw one data point every hour.
#' 
generator <- function(data, lookback, delay, min_index, max_index, shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index)) max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size, max_index))
      i <<- i + length(rows)
    }
    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]],
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }
    list(samples, targets)
  }
}
##################################
lookback <- 24*6 
#' Observations will be sampled at one data point per hour.
step <- 6
#' Targets will be 6 hours in the future.
delay <- 6 * 6
batch_size <- 128
# train_gen <- generator(
#   data,
#   lookback = lookback,
#   delay = delay,
#   min_index = 1,
#   max_index = 500,
#   shuffle = TRUE,
#   step = step,
#   batch_size = batch_size
# )
# val_gen = generator(
#   data,
#   lookback = lookback,
#   delay = delay,
#   min_index = 700,
#   max_index = 800,
#   step = step,
#   batch_size = batch_size
# )
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 797,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

test_gen()
