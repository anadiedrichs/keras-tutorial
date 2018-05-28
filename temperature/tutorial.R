#' # Temperature forecasting with GRU
#' First steps, create an directory, download the dataset, unzip it.
dir.create("~/Downloads/jena_climate", recursive = TRUE)
download.file(
  "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
  "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip"
)
unzip(
  "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip",
  exdir = "~/Downloads/jena_climate"
)

#' Let's look the data 
library(tibble)
library(readr)
data_dir <- "~/Downloads/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)
glimpse(data)
#' Plot the temperature °C
#' 
library(ggplot2)
ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()
#' Plotting the first 10 days of the temperature timeseries
ggplot(data[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()
#' Converting the data into a floating-point matrix
#' Quito columna Date
data <- data.matrix(data[,-1])
#' Normalizing the data
train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)
#' Columna que nos interesa predecir
predictor.target <<- "T (degC)"


#' ## GLOSARY
#' 
#' *Mini-batch or batch* A small set of samples (typically between 8 and 128) 
#' that are processed simultaneously by the model. The number of samples is
#' often a power of 2, to facilitate memory allocation on GPU. When training, a
#' mini-batch is used to compute a single gradient-descent update applied to
#' the weights of the model.
#' 
#' ## Generator yielding timeseries samples and their targets
#' *data* - The original array of floating-point data, which you normalized in listing 6.32.
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
      targets[[j]] <- data[rows[[j]] + delay,predictor.target]
    }
    list(samples, targets)
  }
}
#' The i variable contains the state that tracks the next window of data to return, so it’s
#' updated using superassignment ( i <<- i + length(rows) ).
#' 
#' ## Preparing the training, validation, and test generators
library(keras)
#' Observations will go back 10 days
lookback <- 1440 
#' Observations will be sampled at one data point per hour.
step <- 6
#' Targets will be 24 hours in the future.
delay <- 144
batch_size <- 128
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step,
  batch_size = batch_size
)
val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)
val_steps <- (300000 - 200001 - lookback) / batch_size
test_steps <- (nrow(data) - 300001 - lookback) / batch_size

#' ## Computing the common-sense baseline MAE
#' 
evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}
evaluate_naive_method()

#' Training and evaluating a densely connected model
#' 
model <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)
#' Ploting results
plot(history)  

#' ## A first recurrent baseline
#' Training and evaluating a model with layer_gru
model <- keras_model_sequential() %>%
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)  #<-----------------

#' ## Using recurrent dropout to fight overfitting
#' Training and evaluating a dropout-regularized GRU-based model
model <- keras_model_sequential() %>%
  layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)

#' Training and evaluating a ropout-regularized, stacked GRU model

model <- keras_model_sequential() %>%
  layer_gru(units = 32,
            dropout = 0.1,
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_gru(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)