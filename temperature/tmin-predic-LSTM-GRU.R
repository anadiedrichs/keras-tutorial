#' # Prediciendo temperatura minima con LSTM GRU
INIT <- Sys.time()
setwd("~/phd-repos/keras-tutorial/temperature")
#' Cargamos el dataset 
library(tibble)
library(readr)
source("models.R")
dacc_daily_tmin <- suppressWarnings(read_csv("~/phd-repos/tmin/bnlearn/data/dacc-daily-tmin.csv"))
head(dacc_daily_tmin)
#' Para el experimento inicial tomamos solo un sensor, una ubicación
#' 
datos <- dacc_daily_tmin[,3:9]
#' ¿Quito la radiación por tener valores perdidos o extraños?
datos[!complete.cases(datos[,4]),4]
#' No tengo datos perdidos, la podemos dejar.
#' Quitamos campo radiación para asemejar dataset a los experimentos con bnlearn.
# datos <- datos[-4]
#' Filas en el dataset
nrow(datos)

#' Graficamos la temperatura mínima del dataset, campo `junin.temp_min`
# library(ggplot2)
# ggplot(datos, aes(x = 1:nrow(datos), y = `junin.temp_min`)) + geom_line()

#' Convertimos a matrix los datos (floating-point matrix)
#' Quito columna Date
data <- data.matrix(datos)
#' Normalizing the data
train_data <- data
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

#' IMPORTANTE
#' Columna que nos interesa predecir. Importante, este variable es utilizada en varios lados.
predictor.target <<- "junin.temp_min"
predictor.colIndex <<- 6 # nro de columna en el dataset

source("generator.R")

#' ## Preparing the training, validation, and test generators
library(keras)
use_session_with_seed(42) # reproducible results
#' Observations will go back #lookback days
lookback <- 2
#' Observations will be sampled at one data point per hour.
step <- 1
#' Targets will be 24 hours in the future.
delay <- 0 # so generator take the next row as the next day
#delay <- 1
batch_size <- 32
MAX_INDEX_TRAIN_GEN <- 3762
MIN_INDEX_VAL_GEN <- 3763
MAX_INDEX_VAL_GEN <- 4000
MIN_INDEX_VAL_TEST <- 4001

train_gen <- generator(data,lookback = lookback, delay = delay,
  min_index = 1,max_index = MAX_INDEX_TRAIN_GEN,
  shuffle = FALSE, step = step, batch_size = batch_size)

val_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = MIN_INDEX_VAL_GEN,
  max_index = MAX_INDEX_VAL_GEN,
  step = step,
  batch_size = batch_size
)
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = MIN_INDEX_VAL_TEST,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)
val_steps <- (MAX_INDEX_VAL_GEN - MIN_INDEX_VAL_GEN - lookback) / batch_size
test_steps <- (nrow(data) - MIN_INDEX_VAL_TEST - lookback) / batch_size

  
#' ## Computing the common-sense baseline MAE
#' Simula un predictor aleatorio.
evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[,dim(samples)[[2]],predictor.colIndex]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}
evaluate_naive_method()
# Because the temperature data has been normalized to be
# centered on 0 and have a standard deviation of 1, this number isn’t immediately inter-
#  pretable. It translates to an average absolute error of 0.29 × temperature_std degrees
# Celsius: 
evalnaive <- evaluate_naive_method()*std[predictor.target]
#' ## Densenly connected model (neural network fully connected)
#' Training and evaluating a densely connected model
#' 
#' Vamos a probar el resultado de los modelos en el conjunto de testeo
#' 
#c(samples, targets) %<-% test_gen() #' cambian los datos regresados cada vez que lo llamo
print("======= DENSELY CONNECTED NETWORK =========")
model <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae",
  metrics = c("mean_squared_error")
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

#' Test set generation
tt <- get_test_set(data, lookback, delay, MIN_INDEX_VAL_TEST, NULL, shuffle = FALSE, batch_size = batch_size, step = step)

#' Ploting results
plot(history)
ggsave(paste("dacc-junin-Densely-radiacion-","history.png",sep=""))  
#evaluate model result
# bug reported on https://github.com/rstudio/keras/issues/414
#pred <- predict_generator(model,test_gen,steps = 10,verbose=1)

#pred <- model %>% predict(tt$samples)
print(model %>% evaluate_generator(test_gen,steps = step))
evaluation(model=model, 
           samples=tt$samples, 
           target=tt$target, 
           namePlot="dacc-junin-Densely-radiacion-") 
save_model_hdf5(model, "dacc-junin-Densely-radiacion.h5")

print("======= GRU =========")

#' ## [GRU] A first recurrent baseline
#' Training and evaluating a model with layer_gru
model <- keras_model_sequential() %>%
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae",
  metrics = c("mean_squared_error")
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

plot(history)
ggsave(paste("GRU-radiacion-","history.png",sep=""))  #' TODO SAVE MODEL!!!
evaluation(model=model, 
           samples=tt$samples, 
           target=tt$target, 
           namePlot="GRU-radiacion-") 
save_model_hdf5(model, "GRU-radiacion-.h5")

print("======= LSTM =========")

#' ## [LSTM] Training and evaluating a model with layer_lstm
#' 
model <- keras_model_sequential() %>%
  layer_lstm(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae",
  metrics = c("mean_squared_error")
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

plot(history)
ggsave(paste("LSTM-radiacion-","history.png",sep=""))  #' TODO SAVE MODEL!!!
evaluation(model=model, 
           samples=tt$samples, 
           target=tt$target, 
           namePlot="LSTM-radiacion-") 

save_model_hdf5(model, "LSTM-radiacion-.h5")

print("======= GRU + DROPOUT =========")

#' ## GRU + Dropout. Using recurrent dropout to fight overfitting
#' Training and evaluating a dropout-regularized GRU-based model
model <- keras_model_sequential() %>%
  layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae",
  metrics = c("mean_squared_error")
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)


plot(history)
ggsave(paste("GRU-dropout-radiacion-","history.png",sep=""))  #' TODO SAVE MODEL!!!
evaluation(model=model, 
           samples=tt$samples, 
           target=tt$target, 
           namePlot="GRU-dropout-radiacion-") 

save_model_hdf5(model, "GRU-dropout-radiacion-.h5")

print("======= GRU + GRU + DROPOUT =========")

#' ## GRU + GRU + dropout
#'  Training and evaluating a dropout-regularized, stacked GRU model

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
  loss = "mae",
  metrics = c("mean_squared_error")
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)


plot(history)
ggsave(paste("GRU-GRU-dropout-radiacion-","history.png",sep=""))  #' TODO SAVE MODEL!!!
evaluation(model=model, 
           samples=tt$samples, 
           target=tt$target, 
           namePlot="GRU-GRU-dropout-radiacion-") 
save_model_hdf5(model, "GRU-GRU-radiacion-.h5")

#' example to save and load a model
#' save_model_hdf5(model, "my_model.h5")
#' model <- load_model_hdf5("my_model.h5")

#' # TODO sensitivity, precision, recall, etc
#' 
#' # Going even further
#' 
#' There are many other things you could try, in order to improve performance on the
#' temperature-forecasting problem:
#' * Adjust the number of units in each recurrent layer in the stacked setup. The
#' current choices are largely arbitrary and thus probably suboptimal.
#' * Adjust the learning rate used by the RMSprop optimizer.
#' * Try using layer_lstm instead of layer_gru .
#' * Try using a bigger densely connected regressor on top of the recurrent layers:
#' that is, a bigger dense layer or even a stack of dense layers.
#' 
END <- Sys.time()

print(END-INIT)

# Time difference of 16.80466 mins