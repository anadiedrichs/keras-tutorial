# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* LSTM en KERAS https://keras.io/getting-started/sequential-model-guide/
* entrada o input multivariable http://stackoverflow.com/questions/40331510/how-to-stack-multiple-lstm-in-keras
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)
* recurrent layer in keras https://keras.io/layers/recurrent/
* Ejemplos tomados de este enlace http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/


### REGRESSION IN KERAS 

### ABOUT INPUT SHAPE

input_shape=(1, 1)

The first number is the number of timesteps the model should expect. Given only one timestep of data input the best possible approximation of any moving average is just to repeat the unique input value.

The second number is the number of inputs the model should expect at each timestep. This seems ok given your stated goal.

Your input has to be a list where each array has the batch size as the first dimension. Hence:

[(batch_size, 45, 8, 3),(batch_size, 45, 8, 3)...] 

### LAST LAYER + ACTIVATION

Whether or not you should use an Activation as the last layer, and what kind of activation, depends on the range of the values you want to output (for instance: if you want to output negative and positive values, don't use ReLU, etc. And never use softmax since it ouputs a probability distribution). 

If you aren't sure, it's probably better not to use an Activation as the last layer (Dense would then be the last layer).

Also, "show_accuracy" should not be set for a regression problem. The notion of accuracy only makes sense for a classification problem.

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact
