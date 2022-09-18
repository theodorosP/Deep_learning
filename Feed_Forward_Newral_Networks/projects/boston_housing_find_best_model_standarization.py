from pandas.core.common import standardize_mapping
import pandas as pd
import keras.models as keras_models
import keras.layers as keras_layers
import keras.wrappers.scikit_learn as scikit_learn
import sklearn.model_selection as model_selection
import sklearn.preprocessing as sklearn_preprocessing
import numpy as np
from keras.callbacks import History, EarlyStopping
import json


dataframe = pd.read_csv("BostonHousing.csv")
dataset = dataframe.values 
number_of_columns = len(dataframe.columns) 
scaler = sklearn_preprocessing.StandardScaler()
scaled_data = scaler.fit_transform(dataframe)
x_scaled = scaled_data[:, 0 : number_of_columns - 1]
y_scaled = scaled_data[:, number_of_columns -1]

mu = dataframe.mean()[number_of_columns - 1]
sigma = dataframe.std()[number_of_columns - 1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_scaled, y_scaled, test_size = 0.33)

def create_model(optimizer = "rmsprop", loss ="mse", kernel_initializer = "uniform"):
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(64, input_dim = number_of_columns - 1, kernel_initializer = "uniform", activation = "relu", ))
  model.add(keras_layers.Dense(64, kernel_initializer = "uniform", activation = "relu"))
  model.add(keras_layers.Dense(1, kernel_initializer = "uniform"))
  model.compile(optimizer = "rmsprop", loss = "mse", metrics = ["mae"])
  return model



model = scikit_learn.KerasRegressor(build_fn = create_model, verbose = True)
epochs = [50, 100, 150]
batch_sizes = [5, 10, 20]
optimizers = [ "adam", "rmsprop", "sgd", "adadelta", "adagrad", "adamax", "nadam"] 
kernel_initializers = [ "normal", "uniform", "glorot_uniform"]
param_grid = dict(optimizer = optimizers, nb_epoch = epochs, batch_size = batch_sizes, kernel_initializer = kernel_initializers)
grid = model_selection.GridSearchCV(estimator = model, param_grid = param_grid)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
grid_result = grid.fit(x_train, y_train)


means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']

for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))


print(grid_result)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


print(grid_result)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']

for mean,param in zip(means,params):
    with open("best_params.txt", "w") as file:
      file.write(str(means) + str(params))


with open  ("best_params.txt", "a") as file:
  file.write("\n \n Best model: \n \n" + "best score: " + str(round(grid_result.best_score_, 3)) + json.dumps(dict(grid_result.best_params_))) 



def create_model():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(64, input_dim = number_of_columns - 1, kernel_initializer = "uniform", activation = "relu", ))
  model.add(keras_layers.Dense(64, kernel_initializer = "uniform", activation = "relu"))
  model.add(keras_layers.Dense(1, kernel_initializer = "uniform"))
  model.compile(optimizer = "adam", loss = "mse", metrics = ["mae"])
  return model

model = scikit_learn.KerasRegressor(build_fn = create_model, batch_size = 5, nb_epoch = 50, verbose = True)
kfold = model_selection.KFold(n_splits = 10, shuffle = True)
results = model_selection.cross_val_score(model, x_train, y_train , cv = kfold)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
predictions = model.predict(x_test)

def mae(l1, l2):
  s = 0
  for i in range(len(l1)):
    s = s + np.abs(l1[i] - l2[i])
  MAE = s/len(l1)
  print("MAE = ", MAE)
  return MAE

mae(y_test, predictions)

def back_to_actual_data(data, mu, sigma):
  l3 = list()
  for i in range(len(data)):
    l3.append(mu + data[i] * sigma)
  return l3

actual = back_to_actual_data(y_test, mu, sigma)
predicted = back_to_actual_data(predictions, mu, sigma)

print(y_test)
print(predictions)

for i in range(len(actual)):
  print("expected : ", round(actual[i], 2), " predicted : ", round(predicted[i], 2))
