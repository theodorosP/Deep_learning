import pandas as pd
import numpy as np
import keras.layers as keras_layers
import keras.models as keras_models
import keras.wrappers.scikit_learn as scikit_learn
import sklearn.model_selection as model_selection
from sklearn.preprocessing import StandardScaler

def rmse_new(list_1, list_2):
  mse = 0
  for i in range(len(list_1)):
    mse = mse + (list_1[i] - list_2[i])**2
  rmse = np.sqrt(mse/len(list_1))
  return rmse

def back_to_actual(list_1, mu, sigma):
  l = list()
  for i in range(len(list_1)):
    l.append(mu + list_1[i] * sigma)
  return l

def print_exp_pred(list_1, list_2):
  for i in range(len(list_1)):
    print("exp : ", round(list_1[i], 2), "pred : ", round(list_2[i], 2))


dataframe = pd.read_csv("BostonHousing.csv")
dataset = dataframe.values
x = dataset[:, 0 : len(dataframe.columns) - 1]
y = dataset[:, len(dataframe.columns) - 1]

scaler = StandardScaler()
standarized_data = scaler.fit_transform(dataframe)
mu = dataframe.mean()[len(dataframe.columns) - 1]
sigma = dataframe.std()[len(dataframe.columns) - 1]
input_number = len(dataframe.columns) - 1

x_standarized = standarized_data[:,0 : len(dataframe.columns) -1]
y_standarized = standarized_data[:, len(dataframe.columns) - 1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_standarized, y_standarized, test_size = 0.33)

def baseline_model():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(input_number, input_dim = input_number, kernel_initializer = "normal", activation = "relu")) 
  model.add(keras_layers.Dense(1, kernel_initializer = "normal"))
  model.compile(loss = "mean_squared_error", optimizer = "adam")
  return model

def model_1():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(input_number, input_dim = input_number, kernel_initializer = "normal", activation = "relu"))
  model.add(keras_layers.Dense(6, kernel_initializer = "normal", activation = "relu"))
  model.add(keras_layers.Dense(1, kernel_initializer = "normal"))
  model.compile(loss = "mean_squared_error", optimizer = "adam")
  return model

def model_2():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(20, input_dim = input_number, kernel_initializer = "normal", activation = "relu"))
  model.add(keras_layers.Dense(1, kernel_initializer = "normal"))
  model.compile(loss = "mean_squared_error", optimizer = "adam")
  return model


def model_3():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense((input_number + 1)/2, input_dim = input_number, kernel_initializer = "normal", activation = "relu")) 
  model.add(keras_layers.Dense(1, kernel_initializer = "normal"))
  model.compile(loss = "mean_squared_error", optimizer = "adam")
  return model



for i in [baseline_model, model_1, model_2, model_3]:
  model = scikit_learn.KerasRegressor(build_fn= i, nb_epoch=100, batch_size=5, verbose=False)
  kfold =  model_selection.KFold(n_splits = 10, shuffle = True)
  results = model_selection.cross_val_score(model, x_standarized, y_standarized, cv = kfold, scoring = "neg_root_mean_squared_error") 


  #use cros_val_predict for predictions
  predicted = model_selection.cross_val_predict(model, x_test, y_test, cv = 6)


  predicted_first_list = back_to_actual(predicted, mu, sigma)
  test_list = back_to_actual(y_test, mu, sigma)


  rmse_standarized_data = rmse_new(predicted, y_test)
  print(str(i) +" cros_val_predict method, rmse_standarized_data : ", rmse_standarized_data)

  rmse_actual = rmse_new(predicted_first_list, test_list)
  print(str(i) + " cros_val_predict method, rmse_actual_data : ",  rmse_actual)



  #use predict function for predictions
  model_baseline_fit = model.fit(x_train, y_train)
  score = model.score(x_test, y_test)
  predictions = model.predict(x_test)


  predicted_second_list = back_to_actual(predictions, mu, sigma)


  rmse_standarized_data = rmse_new(predictions, y_test)
  rmse_actual = rmse_new(predicted_second_list, test_list)

  print(str(i) + " predict function, rmse_standarized_data : ", rmse_standarized_data)
  print(str(i) + " predict function, rmse_actual_data : ", rmse_actual)
  print("---" * 100)



  #print_exp_pred(test_list, predicted_second_list)
  #print("---" * 100)
  #print_exp_pred(test_list, predicted_second_list)
