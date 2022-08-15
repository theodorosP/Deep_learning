import pandas as pd
import numpy as np
import keras.models as keras_models
import keras.layers as keras_layers
import keras.wrappers.scikit_learn as scikit_learn
import sklearn.model_selection as model_selection

dataframe = pd.read_csv("BostonHousing.csv")
dataset = dataframe.values
x = dataset[:, 0 : len(dataframe.columns) - 1]
y = dataset[:, len(dataframe.columns) - 1]

input_parameters = len(dataframe.columns) - 1

def baseline_model():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(input_parameters, input_dim = input_parameters, kernel_initializer = "normal", activation = "relu"))
  model.add(keras_layers.Dense(1, kernel_initializer = "normal"))
  model.compile(loss = "mean_squared_error", optimizer = "adam")
  return model

model = scikit_learn.KerasRegressor(build_fn = baseline_model, nb_epoch = 100, batch_size = 5, verbose = True)
kfold = model_selection.KFold(n_splits = 10, shuffle = True)
results = model_selection.cross_val_score(model, x, y, cv = kfold, scoring = "neg_root_mean_squared_error")
print(results)
print(results.mean())


#use .cross_val_predict to check how the neural network acts with the tested data.

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size = 0.33)

y_pred = model_selection.cross_val_predict(model, X_test, Y_test, cv = 6)


mse = 0
for i in range(len(y_pred)):
  mse = mse + (Y_train[i] - y_pred[i])**2
rmse = np.sqrt(mse/len(y_pred))
print("rmse_cross_val_predict = ", rmse)

print("--" * 100)
#use predict to check how the neural network acts with the tested data.

model_fit = model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)
predictions = model.predict(X_test)


mse = 0
for i in range(len(predictions)):
  mse = mse + (Y_train[i] - predictions[i])**2
rmse = np.sqrt(mse/len(predictions))
print("rmse_predict = ", rmse)




