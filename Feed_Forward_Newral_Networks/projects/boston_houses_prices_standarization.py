import pandas as pd
import numpy as np
import keras.models as keras_models
import keras.layers as keras_layers
import keras.wrappers.scikit_learn as scikit_learn
import sklearn.model_selection as model_selection
import sklearn.pipeline as sklearn_pipeline
from sklearn.preprocessing import StandardScaler


dataframe = pd.read_csv("BostonHousing.csv")


input_number = len(dataframe.columns) - 1

def baseline_model():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(input_number, input_dim = input_number, kernel_initializer = "normal", activation = "relu"))
  model.add(keras_layers.Dense(1, kernel_initializer = "normal"))
  model.compile(loss = "mean_squared_error", optimizer = "adam")
  return model


scaler = StandardScaler()
standardized_data = scaler.fit_transform(dataframe)
mu = dataframe.mean()[len(dataframe.columns) - 1]
sigma = dataframe.std()[len(dataframe.columns) - 1]


standardized_data_x = standardized_data[:, 0 : (len(dataframe.columns) - 1)]
standardized_data_y = standardized_data[:, len(dataframe.columns) - 1]



model = scikit_learn.KerasRegressor(build_fn = baseline_model, nb_epoch=100, batch_size=5, verbose=True)
kfold = model_selection.KFold(n_splits = 10, shuffle = True)
results = model_selection.cross_val_score(model, standardized_data_x, standardized_data_y, cv = kfold, scoring = "neg_root_mean_squared_error")



x_train, x_test, y_train, y_test = model_selection.train_test_split(standardized_data_x, standardized_data_y, test_size = 0.33) 

predicted = model_selection.cross_val_predict(model, x_test, y_test, cv = 6)


predicted_list = list()
y_test_list = list()

for i in range(len(predicted)):
  predicted_list.append(mu + predicted[i] *sigma)


for i in range(len(y_test)):
  y_test_list.append(mu + y_test[i] *sigma)

mse = 0
for i in range(len(predicted_list)):
  mse = mse + (predicted_list[i] - y_test_list[i])**2
  #print("pred: ", predicted_list[i], "exp :", y_test_list[i] )
rmse = np.sqrt(mse/len(predicted_list))
print("rmse = ", rmse)


model1 = model.fit(x_train, y_train)
score = model.score(x_test, y_test)
predictions = model.predict(x_test)

score_list = list()

for i in range(len(predictions)):
  score_list.append(mu + sigma*predictions[i])

mse_1 = 0
for i in range(len(score_list)):
  mse_1 = mse_1 + (score_list[i] - y_test_list[i])**2
print("mse = ", mse_1)
rmse_1 = np.sqrt(mse_1/len(score_list))
print("rmse = ", rmse_1)

for i in range(len(score_list)):
  print("exp :", y_test_list[i], "pred :", score_list[i])

  
