import pandas as pd
import numpy as np
import keras.layers as keras_layers
import keras.models as keras_models
import keras.wrappers.scikit_learn as scikit_learn
import sklearn.model_selection as model_selection
import sklearn.preprocessing as sklearn_preprocessing
import sklearn.pipeline as sklearn_pipeline
import keras.utils as keras_utils



def fit(x, y):
  X = np.asarray(x).astype(np.float64)
  Y = np.asarray(y).astype(np.float64)
  return X, Y

def flat_list(list_to_flat):
  l = list()
  for i in list_to_flat:
    for j in i:
      l.append(j)
      return j

dataframe = pd.read_csv("sonar.csv", header = None)
dataset = dataframe.values
x = dataset[:, 0 : len(dataframe.columns) - 1]
y = dataset[:, len(dataframe.columns) - 1]

for i in range(len(y)):
  if y[i] == "R":
    y[i] = 0
  else:
    y[i] = 1


X = fit(x, y)[0]
dummy_Y = fit(x, y)[1]
input = len(dataframe.columns) - 1

def baseline_model():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(input, input_dim = input, kernel_initializer = "normal", activation = "relu"))
  model.add(keras_layers.Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
  model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
  return model


estimators = list()
estimators.append(( "standardize", sklearn_preprocessing.StandardScaler()))
estimators.append(("mlp", scikit_learn.KerasClassifier(build_fn = baseline_model, nb_epoch = 100, batch_size = 5, verbose = 1)))
pipeline = sklearn_pipeline.Pipeline(estimators)

pipeline = sklearn_pipeline.Pipeline(estimators)
kfold = model_selection.KFold(n_splits = 10, shuffle = True)
results = model_selection.cross_val_score(pipeline, X, dummy_Y, cv = kfold)

print(results)
print(results.mean())




#use cross_val_predict to check how the neural network acts with the tested data.

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, dummy_Y, test_size=0.33)
y_pred = model_selection.cross_val_predict(pipeline, X_test, y_test, cv=6)

y_pred_list = y_pred.tolist()

y_test_list = y_test.tolist()

predictions = list()
for i in y_pred_list:
  for j in i:
    predictions.append(j)

print(predictions)

s = 0
for i in range(len(predictions)):
  if predictions[i] == y_test_list[i]:
    s = s + 1

accuracy = s / len(y_pred_list)
print("accuracy = ", accuracy * 100, "%")


#use predict to check how the neural network acts with the tested data.


model = pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
predictions = model.predict(X_test)
print("score = ", round(score * 100, 3) , " %")

pred_list = predictions.tolist()
y_test_list = y_test.tolist()

predictions = list()
for i in pred_list:
  for j in i:
    predictions.append(j)

s = 0
for i in range(len(predictions)):
  if predictions[i] == y_test_list[i]:
    s = s + 1


accuracy = s / len(pred_list)
print("accuracy = ", accuracy * 100, "%")

