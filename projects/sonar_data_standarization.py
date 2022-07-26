import pandas as pd
import numpy as np
import keras.models as keras_models
import keras.layers as keras_layers
import keras.wrappers.scikit_learn as scikit_learn
import sklearn.model_selection as model_selection
import sklearn.preprocessing as sklearn_preprocessing
import sklearn.pipeline as sklearn_pipeline
import keras.utils as keras_utils

dataframe = pd.read_csv("sonar.csv", header = None)
dataset = dataframe.values
x = dataset[:,0 : 60]
y = dataset[:, 60]

def encode_data(y):
  encoder = sklearn_preprocessing.LabelEncoder()
  encoder.fit(y)
  encoded_Y = encoder.transform(y)
  # convert integers to dummy variables (i.e. one hot encoded)
  dummy_y = keras_utils.np_utils.to_categorical(encoded_Y)
  return dummy_y

dummy_y = encode_data(y)

def baseline_model():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(60, input_dim = 60, kernel_initializer = "normal", activation = "relu"))
  model.add(keras_layers.Dense(2, kernel_initializer = "normal", activation = "sigmoid"))
  model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
  return model

X = np.asarray(x).astype(np.float64)
dummy_Y = np.asarray(dummy_y).astype(np.float64)

estimators = list()
estimators.append(( "standardize", sklearn_preprocessing.StandardScaler()))
estimators.append(("mlp", scikit_learn.KerasClassifier(build_fn = baseline_model, nb_epoch = 100, batch_size = 5, verbose = 1)))
pipeline = sklearn_pipeline.Pipeline(estimators)
kfold = model_selection.KFold(n_splits = 10, shuffle = True)
results = model_selection.cross_val_score(pipeline, X, dummy_Y, cv = kfold)
print(results.mean())
