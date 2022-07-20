import pandas as pd
import numpy as np
import keras.models as keras_models
import keras.layers as keras_layers
import keras.wrappers.scikit_learn as scikit_learn
import sklearn.preprocessing as sklearn_preprocessing
import keras.utils as keras_utils
import sklearn.model_selection as model_selection

dataframe = pd.read_csv("sonar.csv", header = 0)
dataset = dataframe.values
x = dataset[:, 0 : 60]
y = dataset[:, 60]

encoder = sklearn_preprocessing.LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(Y)
dummy_y = keras_utils.np_utils.to_categorical(encoded_y)

def baseline_model():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(60, input_dim = 60, kernel_initializer = "normal", activation = "relu"))
  model.add(keras_layers.Dense(2, kernel_initializer = "normal", activation = "sigmoid"))
  model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
  return model

X = np.asarray(X).astype(np.float64)
dummy_Y = np.asarray(dummy_y).astype(np.float64)

model = scikit_learn.KerasClassifier(build_fn= baseline_model, verbose = 1, nb_epoch = 100, batch_size = 5 )
kfold = model_selection.KFold(n_splits=10, shuffle = True)
results = model_selection.cross_val_score(model, X, dummy_Y, cv = kfold)
