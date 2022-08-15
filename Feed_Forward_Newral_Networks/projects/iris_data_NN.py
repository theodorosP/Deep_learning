import pandas as pd
import numpy as np
import keras.models as keras_models
import keras.layers as keras_layers
import keras.wrappers.scikit_learn as scikit_learn
import sklearn.model_selection as model_selection
import keras.utils as keras_utils
import sklearn.preprocessing as sklearn_preprocessing

dataframe = pd.read_csv("iris.csv")
dataset = dataframe.values
x = dataset[:, 0: 4]
y = dataset[:, 4]

encoder = sklearn_preprocessing.LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_y = keras_utils.np_utils.to_categorical(encoded_y)

def baseline_model():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(4, input_dim = 4, kernel_initializer= "normal", activation = "relu"))
  model.add(keras_layers.Dense(3, kernel_initializer = "normal", activation = "sigmoid"))
  model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
  return model

estimator1 = scikit_learn.KerasClassifier(build_fn = baseline_model, verbose = 1, nb_epoch = 200, batch_size = 5 )
kfold = model_selection.KFold(n_splits= 10, shuffle = True)

X=np.asarray(x).astype(np.float64)
dummy_Y =np.asarray(dummy_y).astype(np.float64)
print(x.dtype)
print(X.dtype)

results = model_selection.cross_val_score(estimator1, X, dummy_Y, cv=kfold)
