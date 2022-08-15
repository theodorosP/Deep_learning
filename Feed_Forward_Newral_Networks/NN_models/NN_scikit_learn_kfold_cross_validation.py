import keras.layers as keras_layers
import keras.models as keras_models
import numpy as np
import sklearn.model_selection as model_selection
import keras.wrappers.scikit_learn as scikit_learn
from sklearn import datasets, linear_model


def create_model():
        model = keras_models.Sequential()
        model.add(keras_layers.Dense(12, input_dim = 8 , kernel_initializer = "uniform", activation = "relu"))
        model.add(keras_layers.Dense(8, kernel_initializer = "uniform", activation = "relu"))
        model.add(keras_layers.Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
        model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
        return model

dataset = np.loadtxt("../diabetes.csv", delimiter = ",", skiprows = 1)
x = dataset[:, 0: 8]
y = dataset[:, 8]
model = scikit_learn.KerasClassifier(build_fn = create_model, epochs = 150, batch_size = 10)
kfold = model_selection.StratifiedKFold(n_splits = 10, shuffle = True)
results = model_selection.cross_val_score(model, x, y, cv = kfold)
print(results)
