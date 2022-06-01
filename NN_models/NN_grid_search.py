import keras.models as keras_models
import keras.layers as keras_layers
import keras.wrappers.scikit_learn as scikit_learn
import sklearn.model_selection as model_selection
import numpy as np

def create_model(optimizer = "rmsprop", kernel_initializer = "glorot_uniform"):
        model = keras_models.Sequential()
        model.add(keras_layers.Dense(12, input_dim = 8, kernel_initializer = kernel_initializer, activation = "relu"))
        model.add(keras_layers.Dense(8, kernel_initializer = kernel_initializer, activation = "relu"))
        model.add(keras_layers.Dense(1, kernel_initializer = kernel_initializer, activation = "sigmoid"))
        model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
        return model

dataset = np.loadtxt("diabetes.csv", delimiter = ",", skiprows = 1)
x = dataset[:, 0 : 8]
y = dataset[:, 8]
model = scikit_learn.KerasClassifier(build_fn = create_model, verbose = 1)
epochs = [50, 100, 150]
batch_sizes = [5, 10, 20]
optimizers = [ "adam", "rmsprop" ]
kernel_initializers = [ "normal" , "uniform", "glorot_uniform"]
param_grid = dict(optimizer = optimizers, nb_epoch = epochs, batch_size = batch_sizes, kernel_initializer = kernel_initializers)
grid = model_selection.GridSearchCV(estimator = model, param_grid = param_grid)
grid_result = grid.fit(x, y)
print(grid_result)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']

for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))
