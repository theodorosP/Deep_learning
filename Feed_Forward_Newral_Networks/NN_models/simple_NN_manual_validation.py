import keras.models as keras_models
import keras.layers as keras_layers
import numpy as np
import sklearn.model_selection as skms

dataset = np.loadtxt("../diabetes.csv", delimiter = ",", skiprows = 1)
x = dataset[:, 0 :8]
y = dataset[:, 8]
print(x)
print(y)

x_train, x_test, y_train, y_test = skms.train_test_split(x, y, test_size = 0.33)
model = keras_models.Sequential()
model.add(keras_layers.Dense(12, input_dim = 8, kernel_initializer = "uniform", activation = "relu"))
model.add(keras_layers.Dense(8, kernel_initializer = "uniform", activation = "relu"))
model.add(keras_layers.Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 500, batch_size = 10)
scores = model.evaluate(x, y)
print(scores)
print(model.metrics_names)

