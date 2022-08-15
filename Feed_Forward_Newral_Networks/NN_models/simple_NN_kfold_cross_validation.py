import keras.models as keras_models
import keras.layers as keras_layers
import sklearn.model_selection as model_selection
import numpy as np

dataset = np.loadtxt("../diabetes.csv", delimiter = ",", skiprows = 1)
x = dataset[:, 0:8]
y = dataset[:, 8]
kfold = model_selection.StratifiedKFold(n_splits = 10, shuffle = True)
cvscores = list()
split = kfold.split(x, y)

for train, test in split:
        model = keras_models.Sequential()
        model.add(keras_layers.Dense(12, input_dim = 8, kernel_initializer = "uniform", activation = "relu" ))
        model.add(keras_layers.Dense(8, kernel_initializer = "uniform", activation = "relu"))
        model.add(keras_layers.Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
        model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
        model.fit(x[train], y[train], epochs = 500, batch_size = 10)
        scores = model.evaluate(x[test], y[test])
        cvscores.append(scores[1])
for i in range(len(cvscores)):
        print(i, cvscores[i] * 100, " %" )

print(np.mean(cvscores)*100, " %")
print("+- ", np.std(cvscores))
