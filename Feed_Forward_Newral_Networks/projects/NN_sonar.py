from matplotlib.cbook import flatten
import pandas as pd
import numpy as np
import keras.models as keras_models
import keras.layers as keras_layers
import keras.wrappers.scikit_learn as scikit_learn
import sklearn.preprocessing as sklearn_preprocessing
import keras.utils as keras_utils
import sklearn.model_selection as model_selection
from keras.callbacks import EarlyStopping, History 
import matplotlib.pyplot as plt



dataframe = pd.read_csv("sonar.csv", header = None)
dataset = dataframe.values
x = dataset[:, 0 : 60]
y = dataset[:, 60]

encoder = sklearn_preprocessing.LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_y = keras_utils.np_utils.to_categorical(encoded_y)

def baseline_model():
  model = keras_models.Sequential()
  model.add(keras_layers.Dense(60, input_dim = 60, kernel_initializer = "normal", activation = "relu"))
  model.add(keras_layers.Dense(2, kernel_initializer = "normal", activation = "sigmoid"))
  model.add(keras_layers.Flatten())
  model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
  return model

X = np.asarray(x).astype(np.float64)
dummy_Y = np.asarray(dummy_y).astype(np.float64)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, dummy_y, test_size=0.33)

print(len(y_test))

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model = scikit_learn.KerasClassifier(build_fn= baseline_model, verbose = 1, nb_epoch = 0, batch_size = 5, validation_split=0.33 )
kfold = model_selection.KFold(n_splits=10, shuffle = True)
results = model_selection.cross_val_score(model, X_train, y_train, cv = kfold)
print(results.mean())



y_pred = model_selection.cross_val_predict(model, X_test, y_test, cv=10)


print(y_pred)
y_pred_list = y_pred.tolist()

y_test_list = y_test.tolist()


for i in range(len(y_pred_list)):
  if y_pred_list[i] == 1:
    y_pred_list[i] = [1.0, 0.0]
  else:
    y_pred_list[i] = [0.0, 1.0]
  
print(y_pred_list)
print("----" * 50)
print(y_test_list)

s = 0
for i in range(len(y_pred_list)):
  if y_pred_list[i] == y_test_list[i]:
    s = s + 1

accuracy = s / len(y_pred_list)
print("accuracy = ", accuracy * 100, "%")


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

history = model.fit(X_test, y_train, epochs = 40, batch_size = 5, callbacks=[es])
acc = history.history.keys()
val_loss = history.history["val_loss"]
los = history.history["loss"]
print(val_loss)

epochs = range(1, len(val_loss) + 1)


plt.plot(epochs, val_loss, label = "val_loss")
plt.plot(epochs, los, label = "los")
plt.legend(loc = "best")
plt.show()
