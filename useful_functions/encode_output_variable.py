import numpy as np
import sklearn.preprocessing as sklearn_preprocessing
import keras.utils as keras_utils

encoder = sklearn_preprocessing.LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = keras_utils.np_utils.to_categorical(encoded_Y)

print(dummy_y)
