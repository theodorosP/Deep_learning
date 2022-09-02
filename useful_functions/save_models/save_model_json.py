def save_json_model(m):
  import keras.models as keras_models
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  w = model.save_weights("model.h5")
  return json_file, w
