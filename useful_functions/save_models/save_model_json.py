def save_json_model(model):
  import keras.models as keras_models
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  model.save_weigths("weights.h5")
  return json_file, weights.h5
