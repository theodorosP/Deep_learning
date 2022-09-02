def load_model(file_1, file_2):
  json_file = open(file_1, "r")
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = keras_models.model_from_json(loaded_model_json)
  loaded_model.load_weights(file_2)
  return loaded_model
