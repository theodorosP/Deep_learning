def load_json_model():
  json_file = open( "model.json" , "r" )
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = keras_models.model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("model.h5")
  print("Loaded model from disk")
