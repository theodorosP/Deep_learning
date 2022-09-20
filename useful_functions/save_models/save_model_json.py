def save_json_model(model_to_save):
  model_json = model_to_save.model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  model_to_save.model.save_weights("model.h5")
  print("Saved model to disk")
