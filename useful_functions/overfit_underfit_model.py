def plot_overfit_under_fit(my_model):
  
  from keras.callbacks import History 
  import matplotlib.pyplot as plt

  history = model.fit(X_test, y_train, epochs = 40, batch_size = 5)
  acc = history.history.keys()
  val_loss = history.history["val_loss"]
  los = history.history["loss"]
  epochs = range(1, len(val_loss) + 1)

  plt.plot(epochs, val_loss)
  plt.plot(epochs, los)
  plt.show()
