def plot_overfit_under_fit(my_model):
  
  from keras.callbacks import History, EarlyStopping
  import matplotlib.pyplot as plt

  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

  #es = EarlyStopping(monitor='val_loss', mode='min', verbose=4)

  history = model.fit(X_train, y_train, epochs = 20, batch_size = 50, callbacks=[es])
  
  acc = history.history.keys()
  val_loss = history.history["val_loss"]
  los = history.history["loss"]
  epochs = range(1, len(val_loss) + 1)

  plt.plot(epochs, val_loss)
  plt.plot(epochs, los)
  plt.show()
