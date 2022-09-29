def standarized_data(data_frame):
  import sklearn.preprocessing as sklearn_preprocessing
  scaler = sklearn_preprocessing.StandardScaler()
  n_of_cols = len(data_frame.columns)
  scaled_data = scaler.fit_transform(data_frame)
  x_scaled = scaled_data[:, 0 : n_of_cols - 1]
  y_scaled = scaled_data[:, n_of_cols -1]
  return x_scaled, y_scaled
