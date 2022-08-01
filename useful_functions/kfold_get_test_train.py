import sklearn.model_selection as model_selection
import keras.wrappers.scikit_learn as scikit_learn


def get_test_train():
  model = scikit_learn.KerasClassifier(build_fn = baseline_model, verbose = 1, nb_epoch = 30, batch_size = 5)
  kfold = model_selection.KFold(n_splits = 10, shuffle = True, ) 
  results = model_selection.cross_val_score(model, x, y, cv = kfold)
  for train, test in kfold.split(X):
    print("train = ", train)
    print("test = ", test)
