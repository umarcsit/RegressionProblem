from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf  # Importing TensorFlow for Huber loss
import numpy as np
from graphs.error_graph import Plt
def stack_model(features,labels,valid_x,valid_y,model_name,image_path):
  trainingData=np.array(features)
  # reshape from [samples, timesteps] into [samples, timesteps, features]
  n_features = 1
  n_steps = features.shape[1]
  X = trainingData.reshape((trainingData.shape[0], trainingData.shape[1], n_features))

  model = Sequential()
  model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
  model.add(LSTM(50, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  model.fit(X, labels, epochs=200, verbose=0)
  Plt(model, valid_x, valid_y, model_name,image_path)