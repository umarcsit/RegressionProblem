#
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.optimizers import Adam
import tensorflow as tf  # Importing TensorFlow for Huber loss
from graphs.error_graph import Plt

def RNNModel(features, labels, valid_x, valid_y, model_name,image_path):
    trainingData = np.array(features)
    # Reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    n_steps = features.shape[1]
    X = trainingData.reshape((trainingData.shape[0], trainingData.shape[1], n_features))

    # Define the RNN model
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', input_shape=(n_steps, n_features), return_sequences=True))
    model.add(SimpleRNN(30, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1))  # Output layer

    # Compile the model using Huber loss
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=tf.keras.losses.Huber())

    # Train the model
    model.fit(features, labels, validation_data=(valid_x, valid_y), epochs=2, batch_size=32, verbose=0)

    # Plot the results and show metrics
    Plt(model, valid_x, valid_y, model_name,image_path)