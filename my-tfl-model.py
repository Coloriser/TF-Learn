from __future__ import print_function

import numpy as np
import tflearn
import pickle

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Download the Titanic dataset
# from tflearn.datasets import titanic
# titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
# from tflearn.data_utils import load_csv
# data, labels = load_csv('titanic_dataset.csv', target_column=0,
#                         categorical_labels=True, n_classes=2)


# Preprocessing function
# def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    # for id in sorted(columns_to_ignore, reverse=True):
    #     [r.pop(id) for r in data]
    # for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
    #   data[i][1] = 1. if data[i][1] == 'female' else 0.
    # return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
# to_ignore=[1, 6]

# Preprocess data
# data = preprocess(data, to_ignore)


# Load the data
f = open('train.pickle', 'rb')
loaded_values = pickle.load(f)
train_x = loaded_values[0]
train_y = loaded_values[1]
f.close()




train_x = np.array(train_x)
train_y = np.array(train_y)
train_y = np.array([list(train_y.flatten())])

# t_x = np.empty((0,64))
# train_x = t_x.fill(train_x[0])


# t_y = np.empty((0,64))
# train_y = t_y.fill(train_y[0])


print (train_x.shape)
print (train_y.shape)

# Build neural network
# net = tflearn.input_data(shape=[None, 344, 64])
# net = tflearn.fully_connected(net, 32)
# net = tflearn.fully_connected(net, 32)
# net = tflearn.fully_connected(net, 54810, activation='softmax')
# net = tflearn.regression(net)

# Define model
# model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
# model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)
# model.fit(train_x, train_y, n_epoch=1000, batch_size=16, show_metric=True)






network = input_data(shape=[None, 344, 64, None], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': train_x}, {'target': train_y}, n_epoch=20,
           validation_set=({'input': train_x}, {'target': train_y}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')















# Let's create some data for DiCaprio and Winslet
# dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
# winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]

# Preprocess data
# dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)


# image = train_x
# print(image.shape)
# pred = model.predict(image)
# print("Winslet Surviving Rate:", pred[1][1])
# print("Prediction :", pred)
