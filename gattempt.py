from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing


import numpy as np
import pickle



f = open('train.pickle', 'rb')
loaded_values = pickle.load(f)
train_x = loaded_values[0]
train_y = loaded_values[1]
f.close()

train_x = np.array(train_x)
train_x = train_x.reshape([1, 344, 64, 1])

train_y = np.array(train_y)
train_y = train_y.reshape(1,54810)



# Building convolutional network
network = input_data(shape=[None, 344, 64, 1], name='input')
# network = input_data(shape=[None, 22016, 1, 1], name='input')
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
network = fully_connected(network, 54810, activation='softmax')
network = regression(network, optimizer='sgd', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=2)
model.fit({'input': train_x}, {'target': train_y} , n_epoch=20)


print("Printing shit")
answer = model.predict(train_x)
answer = np.array(answer)
print(answer.shape)
