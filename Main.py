import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import datetime

from sklearn.model_selection import train_test_split

data = pd.read_csv('dataset/samsung.csv')
open_prices = data['Open'].values
high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices + low_prices) / 2
close_prices = data['Close'].values
adj_close = data['Adj Close'].values
volume = data['Volume'].values


data.head()

seq_len = 50
sequence_length = seq_len + 1
result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index:index + sequence_length])

normalized_data = []
for window in result:
    normalized_window = [((float(p)/float(window[0])) - 1) for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)
row = int(round(result.shape[0] * 0.9))

result_x = result[:, :-1]
result_y = result[:, -1]

train_x, test_x, train_y, test_y = train_test_split(result_x, result_y, test_size=0.2, shuffle=False)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
valid_x = np.reshape(valid_x, (valid_x.shape[0], valid_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

# lstm_input_size = 50
# h1 = 100
# batch_size = 10
# output_dim = 1
# num_layers
# model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)
#
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
model.add(LSTM(64, return_sequences=False))

model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')
print(model.summary())
model.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size=10, epochs=20)

pred = model.predict(test_x)
fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(test_y, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()
