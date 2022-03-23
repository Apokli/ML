# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

time_convert = {"minute": 1, "hour": 60, "day": 60 * 24, "week": 60 * 24 * 7}

# The RNN module
class RNN(nn.Module):
    def __init__(self, i_size, h_size, num_layers=1):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = h_size
        self.input_size = i_size

        self.rnn1 = nn.LSTM(i_size, h_size, num_layers, batch_first=True)   # the LSTM RNN layer
        # self.dropout = nn.Dropout(p=0.20)
        self.dense1 = nn.Linear(8, 4)   # fully connected layers
        self.dense2 = nn.Linear(4, 1)   # fully connected layers

    def forward(self, x):
        x_batch = x.view(len(x), 1, -1)
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # initializing hidden layers
        cells = torch.zeros(self.num_layers, x.size(0), self.hidden_size)   # initializing memory cells
        x_r, (hidden, cells) = self.rnn1(x_batch, (hidden, cells))          # passing through the LSTM layer
        # x_d = self.dropout(x_r)
        x_l = self.dense1(x_r)
        x_l2 = self.dense2(x_l)

        return x_l2


# Data Preprocessing
def preprocess_data(data, prediction_period, training_ratio=0.9, time_sampling=1):
    # prices when open
    data_sample = np.array(data.iloc[:, 1:5].values).astype(np.float32)[0::time_sampling]

    # normalization to [0, 1]
    mns = MinMaxScaler()
    norm_data_sample = mns.fit_transform(data_sample)

    # take a sample every <time_sampling> entry
    training_length = int(norm_data_sample.shape[0] * training_ratio)

    # split train and test set, we want to use the first 0.9 data to train, and last 0.1 to test
    train_set = norm_data_sample[:training_length]
    test_set = norm_data_sample[training_length:]

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    # reform it into data-matrices to train prediction model
    for i in range(len(train_set) - prediction_period):
        X_train.append(np.append(train_set[i : (i + prediction_period), :], train_set[i + prediction_period, 0]))
        y_train.append(train_set[i + prediction_period, 3])
    for i in range(len(test_set) - prediction_period):
        X_test.append(np.append(test_set[i: (i + prediction_period), :], test_set[i + prediction_period, 0]))
        y_test.append(test_set[i + prediction_period, 3])

    # shape of X_train is (2252, 7), shape of y_train is (244, 7) by default

    return torch.from_numpy(np.array(X_train).astype(np.float32)), \
           torch.from_numpy(np.array(X_test).astype(np.float32)), \
           torch.from_numpy(np.array(y_train).astype(np.float32)).unsqueeze(1), \
           torch.from_numpy(np.array(y_test).astype(np.float32)).unsqueeze(1), \
           mns


def read_data(route):
    data = pd.read_csv(route).dropna()
    return data


if __name__ == '__main__':
    # ------------------------------ Data Preprocessing -------------------------------
    data = read_data("dataset.csv")
    time_unit = "day"
    prediction_period = 7   # 7 time units for a prediction
    X_train, X_test, y_train, y_test, mns = preprocess_data(data, prediction_period, training_ratio=0.9, time_sampling=time_convert[time_unit])
    print(f"shape of X_train in {time_unit}s:", X_train.shape)
    print(f"shape of y_train in {time_unit}s:", y_train.shape)
    print(f"shape of X_test in {time_unit}s:", X_test.shape)
    print(f"shape of y_test in {time_unit}s:", y_test.shape)

    # ------------------------------- Run RNN Training --------------------------------
    model = RNN(prediction_period * 4 + 1, h_size=8)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 2000

    for epoch in range(epochs):
        model.zero_grad()
        out = model(X_train)
        loss = loss_fn(out.view(-1, 1), y_train)
        if epoch % 20 == 0:
            print('epoch {} loss: {}'.format(epoch, loss.data.numpy().tolist()), flush=True)
        loss.backward()
        optimizer.step()

    # ------------------------------- Test RNN Accuracy -----------------------------

    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train)   # the prediction of prices of train data (normalized)
        y_test_pred = model(X_test)     # the prediction of prices of test data (normalized)

    # Training Data
    train_pred = mns.inverse_transform(np.tile(y_train_pred.numpy().squeeze().reshape(-1, 1), (1, 4)))[:, 3]  # denormalize
    train_actual = mns.inverse_transform(np.tile(y_train.numpy().reshape(-1, 1), (1, 4)))[:, 3]               # denormalize

    plt.plot(range(train_pred.shape[0]), train_actual, 'r', label='Actual Trained Price')
    plt.plot(range(train_pred.shape[0]), train_pred, label='Predicted Trained Price')
    plt.legend()
    plt.show()

    # Testing Data

    test_pred = mns.inverse_transform(np.tile(y_test_pred.numpy().squeeze().reshape(-1, 1), (1, 4)))[:, 3]  # denormalize
    test_actual = mns.inverse_transform(np.tile(y_test.numpy().reshape(-1, 1), (1, 4)))[:, 3]  # denormalize

    plt.plot(range(test_pred.shape[0]), test_actual, 'r', label='Actual Test Price')
    plt.plot(range(test_pred.shape[0]), test_pred, label='Predicted Test Price')
    plt.legend()
    plt.show()

    loss = loss_fn(y_test_pred.view(-1, 1), y_test).numpy()
    print("final loss on test set:", loss)

