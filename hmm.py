import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from hmmlearn.hmm import GaussianHMM
import itertools


class HMM():
    def __init__(self, route, time_unit, period, hidden_states=3, training_ratio=0.9):
        self.time_convert = {"minute": 1, "hour": 60, "day": 60 * 24, "week": 60 * 24 * 7}
        self.time_unit = time_unit
        self.time_sampling = self.time_convert[self.time_unit]
        self.period = period
        self.training_ratio = training_ratio

        self.read_data(route)
        self.preprocess_data()
        self.discrete_outcomes(10, 4, 4)

        self.hmm = GaussianHMM(n_components=hidden_states)

    def read_data(self, route):
        self.data = pd.read_csv(route).dropna()

    def preprocess_data(self):
        # prices when open, high, low, close
        data = np.array(self.data.iloc[-525600:, 1:5].values)

        data_open = data[:, 0][0::self.time_sampling]
        data_high = np.array([])
        data_low = np.array([])
        data_close = data[:, 3][self.time_sampling - 1::self.time_sampling]
        for i in range(len(data_open)):
            data_high = np.append(data_high, np.amax(data[:, 1][(self.time_sampling * i): (self.time_sampling * (i + 1))]))
            data_low = np.append(data_low, np.amin(data[:, 2][(self.time_sampling * i): (self.time_sampling * (i + 1))]))
        data = np.column_stack((data_open, data_high, data_low, data_close))

        # extract features
        feat_change = (data_close - data_open) / data_open
        feat_high = (data_high - data_open) / data_open
        feat_low = (data_open - data_low) / data_open

        # train test split
        self.train_length = int(self.training_ratio * data.shape[0])
        self.train_data = data[:self.train_length, :]
        self.test_data = data[self.train_length:, :]

        # stack the feats ((fc, fh, fl, fp))
        self.train_feat = np.column_stack((feat_change[:self.train_length],
                                           feat_high[:self.train_length],
                                           feat_low[:self.train_length]))
        self.test_feat = np.column_stack((feat_change[self.train_length:],
                                          feat_high[self.train_length:],
                                          feat_low[self.train_length:]))

    # train the hmm model
    def fit(self):
        self.hmm.fit(self.train_feat)

    # identify the possible outcomes, for later picking the one that maximizes the posterior probability
    def discrete_outcomes(self, fc_samples, fh_samples, fl_samples):
        fc = np.linspace(-0.015, 0.015, fc_samples)
        fh = np.linspace(0, 0.025, fh_samples)
        fl = np.linspace(0, 0.02, fl_samples)
        self.possible_outcomes = np.array(list(itertools.product(fc, fh, fl)))

    def predict(self, sets="test"):
        if sets == "test":
            feat = self.test_feat
            openp = self.test_data[:, 0]
        elif sets == "train":
            feat = self.train_feat
            openp = self.train_data[:, 0]

        predicted_close = []
        for t in range(feat.shape[0]):
            open_price = openp[t]
            predicted_close_price = (self.ml_outcome(feat, t) + 1) * open_price  # inverse function of (next_open - open) / open
            predicted_close.append(predicted_close_price)
            print(f"predicted: {t+1} / {feat.shape[0]}")
        return predicted_close

    def ml_outcome(self, feat, t):
        start = max(0, t - self.period)
        end = max(0, t)
        feats = feat[start:end]

        max_score = -1000
        max_outcome = None
        for outcome in self.possible_outcomes:
            total_data = np.row_stack((feats, outcome))
            score = self.hmm.score(total_data)  # get the log-likelihood of the data
            if score > max_score:
                max_score = score
                max_outcome = outcome

        return max_outcome[0]


if __name__ == '__main__':
    hmm = HMM("dataset.csv", "day", 7)
    hmm.fit()
    prediction = hmm.predict("test")

    plt.plot(range(hmm.test_data[:, 3].shape[0]), hmm.test_data[:, 3], 'r', label='Actual Close Price')
    plt.plot(range(len(prediction)), prediction, label='Predicted Close Price')
    plt.legend()
    plt.show()