import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from hmmlearn.hmm import GaussianHMM
import itertools


class HMM():
    def __init__(self, route, time_unit, period, hidden_states=4, training_ratio=0.9):
        self.time_convert = {"minute": 1, "hour": 60, "day": 60 * 24, "week": 60 * 24 * 7}
        self.time_unit = time_unit
        self.time_sampling = self.time_convert[self.time_unit]
        self.period = period
        self.training_ratio = training_ratio

        self.read_data(route)
        self.preprocess_data()
        self.discrete_outcomes(4, 4, 4, 10)

        self.hmm = GaussianHMM(n_components=hidden_states)

    def read_data(self, route):
        self.data = pd.read_csv(route).dropna()

    def preprocess_data(self):
        # prices when open, high, low, close
        data = np.array(self.data.iloc[:, 1:5].values).astype(np.float32)[0::self.time_sampling, :]

        # extract features
        feat_change = (data[:-1, 3] - data[:-1, 0]) / data[:-1, 0]  # (close - open) / open
        feat_high = (data[:-1, 1] - data[:-1, 0]) / data[:-1, 0]    # (high - open) / open
        feat_low = (data[:-1, 0] - data[:-1, 2]) / data[:-1, 0]     # (open - low) / open
        feat_per = (data[1:, 0] - data[:-1, 0]) / data[:-1, 0]      # (next_open - open) / open

        # train test split
        self.train_length = int(self.training_ratio * data.shape[0])
        self.train_data = data[:self.train_length]
        self.test_data = data[self.train_length:]

        # stack the feats ((fc, fh, fl, fp))
        self.train_feat = np.column_stack((feat_change[:self.train_length],
                                           feat_high[:self.train_length],
                                           feat_low[:self.train_length],
                                           feat_per[:self.train_length]))
        self.test_feat = np.column_stack((feat_change[self.train_length:],
                                          feat_high[self.train_length:],
                                          feat_low[self.train_length:],
                                          feat_per[self.train_length:]))

    # train the hmm model
    def fit(self):
        self.hmm.fit(self.train_feat)

    # identify the possible outcomes, for later picking the one that maximizes the posterior probability
    def discrete_outcomes(self, fc_samples, fh_samples, fl_samples, fp_samples):
        fc = np.linspace(-0.015, 0.015, fc_samples)
        fh = np.linspace(0, 0.025, fh_samples)
        fl = np.linspace(0, 0.02, fl_samples)
        fp = np.linspace(-0.5, 0.4, fp_samples)
        self.possible_outcomes = np.array(list(itertools.product(fc, fh, fl, fp)))

    def predict(self, sets="test"):
        if sets == "test":
            feat = self.test_feat
            openp = self.test_data[:, 0]
        elif sets == "train":
            feat = self.train_feat
            openp = self.train_data[:, 0]

        predicted_open = []
        for t in range(feat.shape[0]):
            open_price = openp[t]
            predicted_open_price = (self.ml_outcome(feat, t) + 1) * open_price # inverse function of (next_open - open) / open
            predicted_open.append(predicted_open_price)
            print(f"predicted: {t} / {feat.shape[0]}")
        return predicted_open

    def ml_outcome(self, feat, t):
        start = max(0, t - self.period)
        end = max(0, t - 1)
        feats = feat[start:end]

        max_score = -1
        max_outcome = None
        for outcome in self.possible_outcomes:
            total_data = np.row_stack((feats, outcome))
            score = self.hmm.score(total_data)  # get the log-likelihood of the data
            if score > max_score:
                max_score = score
                max_outcome = outcome

        return max_outcome[3]


if __name__ == '__main__':
    hmm = HMM("dataset.csv", "day", 7)
    hmm.fit()
    prediction = hmm.predict("test")

    plt.plot(range(hmm.test_data[:, 0].shape[0]), hmm.test_data[:, 0], 'r', label='Actual Open Price')
    plt.plot(range(len(prediction)), prediction, label='Predicted Open Price')
    plt.legend()
    plt.show()