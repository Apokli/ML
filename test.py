# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

time_convert = {"minute": 1, "hour": 60, "day": 60 * 24, "week": 60 * 24 * 7}

if __name__ == '__main__':
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    print(np.column_stack((a, b)))
