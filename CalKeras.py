import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import time


def load_feature_data(path=None):
    return pd.read_csv(r'F:\graduate student\zpc\lunwen\feature_value.csv')


def split_train_test(data, test_ratio):
    indices = np.random.permutation(len(data))  # 随机全排列
    test_size = int(len(data) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


class MLPNetworkModel:
    def __init__(self):
        self.__construct_model()

    def __construct_model(self):
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=5, init='uniform', activation='sigmoid'))
        self.model.add(Dense(1, init='uniform', activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])

    def get_model(self):
        return self.model

if __name__ == "__main__":
    feature_data = load_feature_data()
    raw_train_set, raw_test_set = split_train_test(feature_data, 0.2)

    train_set = raw_train_set.copy()
    test_set = raw_test_set.copy()

    train_Ps = np.array(train_set.drop("Q", axis=1))  # 原始数据集并未发生改变
    train_Q = np.array(train_set["Q"].copy())

    test_Ps = np.array(test_set.drop("Q", axis=1))  # 原始数据集并未发生改变
    test_Q = np.array(test_set["Q"].copy())
    print(train_Ps.shape, test_Ps.shape)

    model = MLPNetworkModel().get_model()
    start = time.clock()
    model.fit(train_Ps, train_Q, validation_data=(test_Ps, test_Q), epochs=100000, verbose=2, batch_size=32)
    print("all time", time.clock() - start)

    scores = model.evaluate(test_Ps, test_Q, verbose=0)
    print("MLP Error: %.2f%%" % (100 - scores[1] * 100))

    y_pre = model.predict(test_Ps)

    print(y_pre[100][0])
    print(test_Q[100])
    err = 0
    for i in range(test_Q.shape[0]):
        tmp_err = abs(y_pre[i][0] - test_Q[i]) / test_Q[i]
        err += tmp_err
    print(err/test_Q.shape[0])
