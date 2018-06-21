import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, ShuffleSplit
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


def load_feature_data(path=None):
    return pd.read_csv(r'F:\graduate student\zpc\lunwen\feature_value.csv')


def split_train_test(data, test_ratio):
    indices = np.random.permutation(len(data))  # 随机全排列
    test_size = int(len(data) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == "__main__":
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)

    feature_data = load_feature_data()
    raw_train_set, test_set = split_train_test(feature_data, 0.2)

    train_set = raw_train_set.copy()
    train_Ps = train_set.drop("Q", axis=1)  # 原始数据集并未发生改变
    # train_Ps = pd.DataFrame(min_max_scaler.fit_transform(train_Ps)) # 做归一化
    train_Q = train_set["Q"].copy()

    #######################  Linear regression     ##############################################
    print()
    print('Linear')
    lin_reg = LinearRegression()

    start_time = time.clock()
    lin_reg.fit(train_Ps, train_Q)
    print('Linear time', time.clock() - start_time)
    print('coef: ', lin_reg.coef_)

    some_data = train_Ps.iloc[:5]
    some_labels = train_Q.iloc[:5]
    print('prediction:', lin_reg.predict(some_data))
    print('labels:', some_labels.values)

    Q_predictions = lin_reg.predict(train_Ps)
    lin_mse = mean_squared_error(train_Q, Q_predictions)
    lin_rmse = np.sqrt(lin_mse)  # 平方根
    print('lin_reg rmse', lin_rmse)

    lin_scores = cross_val_score(lin_reg, train_Ps, train_Q,
                                 scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    print(lin_rmse_scores)
    print(lin_rmse_scores.mean())
    print(lin_rmse_scores.std())

    #######################  decesion tree     ##############################################
    print()
    print('tree')
    tree_reg = DecisionTreeRegressor()

    start_time = time.clock()
    tree_reg.fit(train_Ps, train_Q)
    print('tree_time', time.clock() - start_time)
    print('feature_importance:', tree_reg.feature_importances_)

    some_data = train_Ps.iloc[:5]
    some_labels = train_Q.iloc[:5]
    print('prediction:', tree_reg.predict(some_data))
    print('labels:', some_labels.values)

    Q_predictions = tree_reg.predict(train_Ps)
    tree_mse = mean_squared_error(train_Q, Q_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print('tree rmse', tree_rmse)

    #### cross
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(tree_reg, train_Ps, train_Q, scoring='neg_mean_squared_error', cv=10)
    rmse_scores = np.sqrt(-scores)
    print(rmse_scores)
    print(rmse_scores.mean())
    print(rmse_scores.std())

    #######################  network    ##############################################
    print()
    print('network')
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score

    # 4257
    net_reg = MLPRegressor(hidden_layer_sizes=10, activation='tanh', max_iter=32, solver='sgd', batch_size=4257,
                           # learning_rate='adaptive',
                           learning_rate_init=0.01, tol=0)
    start_time = time.clock()
    net_reg.fit(train_Ps, train_Q)
    print('network time', time.clock() - start_time)

    some_data = train_Ps.iloc[:5]
    some_labels = train_Q.iloc[:5]
    print('prediction:', net_reg.predict(some_data))
    print('labels:', some_labels.values)

    Q_predictions = net_reg.predict(train_Ps)
    net_mse = mean_squared_error(train_Q, Q_predictions)
    net_rmse = np.sqrt(net_mse)
    print('network rmse', net_rmse)

    print()
    print('####################### 神经网络测试#############################')
    # 去掉标签
    X_test = test_set.drop("Q", axis=1)
    # X_test = pd.DataFrame(min_max_scaler.fit_transform(X_test)) #归一化
    y_test = test_set["Q"].copy()

    print()

    final_predictions = net_reg.predict(X_test)
    # 均方误差
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print('final_net_rmse', final_rmse)

    final_mae = mean_absolute_error(y_test, final_predictions)
    print('final_net_mae', final_mae)

    final_r2 = r2_score(y_test, final_predictions)
    print('final_net_r2', final_r2)

    print('mean_Q', sum(y_test) / y_test.size)
    print('err', 100 * final_rmse / (sum(y_test) / y_test.size))

    # ---------rf模型网格搜索--------------
    # from sklearn.model_selection import GridSearchCV
    #
    # param_grid = [
    #     # 12 (3×4) 种超参数组合
    #     {'n_estimators': [3, 10, 30], 'max_features': [2, 3, 4, 5]},  # 组合1
    #     #  6 (2×3) 种
    #     {'bootstrap': [False], 'n_estimators': [3, 10, 30],
    #     # 'max_features': [4, 5, 6, 7, 8]},  # 组合2
    #     'max_features': [2, 3, 4, 5]},
    # ]
    #
    # forest_reg = RandomForestRegressor(random_state=42)
    # # 5折交叉验证，总共需要 (12+6)*5=90 次训练
    # grid_search = GridSearchCV(forest_reg, param_grid, cv=10,
    #                            scoring='neg_mean_squared_error')
    # grid_search.fit(train_Ps, train_Q)
    #
    # print()
    # print(grid_search.best_params_)
    # print(grid_search.best_estimator_)
    # print(grid_search.best_score_)
    #
    # cv_result = grid_search.cv_results_
    # for mean_score, params in zip(cv_result["mean_test_score"], cv_result["params"]):
    #     print(np.sqrt(-mean_score), params)
    #
    # forest_reg = grid_search.best_estimator_  ###最终的结果



    #######################  random forest     ##############################################
    print()
    print('forest')

    forest_reg = RandomForestRegressor()  # (max_features=3, n_estimators=200, bootstrap=False)

    start_time = time.clock()
    forest_reg.fit(train_Ps, train_Q)
    print('forest time', time.clock() - start_time)
    print('feature_importance:', forest_reg.feature_importances_)

    some_data = train_Ps.iloc[:5]
    some_labels = train_Q.iloc[:5]
    print('prediction:', forest_reg.predict(some_data))
    print('labels:', some_labels.values)

    Q_predictions = forest_reg.predict(train_Ps)
    forest_mse = mean_squared_error(train_Q, Q_predictions)
    forest_rmse = np.sqrt(forest_mse)
    print('forest rmse', forest_rmse)

    forest_scores = cross_val_score(forest_reg, train_Ps, train_Q,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    print(forest_rmse_scores)
    print(forest_rmse_scores.mean())
    print(forest_rmse_scores.std())

    forest_r2_scores = cross_val_score(forest_reg, train_Ps, train_Q,
                                       scoring="r2", cv=10)
    print(forest_r2_scores)
    print('r2', forest_r2_scores.mean())

    ###-----------------------  测试集 ---------------------------####
    print()
    print('#######################测试#############################')
    # 去掉标签
    X_test = test_set.drop("Q", axis=1)
    # X_test = pd.DataFrame(min_max_scaler.fit_transform(X_test)) #归一化
    y_test = test_set["Q"].copy()

    print()
    final_predictions = forest_reg.predict(X_test)
    # 均方误差
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print('final_forest_rmse', final_rmse)

    final_mae = mean_absolute_error(y_test, final_predictions)
    print('final_forest_mae', final_mae)

    # final_explain = explained_variance_score(y_test, final_predictions)
    # print('final_forest_explain', final_explain)

    final_r2 = r2_score(y_test, final_predictions)
    print('final_forest_r2', final_r2)

    print('mean_Q', sum(y_test) / y_test.size)
    print('err', 100 * final_rmse / (sum(y_test) / y_test.size))

    some_data = X_test.iloc[:40]
    some_labels = y_test.iloc[:40]

    pre_data = forest_reg.predict(some_data)
    print('forest_prediction:', pre_data)
    print('forest_labels:', some_labels.values)

    #######################查看误差####################################
    # for index in range(pre_data.size):
    #     if pre_data[index] != some_labels.values[index]:
    #         print(index, pre_data[index], some_labels.values[index], (pre_data[index] - some_labels.values[index]))

    #######################误差曲线#######################################
    # train_sizes, train_score, test_score = learning_curve(forest_reg, train_Ps, train_Q,
    #                                                       train_sizes=np.linspace(0.01, 1, 120), cv=10,
    #                                                       #scoring='neg_mean_squared_error'
    #                                                     )
    # train_score_mean = np.mean(train_score, axis=1)
    # test_score_mean = np.mean(test_score, axis=1)
    # plt.plot(train_sizes, train_score_mean, color='r', label='training', linewidth=2)
    # plt.plot(train_sizes, test_score_mean, color='g', label='testing', linewidth=2)
    # plt.legend(loc='best', fontsize=10)
    # plt.xlabel('train examples', fontsize=10)
    # plt.ylabel('score', fontsize=10)
    # plt.show()


    #######################系数曲线#####################
    # param_range = [1, 2, 3, 4, 5]  # range(3, 303, 3)
    # train_score2, validation_score2 = validation_curve(forest_reg, train_Ps, train_Q, param_name='max_features', cv=10
    #                                                    , param_range=param_range,
    #                                                    )  # 改变变量，来看得分
    #
    # x_axis = np.array(param_range)
    # train_score2_mean = train_score2.mean(1)
    # train_score2_std = train_score2.std(1)
    # validation_score2_mean = validation_score2.mean(1)
    # validation_score2_std = validation_score2.std(1)
    #
    # plt.plot(x_axis, train_score2_mean, c='r', label='train score', linewidth=4)
    # plt.plot(x_axis, validation_score2_mean, c='g', label='validation score', linewidth=4)
    #
    # plt.xlabel('feature number', fontsize=20)
    # plt.ylabel('score', fontsize=20)
    # plt.legend(fontsize=20)
    # plt.show()


