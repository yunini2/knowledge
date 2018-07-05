# @time:2018/07/5 8:09 am
# @ Ash Test -- 6
# @python3.6
# @author:Lily

import  data_cluster
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

cluster_data = data_cluster.data_cluster_merge

def func_linear_reg(data_cluster_merge):
    x = [] # 73个0，22个1
    y = []
    for i in data_cluster_merge['category'].unique():
        x.append(np.array(data_cluster_merge[data_cluster_merge['category'] == i][['低能比%', '高能比%', '原点距离', '能比']]))
        y.append(np.array(data_cluster_merge[data_cluster_merge['category'] == i]['灰分']))


    x_train = []
    y_tain = []
    model = []
    y_pred = []
    mse = []
    pred_true = []
    for i in range(len(x)):
        ss_x = StandardScaler()
        ss_y =StandardScaler()
        x_train.append(ss_x.fit_transform(x[i]))
        y_tain.append(ss_y.fit_transform(y[i].reshape(-1, 1)))

        regr = linear_model.LinearRegression()
        model.append(regr.fit(x_train[i], y_tain[i]))
        print("x's coefficient:", model[i].coef_)
        print("intercept:", model[i].intercept_)
        y_pred.append(regr.predict(x_train[i]))


        mse.append(mean_squared_error(ss_y.inverse_transform(y_tain[i]), ss_y.inverse_transform(y_pred[i])))
        print('MSE:', mse[i])

        pred_true.append(pd.concat([pd.DataFrame(ss_y.inverse_transform(y_tain[i]), columns=['true']),
                               pd.DataFrame(ss_y.inverse_transform(y_pred[i]), columns=['predict_linear'])], axis=1))
        pred_true[i]['abs_err_linear'] = abs(pred_true[i]['true'] - pred_true[i]['predict_linear'])
        pred_true[i]['cor_err_linear'] = abs(pred_true[i]['true'] - pred_true[i]['predict_linear'])/pred_true[i]['predict_linear']

        pred_true_pd = pd.concat([pred_true[0], pred_true[1]], axis=1)
        return pred_true_pd

pred_true_pd = func_linear_reg(cluster_data)
pred_true_pd.to_csv('E://work//data_analysis//XG_data_analysis//result_data//pred_6_data.csv', index=None)


