import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import keras
import utils
import copy

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 载入模型
def load_models(n_in_timestep):
    keras.backend.clear_session()
    # global LSTM_hour_model
    LSTM_hour_model = load_model('./my_model_in time step_%d_out_timestep_1.h5' % n_in_timestep,
                                 compile=False)  # 选取自己的.h模型名称
    return LSTM_hour_model
    # global hour_graph
    # hour_graph = tf.get_default_graph()


def scale_dataset(values, flag, order, file_path, alpha):
    # 加载文件数据
    """
    values: dataset values
    scale_range: scale range to fit data in
    """
    header_row_index = 0
    index_col_name = 'TM'
    col_to_predict = 'QIN'
    cols_to_drop = None
    # cols_to_drop = []
    col_names_all, values_all, n_features_all, output_col_name_all = utils.load_dataset(file_path, header_row_index,
                                                                                        index_col_name, col_to_predict,
                                                                                        cols_to_drop, order[:-1])
    # normalize features
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = None
    # scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaled2 = scaler.fit_transform(values_all)
    # scaled = scaler.transform(values)

    # normalize features
    # 最大最小
    # global qin_max
    # global qin_min
    # max_list = values_all.max(axis=0)
    # min_list = values_all.min(axis=0)
    # qin_max = max_list[0]
    # qin_min = min_list[0]
    # for i in range(len(min_list)):
    #     for index, value in enumerate(values[:, i]):
    #         if value >= max_list[i]:
    #             values[:, 0][index] = max_list[i]

    # 均值方差
    global qin_avg
    global qin_std
    qin_avg = np.average(values_all[:, 0])
    qin_std = np.std(values_all[:, 0], ddof=1)

    # 微调参数设置
    alpha_pqj = 1  # 1.3
    alpha_qwmp = 1  # 1.2
    # print(values)#QIN PQJ QWMP

    # 对PQJ和QWMP特定区间进行放大调整
    for index, pqj in enumerate(values_all[:, 1]):
        if pqj > 10 and pqj < 20:
            values_all[:, 1][index] *= alpha_pqj

    for index, qwmp in enumerate(values_all[:, 2]):
        if qwmp > 2500 and qwmp < 6000:
            values_all[:, 2][index] *= alpha_qwmp

    for index, pqj in enumerate(values[:, 1]):
        if pqj > 10 and pqj < 20:
            values[:, 1][index] *= alpha_pqj

    for index, qwmp in enumerate(values[:, 2]):
        if qwmp > 2500 and qwmp < 6000:
            values[:, 2][index] *= alpha_qwmp

    scaled = None

    # 最大最小归一化
    # for i in range(len(min_list)):
    #     if i == 1:
    #         tmp_scaled = (values[:, i] - values_all[:, i].min(axis=0)) / (
    #                 values_all[:, i].max(axis=0) - values_all[:, i].min(axis=0)) * alpha  # 1
    #         tmp_scaled = tmp_scaled.reshape((-1, 1))
    #     else:
    #         tmp_scaled = (values[:, i] - values_all[:, i].min(axis=0)) / (
    #                 values_all[:, i].max(axis=0) - values_all[:, i].min(axis=0))   # 1
    #         tmp_scaled = tmp_scaled.reshape((-1, 1))
    #
    #     if i == 0:
    #         scaled = tmp_scaled
    #     else:
    #         scaled = np.append(scaled, tmp_scaled, axis=1)

    # 均值方差归一化
    for i in range(values_all.shape[1]):
        average = np.average(values_all[:, i])
        standard = np.std(values_all[:, i], ddof=1)
        if i == 1:
            tmp_scaled = (values[:, i] - average) / standard * alpha
            tmp_scaled = tmp_scaled.reshape((-1, 1))
        else:
            tmp_scaled = (values[:, i] - average) / standard
            tmp_scaled = tmp_scaled.reshape((-1, 1))
        if i == 0:
            scaled = tmp_scaled
        else:
            scaled = np.append(scaled, tmp_scaled, axis=1)

    return scaler, scaled


def inverse(pre_y, test_y):
    # pre_len = len(pre_y)
    # pre_y = np.array(pre_y).reshape((pre_len, 1))
    # inv_yhat = (pre_y * (qin_max - qin_min) + qin_min)
    #
    # # invert scaling for actual
    # test_y = test_y.reshape((len(test_y), 1))
    # inv_y = (test_y * (qin_max - qin_min) + qin_min)

    pre_len = len(pre_y)
    pre_y = np.array(pre_y).reshape((pre_len, 1))
    inv_yhat = pre_y * qin_std + qin_avg

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = test_y * qin_std + qin_avg
    return inv_yhat, inv_y


def scroll_predict(X, model, n_in_timestep, n_features):
    # 滚动预测
    pre_list = []
    for index, data in enumerate(X):
        if index < 1:
            data1 = data.reshape((1, n_in_timestep, n_features))
            pred = model.predict(data1)
            pred = pred[:, 0].reshape(-1, 1)
            pred = float(np.squeeze(pred))
            pre_list.append(pred)
        else:
            for k in range(1, index + 1):
                if k <= n_in_timestep:
                    data[-k][0] = pre_list[-k]
            data1 = data.reshape((1, n_in_timestep, n_features))
            pred = model.predict(data1)
            pred = pred[:, 0].reshape(-1, 1)
            pred = float(np.squeeze(pred))
            pre_list.append(pred)
    return pre_list


def get_predict_data(time_start, FDNO, order, file_path, n_in_timestep, alpha):
    header_row_index = 0
    index_col_name = 'TM'
    n_out_timestep = 1
    dropnan = True
    # 获得场次洪水所有数据
    flood_code = str(FDNO)
    col_names, dataset, values, time_index = utils.load_database(file_path, header_row_index, index_col_name,
                                                                 flood_code, order)
    # 将预报开始时间-5h
    time_start = pd.to_datetime(time_start)
    # 预报开始时间>洪水结束时间
    if time_start > pd.to_datetime(dataset.index[-1]):
        time_start = pd.to_datetime(dataset.index[0])
    time_pre = time_start - pd.Timedelta(hours=n_in_timestep)
    # 判断预报开始时间和洪水开始时间的关系
    flood_begin = pd.to_datetime(dataset.index[0])
    time_roll = time_start - flood_begin
    time_roll = int(time_roll / np.timedelta64(1, "h"))
    if time_pre >= flood_begin:
        time_pre = datetime.strftime(time_pre, "%Y-%m-%d %H:%M:%S")
        dataset2 = dataset[time_pre:]
    else:
        dataset2 = dataset
    # dataset2['PQJ'] = dataset2['PQJ'].shift(-1)
    # dataset2['PQJ'][-1] = dataset2['PQJ'][-2]
    # dataset2['QWMP'] = dataset2['QWMP'].shift(-1)
    # dataset2['QWMP'][-1] = dataset2['QWMP'][-2]
    # dataset2['QIN_PROCESSED'] = dataset2['QIN_PROCESSED'].shift(-1)
    # dataset2['QIN_PROCESSED'][-1] = dataset2['QIN_PROCESSED'][-2]
    tmp_order = order[:-2]
    dataset1 = dataset2[tmp_order]
    '''数据处理'''
    count_values, col_names, output_col_name = utils.data_process(dataset1)
    '''values未归一化之前的备份'''
    all = np.array(count_values, copy=True)
    '''归一化'''
    scaler, values3 = scale_dataset(count_values, 'offline', order, file_path, alpha)
    '''数据加工'''
    agg_quzhi = utils.series_to_supervised(values3, n_in_timestep, n_out_timestep, dropnan, col_names)
    '''转为神经网络要求形式'''
    test_X, test_Y = utils.split_data_to_test_sets(agg_quzhi.values, n_in_timestep, values3.shape[1])
    n_features = values3.shape[1]
    '''滚动预测'''
    # 载入模型
    LSTM_hour_model = load_models(n_in_timestep)
    pre_list = scroll_predict(test_X, LSTM_hour_model, n_in_timestep, n_features)
    '''模拟计算'''
    # pre_list = LSTM_hour_model.predict(test_X, batch_size=50, verbose=1)
    '''反归一化'''
    pred, y_inv = inverse(pre_list, test_Y)
    pred = list(np.array(pred).flatten())
    QT = pred
    # TODO
    QR = values[:, 0].tolist()
    PT1 = values[:, 1].tolist()
    PT2 = values[:, 2].tolist()
    PT3 = values[:, 3].tolist()
    PT4 = values[:, 4].tolist()
    PT5 = values[:, 5].tolist()
    PT6 = values[:, 6].tolist()
    PT7 = values[:, 7].tolist()
    DT = time_index
    if time_roll > 0:
        for i in range(time_roll - 1, -1, -1):
            QT.insert(0, QR[i])
    else:
        time_roll = n_in_timestep
        for i in range(time_roll - 1, -1, -1):
            QT.insert(0, QR[i])
    return QT, QR, PT1, PT2, PT3, PT4, PT5, PT6, PT7, DT, time_roll


def get_predict_data_ol(df):
    n_in_timestep = 15
    n_out_timestep = 1
    verbose = 2
    dropnan = True
    time_values = df.values
    # read NT,DT,PT,ET,QR,QC
    NT = time_values.shape[0]
    PT = df['PQJ'].values
    try:
        ET = df['EVP'].values
    except:
        ET = []
    QR = df['QIN'].values
    QC = df['QSQ'].values
    start_time = df['start_time'].values[0]
    pre_time = df['TM'].values[0]
    pre_time = pd.to_datetime(pre_time)
    start_time = pd.to_datetime(start_time)
    time_roll = start_time - pre_time
    time_roll = int(time_roll / np.timedelta64(1, 'h'))
    pre_ai_time = start_time - pd.Timedelta(hours=n_in_timestep)
    pre_ai_time = datetime.strftime(pre_ai_time, "%Y-%m-%d %H:%M:%S")
    df.set_index("TM", inplace=True)
    DT = df.index
    for i in range(time_roll):
        df['QF'][i] = df['QIN'][i]
    order = ['QF', 'QSQ', 'PQJ']
    dataset2 = df[order]
    # # dataset2['PQJ'] = dataset2['PQJ'].shift(-1)
    # # dataset2['PQJ'][-1] = dataset2['PQJ'][-2]
    # dataset2['QSQ'] = dataset2['QSQ'].shift(-1)
    # dataset2['QSQ'][-1] = dataset2['QSQ'][-2]
    # dataset2['QF'] = dataset2['QF'].shift(-1)
    # dataset2['QF'][-1] = dataset2['QF'][-2]
    dataset3 = dataset2.loc[pre_ai_time:, :]
    dataset3.columns = ['QIN', 'QSQ', 'PQJ']
    '''数据处理'''
    count_values, col_names, output_col_name = utils.data_process(dataset3)
    '''values未归一化之前的备份'''
    all = np.array(count_values, copy=True)
    '''归一化'''
    scaler, values3 = scale_dataset(count_values, 'online')
    '''数据加工'''
    agg_quzhi = utils.series_to_supervised(values3, n_in_timestep, n_out_timestep, dropnan, col_names)
    '''转为神经网络要求形式'''
    test_X, test_Y = utils.split_data_to_test_sets(agg_quzhi.values, n_in_timestep, values3.shape[1])
    n_features = values3.shape[1]
    '''滚动预测'''
    # 载入模型
    LSTM_hour_model = load_models()
    pre_list = scroll_predict(test_X, LSTM_hour_model, n_in_timestep, n_features)
    '''模拟计算'''
    # pre_list = LSTM_hour_model.predict(test_X, batch_size=50, verbose=1)
    '''反归一化'''
    pred, y_inv = inverse(pre_list, test_Y)
    pred = list(np.array(pred).flatten())
    QT = list(QR)[:time_roll] + pred
    QT = np.float64(QT)
    return QT, QR, QC, PT, DT, time_roll


def get_predict_data_test(time_start, time_end, flood_start):
    n_in_timestep = 15
    n_out_timestep = 1
    dropnan = True
    # 载入模型
    time_start = pd.to_datetime(time_start)
    time_start = datetime.strftime(time_start, "%Y-%m-%d %H")
    time_start = pd.to_datetime(time_start)  # 预报开始

    time_end = pd.to_datetime(time_end)
    time_end = datetime.strftime(time_end, "%Y-%m-%d %H")
    time_end = pd.to_datetime(time_end)  # 预报结束

    flood_start = pd.to_datetime(flood_start)
    flood_start = datetime.strftime(flood_start, "%Y-%m-%d %H")
    flood_start = pd.to_datetime(flood_start)  # 洪水开始

    time_pre = time_start - pd.Timedelta(hours=n_in_timestep)  # 计算开始
    time_roll = time_start - flood_start  # 预报开始与洪水开始之间的差值
    time_roll = int(time_roll / np.timedelta64(1, 'h'))
    if time_roll <= 0:
        time_roll = 0
    # 获得场次洪水所有数据
    # read data1
    col_names, dataset1, values1, time_index1 = utils.load_database_check(flood_start, time_end)
    # read AI data
    col_names, dataset2, values2, time_index2 = utils.load_database_check(time_pre, time_end)
    if col_names == 0 and dataset2 == 0 and values2 == 0 and time_index2 == 0:
        return 0, 0, 0, 0, 0, 0
    elif type(col_names) != int and type(dataset2) == int and type(values2) == int and type(time_index2) == int:
        return col_names, 0, 0, 0, 0, 0

    if time_pre >= flood_start:
        # dataset2['PQJ'] = dataset2['PQJ'].shift(-1)
        # dataset2['PQJ'][-1] = dataset2['PQJ'][-2]
        # dataset2['QSQ'] = dataset2['QSQ'].shift(-1)
        # dataset2['QSQ'][-1] = dataset2['QSQ'][-2]
        # dataset2['QIN'] = dataset2['QIN'].shift(-1)
        # dataset2['QIN'][-1] = dataset2['QIN'][-2]
        order = ['QIN', 'QSQ', 'PQJ']
        dataset2 = dataset2[order]
        '''数据处理'''
        count_values, col_names, output_col_name = utils.data_process(dataset2)
        '''values未归一化之前的备份'''
        all = np.array(count_values, copy=True)
        '''归一化'''
        scaler, values3 = scale_dataset(count_values, 'online')
        '''数据加工'''
        agg_quzhi = utils.series_to_supervised(values3, n_in_timestep, n_out_timestep, dropnan, col_names)
        '''转为神经网络要求形式'''
        test_X, test_Y = utils.split_data_to_test_sets(agg_quzhi.values, n_in_timestep, values3.shape[1])
        n_features = values3.shape[1]
        '''滚动预测'''
        # 载入模型
        LSTM_hour_model = load_models()
        pre_list = scroll_predict(test_X, LSTM_hour_model, n_in_timestep, n_features)
        '''模拟计算'''
        # pre_list = LSTM_hour_model.predict(test_X, batch_size=50, verbose=1)
        '''反归一化'''
        pred, y_inv = inverse(pre_list, test_Y)
        pred = list(np.array(pred).flatten())
        QR = values1[:, 0].tolist()
        QC = values1[:, 1].tolist()
        PT = values1[:, 2].tolist()
        DT = time_index1
        QT = list(values1[:, 0])[:time_roll] + pred
        QT = np.float64(QT)
        return QT, QR, QC, PT, DT, time_roll
    else:
        return col_names, 0, 0, 0, 0, 0
