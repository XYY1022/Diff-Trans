import collections
import os.path
from datetime import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates, rcParams
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import MinMaxScaler

"""
画图函数的基本设置
"""
rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
font_song = FontProperties(fname='C://Windows//Fonts//simsun.ttc')
# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman', 'weight': 'normal'}
plt.rc('font', family='Times New Roman')  # 全局变为Times
config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

"""
数据库信息设置
"""
# connection = pymysql.connect(
#     host="119.96.177.102",
#     user="byhbjgtest",
#     passwd="byhbjg@@7053",
#     db="akyb",
#     port=47280,
#     charset="utf8",
# )
# connection = pymysql.connect(host='192.168.1.106',
#                              user='root',
#                              passwd='root',
#                              db='akyb',
#                              port=47280,
#                              charset='utf8'
#                              )

"""
数据初始化
"""
n_in_timestep = 3
n_out_timestep = 1
verbose = 2
dropnan = True
scale_range = (0, 1)
train_percentage = 0.5


def load_dataset(file_path, header_row_index, index_col_name, col_to_predict, cols_to_drop, order):
    """
    从csv或excel文件中加载文件数据
    file_path: 文件路径
    header_row_index: 文件列名称所在的位置
    index_col_name: 索引列的名称
    col_to_predict: 预测列的名称
    cols_to_drop: 要删除的索引列的名称
    """
    # 读取数据
    dataset = pd.read_excel(file_path, header=header_row_index, index_col=False)
    # 将数据按规定的列进行排序
    dataset = dataset[order]
    # 设置索引列，参数输入列的名字列表
    if index_col_name:
        dataset.set_index(index_col_name, inplace=True)
        global data_index
        data_index = dataset.index
    # 删除不需要的列，参数输入列的名字列表
    if cols_to_drop:
        if type(cols_to_drop[0]) == int:
            dataset.drop(index=cols_to_drop, axis=0, inplace=True)
        else:
            dataset.drop(columns=cols_to_drop, axis=1, inplace=True)
    # get rows and column names
    col_names = dataset.columns.values.tolist()
    values = dataset.values
    # 把预测列调至第一列
    # col_to_predict == "QIN"
    col_to_predict_index = (
        col_to_predict
        if type(col_to_predict) == int
        else col_names.index(col_to_predict)  # col_to_predict在col_names中第一个匹配值的索引
    )
    # col_to_predict_index == 0
    output_col_name = col_names[col_to_predict_index]
    if col_to_predict_index > 0:
        col_names = (
                [col_names[col_to_predict_index]]
                + col_names[:col_to_predict_index]
                + col_names[col_to_predict_index + 1:]
        )
    values = np.concatenate(
        (
            values[:, col_to_predict_index].reshape((values.shape[0], 1)),
            values[:, :col_to_predict_index],
            values[:, col_to_predict_index + 1:],
        ),
        axis=1,
    )
    # 将所有values中的值转为float
    values = values.astype("float32")
    # 将QWMP列的数据上移一格，最后一格用倒数第二格填充
    qwmp_values = values[:, 2]
    last_qwmp_val = qwmp_values[-1:]
    qwmp_values = np.append(qwmp_values[1:], last_qwmp_val, axis=0)
    values[:, 2] = qwmp_values
    return col_names, values, values.shape[1], output_col_name


def load_database(file_path, header_row_index, index_col_name, flood_code, order):
    """
    根据洪号读取excel数据
    """
    # 读取数据
    dataset = pd.read_excel(file_path, header=header_row_index, index_col=False)
    #     # 删除不需要的列，参数输入列的名字列表,将数据按规定的列进行排序

    dataset = dataset[order]
    # 设置索引列，参数输入列的名字列表
    if index_col_name:
        dataset.set_index(index_col_name, inplace=True)
        global data_index
        data_index = dataset.index
    # 根据flood_code找到对应的洪水数据
    dataset = dataset.query('FDNO=={}'.format(flood_code))
    dataset = dataset[order[:-2]]
    # dataset.set_index("TM", inplace=True)
    col_names = dataset.columns.values.tolist()
    # 将所有values中的值转为float
    values = dataset.values.astype("float32")
    data_index = dataset.index
    return col_names, dataset, values, data_index


# def load_database_check(time_start, time_end, connection=connection):
#     # 创建连接数据库
#     connection = connection
#     cur = connection.cursor()  # 游标（指针）cursor的方式操作数据
#     sql2 = "SELECT TM,QIN,QSQ,PQJ FROM d_realtime_flood " \
#            "where TM>='{}'and TM<='{}' order by TM"
#     sql = sql2.format(str(time_start), str(time_end))  # 转化后的sql语句{}赋值
#     connection.ping(reconnect=True)
#     cur.execute(sql)  # execute(query, args):执行单条sql语句。
#     rows = cur.fetchall()  # 使结果全部可看
#     # 创建json数据
#     objects_list = []
#     for row in rows:
#         d = collections.OrderedDict()
#         d['TM'] = str(row[0])
#         d['QIN'] = float(row[1])
#         # d['EVP'] = float(row[2])
#         d['QSQ'] = float(row[2])
#         d['PQJ'] = float(row[3])
#         objects_list.append(d)
#     # 将json格式转成str，因为如果直接将dict类型的数据写入json会发生报错，因此将数据写入时需要用到该函数。
#     j = json.dumps(objects_list, ensure_ascii=False)
#     cur.close()
#     connection.close()
#     df = pd.read_json(j, orient='records')
#     df.set_index("TM", inplace=True)
#     dataset = df
#     col_names = dataset.columns.values.tolist()
#     values = df.values
#     data_index = df.index
#     return col_names, dataset, values, data_index

def scale_dataset_train(values, order, file_path, alpha):
    # 加载文件数据
    """
    values: dataset values
    scale_range: scale range to fit data in
    """
    header_row_index = 0
    index_col_name = 'TM'
    col_to_predict = 'QIN_PROCESSED'
    cols_to_drop = None
    col_names_all, values_all, n_features_all, output_col_name_all = load_dataset(file_path, header_row_index,
                                                                                  index_col_name, col_to_predict,
                                                                                  cols_to_drop, order)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = None

    # normalize features
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

    # print(values)#QIN PQJ QWMP

    scaled = None

    # 最大最小值归一化
    # for i in range(len(min_list)):
    #     if i == 1:
    #         tmp_scaled = (values[:, i] - values_all[:, i].min(axis=0)) / (
    #                 values_all[:, i].max(axis=0) - values_all[:, i].min(axis=0)) * alpha  # 1
    #         tmp_scaled = tmp_scaled.reshape((-1, 1))
    #     else:
    #         tmp_scaled = (values[:, i] - values_all[:, i].min(axis=0)) / (
    #                 values_all[:, i].max(axis=0) - values_all[:, i].min(axis=0))  # 1
    #         tmp_scaled = tmp_scaled.reshape((-1, 1))
    #     if i == 0:
    #         scaled = tmp_scaled
    #     else:
    #         scaled = np.append(scaled, tmp_scaled, axis=1)

    # 均值方差归一化
    for i in range(values_all.shape[1]):
        average = np.average(values_all[:, i])
        standard = np.std(values_all[:, i], ddof=1)
        print(average, standard)
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


def _scale_dataset(values, scale_range):
    """
    数据归一化处理
    values: 数据名称
    scale_range: 归一化后的区间范围
    """
    scaler = MinMaxScaler(feature_range=scale_range)
    scaled = scaler.fit_transform(values)
    scaler_out = MinMaxScaler(feature_range=scale_range)
    scaler_out = scaler_out.fit(values[:, 0].reshape(-1, 1))
    return scaler, scaler_out, scaled


def series_to_supervised(values, n_in, n_out, dropnan, col_names):
    """
    将数据格式转化为监督学习的格式
    values: 数据名称
    n_in: 输入的个数
    n_out:输出的个数
    dropnan:删除有Nan的行
    col_names:列名
    """
    n_vars = 1 if type(values) is list else values.shape[1]
    if col_names is None:
        col_names = ["var%d" % (j + 1) for j in range(n_vars)]
    df = pd.DataFrame(values)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("%s(t-%d)" % (col_names[j], i)) for j in range(n_vars)]
    # print('names：',names)
    # print('cols:',cols)
    for i in range(0, n_out):
        cols.append(df.shift(-i))  # 这里循环结束后cols是个列表，每个列表都是一个shift过的矩阵
        if i == 0:
            names += [("%s(t)" % (col_names[j])) for j in range(n_vars)]
        else:
            names += [("%s(t+%d)" % (col_names[j], i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)  # 将cols中的每一行元素一字排开，连接起来
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def split_data_to_test_sets(values, n_intervals, n_features):
    n_obs = n_intervals * n_features
    test_X, test_y = values[:, :n_obs], values[:, -n_features]
    test_X = test_X.reshape((test_X.shape[0], n_intervals, n_features))
    return test_X, test_y


def split_data_to_train_test_sets(values, n_intervals, n_features, train_percentage, verbose):
    n_train_intervals = int(np.ceil(values.shape[0] * train_percentage))  # ceil(x)->得到最接近的一个不小于x的整数，如ceil(2.001)=3
    train = values[:n_train_intervals, :]
    test = values[n_train_intervals:, :]
    # test = values[:n_train_intervals, :]
    # train = values[n_train_intervals:, :]

    # split into input and outputs
    n_obs = n_intervals * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    # train_X此时的shape为[train.shape[0], timesteps * features]
    # print('before reshape\ntrain_X shape:', train_X.shape)
    test_X, test_y = test[:, :n_obs], test[:, -n_features]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_intervals, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_intervals, n_features))

    if verbose:
        print("")
        print("train_X shape:", train_X.shape)
        print("train_y shape:", train_y.shape)
        print("test_X shape:", test_X.shape)
        print("test_y shape:", test_y.shape)

    return train_X, train_y, test_X, test_y


# 数据处理
def data_process(df):
    col_to_predict = "QIN"
    # df.set_index("时间", inplace=True)
    data_index = df.index
    col_names = df.columns.values.tolist()
    values = df.values
    target = np.copy(values[:, 0])
    # move the column to predict to be the first col: 把预测列调至第一列
    col_to_predict_index = col_to_predict if type(col_to_predict) == int else col_names.index(col_to_predict)
    output_col_name = col_names[col_to_predict_index]
    if col_to_predict_index > 0:
        col_names = [col_names[col_to_predict_index]] + col_names[:col_to_predict_index] + col_names[
                                                                                           col_to_predict_index + 1:]
    values = np.concatenate((values[:, col_to_predict_index].reshape((values.shape[0], 1)),
                             values[:, :col_to_predict_index], values[:, col_to_predict_index + 1:]), axis=1)
    # ensure all data is float
    values = values.astype("float32")
    # QWMP上移
    qwmp_values = values[:, 2]
    last_qwmp_val = qwmp_values[-1:]
    qwmp_values = np.append(qwmp_values[1:], last_qwmp_val, axis=0)
    values[:, 2] = qwmp_values
    return values, col_names, output_col_name


# 最大最小值归一化
def normalize_feature_minmax(df):
    return df.apply(lambda column: (column - column.min()) / (column.max() - column.min()))


# 纳什系数的计算函数
def get_nash(a, b, time_roll):
    a = a[time_roll:]
    b = b[time_roll:]
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    c = np.zeros(len(a))
    d = np.zeros(len(a))
    for i in range(len(a) - 1):
        c[i] = (a[i] - b[i]) ** 2
        d[i] = (b[i] - b_mean) ** 2
    c_sum = np.sum(c)
    d_sum = np.sum(d)
    nash = 1 - c_sum / d_sum
    if nash <= -10:
        nash = 0
    return np.float64(nash)


# R2的计算函数
def get_R2(a, b, time_roll):
    a = a[time_roll:]
    b = b[time_roll:]
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    c = np.zeros(len(a))
    d = np.zeros(len(a))
    e = np.zeros(len(a))
    for i in range(len(a) - 1):
        c[i] = (a[i] - a_mean) * (b[i] - b_mean)
        d[i] = (a[i] - a_mean) ** 2
        e[i] = (b[i] - b_mean) ** 2
    c_sum = np.sum(c) ** 2
    d_sum = np.sum(d) * np.sum(e)
    R2 = c_sum / d_sum
    return R2


# RE的计算函数
def get_RE(a, b, time_roll):
    a = a[time_roll:]
    b = b[time_roll:]
    a_sum = np.sum(a)
    b_sum = np.sum(b)
    RE = (b_sum - a_sum) / a_sum * 100
    return RE


# RMSE的计算函数
def get_RMSE(a, b, time_roll):
    a = a[time_roll:]
    b = b[time_roll:]
    c = np.zeros(len(a))
    for i in range(len(a) - 1):
        c[i] = (a[i] - b[i]) ** 2
    c_sum = np.sum(c)
    RMSE = np.sqrt(c_sum / len(a))
    return RMSE


# 计算洪水总量
def get_all_flood(QIN, time_roll):
    QIN = QIN[time_roll:]
    W = 0.0
    for i in range(len(QIN) - 1):
        W = W + (QIN[i] + QIN[i + 1]) / 2
    W = W * 3600 * 1.0e-8  # 单位转换为亿m³
    return W


# 计算洪水总量(日)
def get_all_flow(QIN, time_roll):
    QIN = QIN[time_roll:]
    W = 0.0
    for i in range(len(QIN) - 1):
        W = W + (QIN[i] + QIN[i + 1]) / 2
    W = W * 3600 * 1.0e-8 * 24  # 单位转换为亿m³
    return W


# 画图函数
def draw_pic_flood(QIN, PQJ3, PQJ1, PQJ2, QF, DT, time_roll, FDNO,newdata):
    # time_index = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), DT))
    time_index = DT.tolist()
    if time_roll <= 0:
        time_roll = n_in_timestep
    else:
        time_roll = time_roll - 1
    len_time = len(time_index[0:time_roll])
    fig = plt.figure(figsize=(15, 7))
    left, bottom, width, height = 0.08, 0.05, 0.9, 0.65
    ax1 = fig.add_axes([left, bottom, width, height])
    # plt.gca()函數獲得當前坐標軸，然後才能設置參數或作圖，plt.plot()內部實現了這一步驟
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%m/%d/%H:%M'))  # 設置x軸主刻度顯示格式（日期）
    plt.gca().xaxis.set_major_locator(dates.HourLocator(interval=12))  # 設置x軸主刻度間距
    ax1.plot(time_index, QIN, color='r', linewidth=1.5, label='observed flood')
    # ax1.plot(time_index, QF_zhexi, color='blue', linestyle='--', linewidth=1.5, label='预报流量')
    ax1.plot(time_index, QF, color='black', linestyle='--', linewidth=1.5, label='Transformer')

    ax1.plot(time_index, newdata, color='blue', linestyle='--', linewidth=1.5, label='Diffusion+Transformer')

    ax1.plot([time_index[len_time], time_index[len_time]], [QIN[len_time], 0], 'k--', linewidth=2)
    plt.axvline(time_index[len_time], linestyle='--', linewidth=2)
    plt.xticks(rotation=45)
    ax1.set_ylabel('流量(${m^3}/s$)', fontproperties=font_song, rotation=90, fontsize=15)
    plt.tick_params(labelsize=10)
    plt.legend(loc=2, fontsize=10)
    plt.grid(ls='--')
    left, bottom, width, height = 0.08, 0.74, 0.9, 0.08
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.bar(range(0, len(time_index)), height=PQJ1, alpha=1, color="r", label="PQJ1")
    ax2.invert_yaxis()
    plt.legend()
    left, bottom, width, height = 0.08, 0.82, 0.9, 0.08
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.bar(range(0, len(time_index)), height=PQJ2, alpha=1, color="g", label="PQJ2")
    ax2.invert_yaxis()
    plt.legend()
    left, bottom, width, height = 0.08, 0.90, 0.9, 0.08
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.bar(range(0, len(time_index)), height=PQJ3, alpha=1, color="b", label="PQJ3")
    ax2.invert_yaxis()
    plt.legend()
    #ax2.set_title('%s号洪水预报' % (FDNO), fontproperties=font_song, fontsize=20)
    plt.savefig("./pic/{}.png".format(FDNO), dpi=300, bbox_inches='tight')  # 解决图片不清晰，不完整的问题


#  注意力机制热力图
def draw_attention(attention, DT, time_roll):
    time_index = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), DT))
    time_roll = n_in_timestep
    time_index = time_index[time_roll:]
    # fig, ax = plt.subplots()  # 将元组分解为fig和ax两个变量
    fig = plt.figure(figsize=(15, 15), frameon=True)  # [batch_size, n_step]
    ax = fig.add_subplot(111)
    ax1 = ax.imshow(attention.cpu().data, cmap='viridis', aspect='auto')
    # 设置x轴刻度间隔
    ax.set_xticks(np.arange(attention.shape[1]))
    # 设置y轴刻度间隔
    ax.set_yticks(np.arange(attention.shape[0]))
    # 设置x轴标签
    ax.set_xticklabels([i for i in range(time_roll)])
    # 设置y轴标签
    ax.set_yticklabels(time_index)
    # 设置标签 旋转45度 ha有三个选择：right,center,left（对齐方式）
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # plt.yticks(rotation=45)
    # ax.set_xticklabels([''] + ['first_word', 'second_word', 'third_word'], fontdict={'fontsize': 14}, rotation=90)
    # ax.set_yticklabels([''] + ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6'], fontdict={'fontsize': 14})
    plt.colorbar(ax1, ax=ax)
    plt.show()


'''
画热力图中间格子的数值
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")
j,i:表示坐标值上的值
harvest[i, j]表示内容
ha有三个选择：right,center,left（对齐方式）
va有四个选择：'top', 'bottom', 'center', 'baseline'（对齐方式）
color:设置颜色
'''
