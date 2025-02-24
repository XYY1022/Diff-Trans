import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import utils
from datetime import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import read_excel
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
# from keras.utils import plot_model
from sklearn.preprocessing import RobustScaler
from keras.layers import Dropout
import keras.backend as K
from keras.callbacks import LearningRateScheduler


def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


# 建模
# create the nn model
def _create_model(train_X, train_y, test_X, test_y, n_neurons, n_batch, n_epochs, is_stateful, has_memory_stack,
                  loss_function, optimizer_function, draw_loss_plot, output_col_name, verbose):
    """
    train_X: train inputs
    train_y: train targets
    test_X: test inputs
    test_y: test targets
    n_neurons: number of neurons for LSTM nn
    n_batch: nn batch size
    n_epochs: training epochs
    is_stateful: whether the model has memory states
    has_memory_stack: whether the model has memory stack
    loss_function: the model loss function evaluator
    optimizer_function: the loss optimizer function
    draw_loss_plot: whether to draw the loss history plot
    output_col_name: name of the output/target column to be predicted
    verbose: whether to output some debug data
    """
    # design network
    model = Sequential()
    if is_stateful:
        # calculate new compatible batch size
        for i in range(n_batch, 0, -1):
            if train_X.shape[0] % i == 0 and test_X.shape[0] % i == 0:
                if verbose and i != n_batch:
                    print(
                        "\n*In stateful network, batch size should be dividable by training and test sets; had to decrease it to %d." % i)
                n_batch = i
                break
        model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True,
                       return_sequences=has_memory_stack))
        if has_memory_stack:
            model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True))
    else:
        model.add(LSTM(n_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    model.summary()  # 打印模型
    # plot_model(model, show_shapes=True, to_file='lstm_model.png')  # 绘制模型结构图，并保存成图片
    model.compile(loss=loss_function, optimizer=optimizer_function, metrics=["accuracy"])
    if verbose:
        print("")
    # fit network
    losses = []
    val_losses = []
    if is_stateful:
        for i in range(n_epochs):
            # checkpoint
            filepath = './my_model_in time step_%d_out_timestep_%d.h5' % (n_in_timestep, n_out_timestep)
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                         mode='min')
            callbacks_list = [checkpoint]
            history = model.fit(train_X, train_y, epochs=1, batch_size=n_batch,
                                validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=callbacks_list)
            if verbose:
                print("Epoch %d/%d" % (i + 1, n_epochs))
                print("loss: %f - val_loss: %f" % (history.history["loss"][0], history.history["val_loss"][0]))
            losses.append(history.history["loss"][0])
            val_losses.append(history.history["val_loss"][0])
            model.reset_states()
    else:
        filepath = './my_model_in time step_%d_out_timestep_%d.h5' % (n_in_timestep, n_out_timestep)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='auto')
        LROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto',
                                        epsilon=0.0001, cooldown=0, min_lr=0)
        callbacks_list = [checkpoint, LROnPlateau]
        history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch,
                            validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=callbacks_list)
    if draw_loss_plot:
        pyplot.plot(history.history["loss"] if not is_stateful else losses, label="Train Loss (%s)" % output_col_name)
        pyplot.plot(history.history["val_loss"] if not is_stateful else val_losses,
                    label="Test Loss (%s)" % output_col_name)
        pyplot.legend()
        pyplot.show()
    print(history.history)
    # model.save('./my_model_%s.h5'%datetime.datetime.now())
    return model, n_batch


# 检验模型结果
def _make_prediction(model, train_X, train_y, test_X, test_y, compatible_n_batch, n_intervals, n_features, scaler,
                     draw_prediction_fit_plot, output_col_name, verbose):
    """
    train_X: train inputs
    train_y: train targets
    test_X: test inputs
    test_y: test targets
    compatible_n_batch: modified (compatible) nn batch size
    n_intervals: number of time lags (intervals) to use in each neuron
    n_features: number of features (variables) per neuron
    scaler: the scaler object used to invert transformation to real scale
    draw_prediction_fit_plot: whether to draw the the predicted vs actual fit plot
    output_col_name: name of the output/target column to be predicted
    verbose: whether to output some debug data
    """
    if verbose:
        print("")

    yhat = model.predict(test_X, batch_size=compatible_n_batch, verbose=1 if verbose else 0)
    test_X = test_X.reshape((test_X.shape[0], n_intervals * n_features))

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, (1 - n_features):]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, (1 - n_features):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

    # calculate average error percentage
    avg = np.average(inv_y)
    error_percentage = rmse / avg
    if verbose:
        print("")
        print("Test Root Mean Square Error: %.3f" % rmse)
        print("Test Average Value for %s: %.3f" % (output_col_name, avg))
        print("Test Average Error Percentage: %.2f/100.00" % (error_percentage * 100))

    if draw_prediction_fit_plot:
        pyplot.plot(inv_y, label="Actual (%s)" % output_col_name)
        pyplot.plot(inv_yhat, label="Predicted (%s)" % output_col_name)
        pyplot.legend()
        pyplot.show()

    return inv_y, inv_yhat, rmse, error_percentage


"""
数据预处理参数设置
"""
# !input

file_path = "./allflood.xlsx"
header_row_index = 0
index_col_name = "TM"
col_to_predict = "QIN_PROCESSED"
cols_to_drop = None
n_in_timestep = 5
n_out_timestep = 1
verbose = 2
dropnan = True
scale_range = (0, 1)
alpha = 0.5#更改归一化区间
"""
网络模型超参数设置
"""
# !input
file_path = file_path
n_neurons = 50  # 50
n_batch = 50    # 50
n_epochs = 100  #100
is_stateful = False
has_memory_stack = False
loss_function = 'mae'
optimizer_function = 'Adam'
draw_loss_plot = True
train_percentage = 0.7#0.51
draw_prediction_fit_plot = True
order = ["QIN_PROCESSED", "PQJ",'QWMP', "TM"]

# 程序
# 读取数据
col_names, values, n_features, output_col_name = utils.load_dataset(file_path, header_row_index, index_col_name,
                                                                    col_to_predict, cols_to_drop, order)

all = np.array(values, copy=True)  # values未归一化之前的备份
# 归一化
scaler, scaled = utils.scale_dataset_train(values,order,file_path,alpha)
#scaler,scale_out, scaled = utils._scale_dataset(values,scale_range)
# keras指定格式
agg1 = utils.series_to_supervised(scaled, n_in_timestep, n_out_timestep, dropnan, col_names)
print("\nagg1.shape:", agg1.shape, )
# 分成测试集训练集
train_X, train_Y, test_X, test_Y = utils.split_data_to_train_test_sets(agg1.values, n_in_timestep, n_features,
                                                                       train_percentage, verbose)
# 训练
model, compatible_n_batch = _create_model(train_X, train_Y, test_X, test_Y, n_neurons, n_batch, n_epochs,
                                          is_stateful, has_memory_stack, loss_function, optimizer_function,
                                          draw_loss_plot, output_col_name, verbose)
# actual_target, predicted_target, error_value, error_percentage = _make_prediction(model, train_X, train_Y,
#                                                                                   test_X, test_Y, compatible_n_batch,
#                                                                                   n_in_timestep, n_features, scaler,
#                                                                                   draw_prediction_fit_plot,
#                                                                                   output_col_name,
#                                                                                   verbose)
