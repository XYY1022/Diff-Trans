import pandas as pd
import os
import math
import xlrd
from matplotlib import pyplot,dates, rcParams
from matplotlib.font_manager import FontProperties
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

"""
参数设置
"""
ratio = 10 #保留前ratio%个数据
m_size = 4 #散点的点的大小


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

# 画图
def draw_pic(inv_y, hehe,PQJ, time_index, i):#oringin,iris1['安康入库2'],iris1['安康数据修正'],iris1['时间'],i
    fig = plt.figure(figsize=(15, 5.5))

    # time_index = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"), time_index))
    plt.plot(time_index, inv_y, label="入库流量", linewidth=2.5)
    plt.tick_params(labelsize=15)
    plt.plot(time_index, hehe, label="入库流量（平滑）")
    plt.title('%s号洪水' % (i), fontsize=20)
    pyplot.legend(fontsize=15)
    plt.savefig("./pic_data_process/{}.png".format(i), dpi=300, bbox_inches='tight')
    #pyplot.show()

def draw_pic2(QIN, QIN_P,PQJ, time_index, FDNO,n_in_timestep):
    #创建存储图片的文件夹
    if os.path.isdir('./pic_data_process') == False:
        os.makedirs('pic_data_process')
    # time_index = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), DT))
    # if time_roll <= 0:
    time_roll = n_in_timestep
    # else:
    #     time_roll = time_roll - 1
    len_time = len(time_index[0:time_roll])
    fig = plt.figure(figsize=(15, 7))
    left, bottom, width, height = 0.08, 0.1, 0.9, 0.75
    ax1 = fig.add_axes([left, bottom, width, height])
    # plt.gca()函數獲得當前坐標軸，然後才能設置參數或作圖，plt.plot()內部實現了這一步驟
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%m/%d/%H:%M'))  # 設置x軸主刻度顯示格式（日期）
    plt.gca().xaxis.set_major_locator(dates.HourLocator(interval=12))  # 設置x軸主刻度間距
    ax1.plot(time_index, QIN, linewidth=2.5,marker='o',markersize=m_size, label='入库流量')
    ax1.plot(time_index, QIN_P,  linewidth=1.5, marker='o',markersize=m_size, label='入库流量（修正）')
    #ax1.plot(time_index, QC, color='g', linestyle='--', linewidth=0.5, label='碗米坡出库')
    #ax1.plot([time_index[len_time], time_index[len_time]], [QIN[len_time], 0], 'k--', linewidth=2)
    #plt.axvline(time_index[len_time], linestyle='--', linewidth=2)
    plt.xticks(rotation=45)
    ax1.set_ylabel('流量(${m^3}/s$)', fontproperties=font_song, rotation=90, fontsize=15)
    plt.tick_params(labelsize=10)
    plt.legend(loc=2, fontsize=10)
    plt.grid(ls='--')
    left, bottom, width, height = 0.08, 0.85, 0.9, 0.1
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.bar(range(0,len(time_index)), height=PQJ, alpha=1, color='g')
    ax2.invert_yaxis()
    ax2.set_ylabel('降雨量(mm)', labelpad=12, rotation=90, fontsize=15)
    # ax2.set_title('%s至%s 日径流预报' % (str(DT[0]), str(DT[-1])), FontProperties=font_song, fontsize=20)
    if FDNO != 0:
        ax2.set_title('%s号洪水' % (FDNO), fontproperties=font_song, fontsize=20)
    # else:
    #     ax2.set_title('%s至%s 径流预报' % (str(DT[0]), str(DT[-1])), fontproperties=font_song, fontsize=20)
    plt.xticks([])
    plt.tick_params(labelsize=15)
    plt.grid(ls='--')
    plt.savefig("./pic_data_process/{}.png".format(FDNO), dpi=300, bbox_inches='tight')  # 解决图片不清晰，不完整的问题
    # plt.show()

# 滑动平均
def smooth(a, WSZ):#oringin, 3
    a = np.squeeze(a)
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))

def smooth_data():
    file_path = 'allflood.xlsx'
    n_in_timestep = 7
    # 获得所有洪水编号并按照时间前后排序
    df = pd.read_excel(file_path)
    df1 = df.loc[:, ['FDNO']]
    FDNO = df1['FDNO'].tolist()
    temp = set(FDNO)
    FDNO = list(temp)
    FDNO.sort()
    FDNO = np.array(FDNO)
    for i in FDNO:
        #获取单场洪水数据
        a = df.loc[df['FDNO'] == i]
        #对单场洪水数据进行平滑处理
        origin = a['QIN'].copy()
        length = math.floor(len(origin) / ratio)
        after = smooth(origin,3)
        test = a.sort_values(by='QIN',ascending=False)
        test = test.iloc[0:length,:]
        a['QIN'] = after
        #还原最大的length场数据
        for j in test['TM']:
            a.loc[a['TM'] == j] = test.loc[test['TM'] == j]
        #draw_pic(origin, a['QIN'],a['PQJ'] ,a['TM'], i)
        draw_pic2(origin,a['QIN'],a['PQJ'],a['TM'],i,n_in_timestep)
        if i == FDNO[0]:
            b = a['QIN'].tolist()
            qin_processed = np.array(b)
        else:
            b = a['QIN'].tolist()
            qin_processed = np.append(qin_processed,b,axis=0)
    col_name=df.columns.tolist()
    col_name.insert(4,'QIN_PROCESSED')
    df.reindex(columns=col_name)
    df['QIN_PROCESSED'] = qin_processed
    df.to_excel(file_path,index=False)

smooth_data()