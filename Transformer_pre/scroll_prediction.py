import os

import img2pdf

import AIfloodscore as a_fs
import json
import time
import logging
import openpyxl
import pandas as pd
import numpy as np
from datetime import datetime
import datetime as dt
import threading
import time
import utils


# 人工智能洪水调试
def ai_flood():
    time_start = '2001-10-24 10:00:00'
    # data_process.smooth_data()
    alpha = 0.5
    # 读取洪水编号
    file_path = './allflood.xlsx'
    df = pd.read_excel(file_path)
    OT = df['OT'].tolist()
    PQJ3_ = df['PQJ3'].tolist()
    PQJ1_ = df['PQJ1'].tolist()
    PQJ2_ = df['PQJ2'].tolist()
    DT_ = df['TM']
    df = df.loc[:, ['FDNO']]
    a = df['FDNO'].tolist()
    b = set(a)
    FDNO = list(b)
    FDNO.sort()
    FDNO = np.array(FDNO)
    key = ['洪水编号', '纳什系数', '决定系数', '均方根误差', '相对误差', '洪水总量', '预报总量', '总量差值', '实际洪峰', '预报洪峰', '洪峰差值']
    a = dict([(k, []) for k in key])
    pred = np.load(r'E:\Transformer\results\real_prediction.npy')
    newdata_ = np.load(r'E:\Transformer\results\newdata.npy')
    j = 0
    time_roll = 4
    for i in FDNO:
        """前端回传参数"""
        num = df['FDNO'].value_counts()[i]
        border1 = j
        border2 = j + num
        values = pred[border1:border2]
        QF = list(np.array(values))
        QIN = OT[border1:border2]
        PQJ3 = PQJ3_[border1:border2]
        PQJ1 = PQJ1_[border1:border2]
        PQJ2 = PQJ2_[border1:border2]
        DT = DT_[border1:border2]
        newdata = newdata_[border1:border2]
        j = border2
        """画图"""
        #utils.draw_pic_flood(QIN, PQJ3,PQJ1,PQJ2, QF, DT,time_roll, i,newdata)
        """nash系数"""
        nash = utils.get_nash(QF, QIN, time_roll)
        R2 = utils.get_R2(QF, QIN, time_roll)
        RMSE = utils.get_RMSE(QF, QIN, time_roll)
        RE = utils.get_RE(QF, QIN, time_roll)
        max_flood = max(QIN[time_roll:])  # 实测洪峰
        max_flood_pred = max(QF[time_roll:])  # 预测洪峰
        total_flood = utils.get_all_flood(QIN, time_roll)  # 实测洪水总量
        total_flood_pred = utils.get_all_flood(QF, time_roll)  # 预测洪水总量
        a['洪水编号'].append(i)
        a['纳什系数'].append(round(nash, 3))
        a['决定系数'].append(round(R2, 3))
        a['均方根误差'].append(round(RMSE, 3))
        a['相对误差'].append(round(RE, 3))
        a['洪水总量'].append(round(total_flood, 2))
        a['预报总量'].append(round(total_flood_pred, 2))
        a['总量差值'].append(round((total_flood_pred - total_flood) / total_flood, 3))
        a['实际洪峰'].append(round(max_flood, 2))
        a['预报洪峰'].append(round(max_flood_pred, 2))
        a['洪峰差值'].append(round((max_flood_pred - max_flood) / max_flood, 3))
    keys = list(a.keys())
    value = list(a.values())
    result_excel = pd.DataFrame()
    for num, i in enumerate(key):
        result_excel[i] = value[num]
    hfmax = result_excel['洪峰差值'].max()
    hfmin = result_excel['洪峰差值'].min()
    print(result_excel[result_excel['洪峰差值'].isin([hfmax])])
    print(result_excel[result_excel['洪峰差值'].isin([hfmin])])
    nashimean = result_excel['纳什系数'].mean()
    df = result_excel.abs()
    dfmean = df['洪峰差值'].mean()
    dfmax = df['洪峰差值'].max()
    nashi = df['纳什系数'][df['纳什系数'] >= 0.9].count()
    zongliang = df['总量差值'][df['总量差值'] <= 0.1].count()
    hongfeng = df['洪峰差值'][df['洪峰差值'] <= 0.1].count()
    print("最大差值为" + '%.2f%%' % (dfmax * 100))
    # print("平均差值为" + '%.2f%%' % (dfmean * 100))
    print("平均纳什系数为" + '%.3f' % nashimean)
    print("纳什系数大于等于0.9的场数为" + str(nashi))
    print("总量差值小于等于0.1的场数为" + str(zongliang))
    print("洪峰差值小于等于0.1的场数为" + str(hongfeng))
    weatherfile = "output.xlsx"
    writer = pd.ExcelWriter(weatherfile, engine='openpyxl')
    try:
        sheet = pd.read_excel("output.xlsx", sheet_name=None)
        sheet_name = str(int(list(sheet.keys())[-1]) + 1)
        result_excel.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(e)
        sheet_name = '1'
        result_excel.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()
    print("计算结束")
    # 转成pdf

    # photo_list = os.listdir(r"./pic")
    # photo_list = [os.path.join(r"./pic/", i) for i in photo_list if i.endswith("png")]
    # save_pdf_file = "output.pdf"  # 保存的PDF的名称
    # with open(save_pdf_file, "wb") as f:
    #    write_content = img2pdf.convert(photo_list)
    #    f.write(write_content)  # 写入文件
    # print("转换成功！")  # 提示语


if __name__ == "__main__":
    ai_flood()
