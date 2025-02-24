import numpy as np
import pandas as pd

# path处填入npy文件具体路径
npfile = np.load("E:/Diffusion-TS-main/OUTPUT/etth/samples/etth_ground_truth_24_test.npy")
# 将npy文件的数据格式转化为csv格式
np_to_csv = pd.DataFrame(data=npfile)
# 存入具体目录下的np_to_csv.csv 文件
np_to_csv.to_csv('etth_ground_truth_24_train.csv')