#队数据集进行扩增，
import random
import pandas as pd
import numpy as np

#将两个样本的特征取值进行随机对换，生成第三个样本
def generate_new_by_fix_two(sample_1, sample_2):
    real_samples = zip(sample_1, sample_2)
    sample_new = []
    for i in range(len(real_samples)):
        if random.uniform(0,1)>0.5:
            sample_new

def reset_index(df):
    df = df.reset_index(drop=True)
    df = df.reindex(list(range(len(df))))   
    return df 

def worker(data, hps, mode=0):
    if mode==1:
        data_aug1 = data.apply(lambda x: x*random.uniform(0, 0.1))
    if mode==2:
        data_aug1 = data.apply(lambda x: x*0.1*np.random.random())
    if mode==3:
        data_aug1 = data.apply(lambda x: x*0.3*np.random.random())
    data_aug1 = data + data_aug1
    data_aug1 = reset_index(data_aug1)
    hps = reset_index(hps)
    left_index = data_aug1[(data_aug1['birth']<25) | (data_aug1['birth']>70)].index.tolist()
    data_aug1 = data_aug1.iloc[left_index]
    left_hps1 = hps.iloc[left_index]
   
    return data_aug1, left_hps1

#首先将年轻人和老年人的数据进行扩增。
def aug_data_age_group(data, hps):
    print("开始扩增数据。")
    data1, hps1 = worker(data, hps, mode=1)
    data2, hps2 = worker(data, hps, mode=2)
    data3, hps3 = worker(data, hps, mode=3)
    print("扩增第一波数据量是", len(data1))
    data_new  = pd.concat((data,data1, data2, data3), axis=0)
    hps_new  = pd.concat((hps, hps1, hps2, hps3), axis=0) 
    print("扩增第二波")
    import copy
    data_new = copy.deepcopy(data_new)
    data_new = data_new.reset_index(drop=True)
    data_new = data_new.reindex(list(range(len(data_new))))
    print("数据扩增已经完成，数据量是", len(data_new))
    return data_new, hps_new


    