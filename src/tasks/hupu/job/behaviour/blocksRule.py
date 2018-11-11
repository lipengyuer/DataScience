#用关联规则分析用户混迹的板块
import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
path_src = path.split('src')[0]
sys.path.append(path_src + "src")#项目的根目录
import time
from pymongo import MongoClient
from pyspark import SparkContext, SparkConf
import re
from pyspark.mllib.fpm import FPGrowth
from analysis.algorithm.fp_growth import mineRuleWithFPTree
from matplotlib import pyplot as plt
def getBlocksData(userInfoCollectionName):
    MONGO_IP = '192.168.1.198'
    MONGO_PORT = 27017
    MONGO_DB_NAME = "hupu"
    MONGO_IP, MONGO_PORT ,dbname, username, password = MONGO_IP, MONGO_PORT, \
                                                          MONGO_DB_NAME, 'root', '1q2w3e4r'
    conn = MongoClient(MONGO_IP, MONGO_PORT)
    db = conn[dbname]
    db.authenticate(username, password)
    data = db[userInfoCollectionName].find({}, {"blockList": 1})
    res = []
    c = 0
    for line in data:
        c+= 1
        # if c==500:
        #     break
        print(line['_id'], c)
        res.append(line['blockList'])
    return res
import pickle
if __name__ == '__main__':
    local = True
    # data = getBlocksData('oriUserFeatureAll_adv')
    # pickle.dump(data, open('blockData.pkl', 'wb'))
    data = pickle.load(open('blockData.pkl', 'rb'))
    print("人数是", len(data))
    # fig = plt.subplot()
    # blockNumList = list(map(lambda x: len(x), data))
    # print(set(blockNumList))
    # plt.hist(blockNumList, bins=50)
    # plt.show()
    blockFreqMap = {}
    for line in data:
        for block in line:
            blockFreqMap[block] = blockFreqMap.get(block, 0) + 1
    blockFreqList = sorted(blockFreqMap.items(), key=lambda x: x[1], reverse=True)
    print("出现频率是", blockFreqList)
    if local:
        print(len(data))
        # data = data[:100000]
        res = mineRuleWithFPTree(data, suport_min=2000)
        for line in res:
            if len(line)>1:
                print(line)
    else:
        conf = SparkConf().setAppName("hupu_block")#配置spark任务的基本参数
        sc = SparkContext(conf = conf)#创建sc对象

        rdd = sc.parallelize(data, 2)
        model = FPGrowth.train(rdd, 0.6, 2)
        res = sorted(model.freqItemsets().collect())
