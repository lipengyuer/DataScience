import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
path_src = path.split('src')[0]
sys.path.append(path_src + "src")#项目的根目录
from pyhanlp import HanLP
from pyhanlp import *
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import time
from pymongo import MongoClient
import pandas as pd
#from analysis.algorithm import splitSentence, nlp
import splitSentence, nlp
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession,SQLContext
import runTime
import re

#向mongo插入一条数据
def getUIDList(collectionName = "oriUserFeatureSample_foll"):
    MONGO_IP = '192.168.1.198'
    MONGO_PORT = 27017
    MONGO_DB_NAME = "hupu"
    MONGO_IP, MONGO_PORT ,dbname, username, password = MONGO_IP, MONGO_PORT, \
                                                          MONGO_DB_NAME, 'root', '1q2w3e4r'
    conn = MongoClient(MONGO_IP, MONGO_PORT)
    db = conn[dbname]
    db.authenticate(username, password)
    collection = db[collectionName]
    uidList = list(collection.find({}, {"uid": 1}))
    uidList = list(map(lambda x: {'uid': x['uid']}, uidList))
    return uidList

def annalysisThem(data):
    uid, dataList = data[0], data[1]
    postList, oriUIDList  = [], []
    for line in dataList:
        postList.append(line[0])
        oriUIDList.append(line[1])
    res = {"uid": uid, "overturesNum": 0, "follPostNum": 0, "chatNumWithfollPoster": 0}
    for line in postList:
        if "交个朋友" in line or "交朋友" in line:
            res['overturesNum'] += 1
    for uid in oriUIDList:
        res['follPostNum'] += 1
        if uid==uid:
            res['chatNumWithfollPoster'] += 1
    return res

#向mongo插入一条数据
def saveRecord2Mongo(data, collectionName):
    MONGO_IP = '192.168.1.198'
    MONGO_PORT = 27017
    MONGO_DB_NAME = "hupu"
    MONGO_IP, MONGO_PORT ,dbname, username, password = MONGO_IP, MONGO_PORT, \
                                                          MONGO_DB_NAME, 'root', '1q2w3e4r'
    conn = MongoClient(MONGO_IP, MONGO_PORT)
    db = conn[dbname]
    db.authenticate(username, password)
    collection = db[collectionName]
    collection.insert(data, check_keys=False)

if __name__ == '__main__':
    conf = SparkConf().setAppName("hupu_user_feature_extraction")  # 配置spark任务的基本参数
    sc = SparkContext(conf=conf)
    sqlcontext = SQLContext(sc)
    #获取用户个人资料
    print("正在获取uid数据")
    uidList = pd.DataFrame(getUIDList())
    print("开始联表查询")
    uidListDF = sqlcontext.createDataFrame(uidList)
    #基于用户个人资料统计基本信息
    #与主贴信息连接，
    sql_str = ""
    sql_str += "select uid, post_id from hupu_bxj_advocate_post_dir"
    advPostData = sqlcontext.sql(sql_str).dropDuplicates(['post_id']).repartition(10000)
    usefulAdvPost = advPostData.join(uidListDF, how='left', on="uid").dropna()
    # # 然后与这些主贴的回复连接,
    sql_str = ""
    sql_str += "select post_id, content,cited_uid from hupu_bxj_foll_post_dir"
    follPostData = sqlcontext.sql(sql_str).repartition(10000)
    advFollPostData = follPostData.join(usefulAdvPost, how="left", on="post_id").dropna().repartition(100000)
    # # num = advFollPostData.count()
    # # print("最后的数据总量是", num)
    # # # 按照uid对回复分组之后，统计这些回复的情况
    advFollPostDataRDD = advFollPostData.rdd.map(lambda x: (x[2], x[1], x[3])).\
        groupByKey().map(annalysisThem)
    # print("记录数是",usefulAdvPost.count(), "表1的大小是", usefulAdvPost.take(2))
    # print("记录数是", advFollPostData.count(), "表1的大小是", advFollPostData.take(2))

    #存储数据
    advFollPostDataRDD.foreach(lambda x: saveRecord2Mongo(x, "onesFollPostFeature"))
    # print("记录数是", advFollPostDataRDD.count(), "表1的大小是", advFollPostDataRDD.take(2))
    #没毛病，这次我们需要知道一个用户的主贴的跟帖的情况。而原始回复数据里没有主贴对应用户uid,需要关联才能得到.
    #使用用户个人资料来开始，也是为了缩小数据规模.
    # spark2-submit --master yarn-client --executor-memory 5G --conf spark.pyspark.python=/opt/anaconda2/envs/python36/bin/python --conf spark.executorEnv.PYTHONHASHSEED=0 userFeatureExtract_onesFollPosts.py

