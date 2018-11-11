from hupu.analisys.dbconnection.getMySQL import getConnection
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
def timestamp2time(timestamp):
    time_local = time.localtime(timestamp)
    # 转换成新的时间格式(2016-05-05 20:28:54)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return dt
def showDataInTimeLine(datadf, stepNum):
    t1 = time.time()
    startTime = int(np.percentile(datadf['timestamp'], 10))
    endTime = int(np.percentile(datadf['timestamp'], 90))
    timeStep = int(round((endTime-startTime)/float(stepNum)))
    print(startTime, endTime, timeStep)
    timePoints = range(startTime, endTime + timeStep, timeStep)
    timeSlices = [[x, x+timeStep] for x in timePoints]
    resList = [[line[0], 0] for line in timeSlices]
    count = 0
    datadf.sort_values(by=['timestamp'], ascending=False)
    for i in range(len(datadf['timestamp'])):
        for j in range(len(timeSlices)):
            #print(timeSlices[j])
            if timeSlices[j][0]<=datadf['timestamp'].iloc[i]<timeSlices[j][1]:
                resList[j][1] += 1
    resList = np.array(resList)
    #print(resList)
    t2 = time.time()
    print('整理数据阶段耗时为', t2 - t1, '秒。')
    f = plt.figure()
    ax = plt.gca()

    plt.plot(resList[:,0], resList[:,1])
    ax.set_xticks(resList[:,0])
    xlabels = list(map(lambda x:timestamp2time(x[0]).split(' ')[0] + ' to '\
                       + timestamp2time(x[1]).split(' ')[0], timeSlices))
    ax.set_xticklabels(xlabels)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def numOfPosts():
    mysqlcon = getConnection()
    mysqlcon.setConnecter(host='192.168.1.199',port=3306, user="root",\
                            passwd='1q2w3e4r', db='test',\
                             use_unicode=True, charset="utf8")

    sqlstr = ""
    sqlstr += 'select cast(post_id as signed int), post_time '
    sqlstr += 'from hupu_bxj_advocate_posts_1'
    t1 = time.time()
    data = mysqlcon.queryWithReturn(sqlstr)
    # print(len(data),data)
    data = np.array(data)
    df = pd.DataFrame(data, columns = ['data', 'timestamp'])
    t2 = time.time()
    showDataInTimeLine(df, 10)
    print('查询数据阶段耗时为', t2-t1, '秒。')

def numOfUsers():
    t1 = time.time()
    mysqlcon = getConnection()
    mysqlcon.setConnecter(host='192.168.1.199',port=3306, user="root",\
                            passwd='1q2w3e4r', db='test',\
                             use_unicode=True, charset="utf8")

    sqlstr = ""
    sqlstr += 'select cast(uid as signed int), post_time '
    sqlstr += 'from hupu_bxj_advocate_posts_1 '
    # sqlstr += ' where post_time>1059667200 and post_time<1059667200 + 86400*36'
    data1 = mysqlcon.queryWithReturn(sqlstr)
    sqlstr = ""
    sqlstr += 'select cast(uid as signed int), post_time '
    sqlstr += ' from hupu_bxj_foll_posts_1'
    # sqlstr += ' where post_time>1059667200 and post_time<1059667200 + 86400*36'
    data2 = mysqlcon.queryWithReturn(sqlstr)
    data = data1 + data2
    #print(data)
    data = np.array(data)
    df = pd.DataFrame(data, columns = ['data', 'timestamp'])
    t2 = time.time()
    showDataInTimeLine(df, 20)
    print('查询数据阶段耗时为', t2-t1, '秒。')

def searchDictByValue(aDict, value):
    for key in aDict:
        if aDict[key]==value:
            return key

def numOfEachUser():
    t1 = time.time()
    mysqlcon = getConnection()
    mysqlcon.setConnecter(host='192.168.1.199', port=3306, user="root", \
                          passwd='1q2w3e4r', db='test', \
                          use_unicode=True, charset="utf8")

    sqlstr = ""
    sqlstr += 'select cast(uid as signed int), post_time '
    sqlstr += 'from hupu_bxj_advocate_posts_1 '
    # sqlstr += ' limit 10000'
    sqlstr += ' where post_time>1041350400 and post_time<1072886400'
    data1 = mysqlcon.queryWithReturn(sqlstr)
    sqlstr = ""
    sqlstr += 'select cast(uid as signed int), post_time '
    sqlstr += ' from hupu_bxj_foll_posts_1'
    sqlstr += ' where post_time>1041350400 and post_time<1072886400'
    # sqlstr += ' limit 10000'

    data2 = mysqlcon.queryWithReturn(sqlstr)
    data = data1 + data2
    res = {}
    for line in data:
        if line[0] in res:
            res[line[0]] += 1
        else:
            res[line[0]] = 1
    vs = list(res.values())
    print(vs)
    stastics = {'人均发帖数': np.mean(vs),
                '用户发帖数中位数':np.median(vs),
                '用户发帖数四分位数':[np.percentile(vs, 25), np.percentile(vs, 75)],
                '发帖最多的用户':{'用户id':searchDictByValue(res, np.max(vs)) , '帖子数': np.max(vs)}}
    print(stastics)
    plt.hist(vs,100)
    plt.show()
    return stastics


if __name__ == '__main__':
    # numOfPosts()
    #numOfUsers()
    numOfEachUser()