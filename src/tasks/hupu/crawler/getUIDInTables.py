from hupu.analisys.dbconnection.getMySQL import getConnection
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import pickle
def getAllUIDs():
    t1 = time.time()
    mysqlcon = getConnection()
    mysqlcon.setConnecter(host='192.168.1.199', port=3306, user="root", \
                          passwd='1q2w3e4r', db='test', \
                          use_unicode=True, charset="utf8")

    sqlstr = ""
    sqlstr += 'select cast(uid as signed int) '
    sqlstr += 'from hupu_bxj_advocate_posts_1 '
    # sqlstr += ' limit 10000'
    # sqlstr += ' where post_time>1059667200 and post_time<1059667200 + 86400*36'
    data1 = mysqlcon.queryWithReturn(sqlstr)
    sqlstr = ""
    sqlstr += 'select cast(uid as signed int) '
    sqlstr += ' from hupu_bxj_foll_posts_1'
    # sqlstr += ' where post_time>1059667200 and post_time<1059667200 + 86400*36'
    # sqlstr += ' limit 10000'

    data2 = mysqlcon.queryWithReturn(sqlstr)
    data = data1 + data2
    data = map(lambda x: x[0], data)
    data = list(data)
    data = set(data)

    print('目前为止出现的用户数量是', len(data))
    return data


if __name__ == '__main__':
    uids = getAllUIDs()
    pickle.dump(uids, open('uids.pkl', 'wb'))