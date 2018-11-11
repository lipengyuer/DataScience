from dbconnection.getMySQL import getConnection
import time
mysqlcon = getConnection()
mysqlcon.setConnecter(host='192.168.1.198', port=3306, user="root",
                      passwd='1q2w3e4r', db='test',
                      use_unicode=True, charset="utf8")
# t1 = time.time()
# print("正在统计主贴数量")
# sqlstr = "select count(uid) from hupu_bxj_advocate_posts_1"
# data1 = mysqlcon.queryWithReturn(sqlstr)
# sqlstr = "select count(uid) from hupu_bxj_advocate_posts"
# data2 = mysqlcon.queryWithReturn(sqlstr)
# t2 = time.time()
# print("主贴数据总数是", data1[0] + data2[0], '条，', "耗时", int(t2-t1), "秒。")
#
t1 = time.time()
print('正在统计跟帖数量')
sqlstr = "select count(distinct uid) from hupu_bxj_advocate_posts_1"
data1 = mysqlcon.queryWithReturn(sqlstr)
sqlstr = "select count(distinct uid) from hupu_bxj_foll_posts_1"
data2 = mysqlcon.queryWithReturn(sqlstr)
t2 = time.time()
print("用户总数是", data1[0], data2[0], '个', "耗时", int(t2-t1), "秒。")
#
#
# print("正在最新胡哦哦去的帖子")
t1 = time.time()
sqlstr = "select max(cast(post_id as signed int)) from hupu_bxj_advocate_posts_1"
data1 = mysqlcon.queryWithReturn(sqlstr)
# sqlstr = "select max(cast(post_id as signed int)) from hupu_bxj_advocate_posts"
# data2 = mysqlcon.queryWithReturn(sqlstr)
t2 = time.time()
print("最新主贴是", data1[0], '条，', "耗时", int(t2-t1), "秒。")
#
