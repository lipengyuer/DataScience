from  http import cookiejar
from  urllib import request
import config
# import ssl
#
# ssl._create_default_https_context=ssl._create_unverified_context

def myLogin():
    # 设置保存cookie的文件，同级目录下的cookie.txt
    filename = r'C:\Users\Administrator\PycharmProjects\test\src\crawler\cookies.txt'  # cookie位置，这里使用的是火狐浏览器的cookie
    # 声明一个MozillaCookieJar对象实例来保存cookie，之后写入文件
    cookie = cookiejar.MozillaCookieJar()
    cookie.load(filename, ignore_discard=True, ignore_expires=True)
    # 利用urllib2库的HTTPCookieProcessor对象来创建cookie处理器
    handler = request.HTTPCookieProcessor(cookie)
    # 通过handler来构建opener
    opener = request.build_opener(handler)
    return opener

from pymongo import MongoClient
def getConnectionMongo():
    conn = MongoClient(config.MONGO_IP, 27017)
    db = conn.hupu  #连接mydb数据库，没有则自动创建
    my_set = db.hupuUserInfo#使用test_set集合，没有则自动创建
    return my_set