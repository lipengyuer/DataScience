from pymongo import MongoClient
from config import enviroment
def getConnectionMongo():
    conn = MongoClient(enviroment.MONGO_IP, 27017)
    db = conn.hupu  #连接mydb数据库，没有则自动创建
    my_set = db.hupuUserInfo#使用test_set集合，没有则自动创建
    return my_set