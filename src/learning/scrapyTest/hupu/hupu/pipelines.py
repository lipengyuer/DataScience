# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pymongo
from multiprocessing import Queue
from multiprocessing import Process


def getConnectionMongo():
    conn = pymongo.MongoClient('192.168.1.199', 27017)
    db = conn.hupuPost  # 连接mydb数据库，没有则自动创建
    return db

class HupuPipeline(object):
    def __init__(self):
        self.advPostQueue = Queue(maxsize=500)
        self.follPostQueue = Queue(maxsize=500)
        self.db = getConnectionMongo()

    def insertThem(self, q, collectionName):
        dataList = []
        while q.qsize() > 0:
            data = q.get()
            dataList.append(dict(data))
        # if dataList[0]['type']=='advPost':
        #     print("存储主贴数据", collectionName, dataList)
        try:
            self.db[collectionName].insert_many(dataList)
            dataList = None
        except Exception as e:
            with open('error.txt', 'a+') as f:
                f.write("存储失败" + collectionName + ' ' + str(dataList))


    def process_item(self, item, spider):
        # print("获取到的数据是", item['floor_num'])
        if item['type']=='advPost':
            # print("主贴队列长度是", self.advPostQueue.qsize())
            self.advPostQueue.put(item)
        else:
            # print("跟帖队列长度是", self.follPostQueue.qsize())
            self.follPostQueue.put(item)

        if self.advPostQueue.qsize() == 10:
            self.insertThem(self.advPostQueue, 'advPosts')
        if self.follPostQueue.qsize() == 10:
            self.insertThem(self.follPostQueue, 'follPosts')
        return item

