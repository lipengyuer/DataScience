import pymysql

class getConnection():
    def __init__(self):
        self.connector = None
        self.cur = None
    def setConnecter(self,host='192.168.1.111',port=3306, user="root",\
                            passwd='1q2w3e4r', db='lpy',\
                             use_unicode=True, charset="utf8"):
        self.connector = pymysql.connect(host=host, port=port, user=user, \
                                    passwd=passwd, db=db, \
                                    use_unicode=use_unicode, charset=charset)
        self.cur = self.connector.cursor()
        return self.connector

    def insertOne(self, sqlstr, data):
        self.cur.execute(sqlstr, data)
        self.connector.commit()

    def insertMany(self, sqlstr, dataList):
        self.cur.execute(sqlstr, dataList)
        self.connector.commit()

    def queryWithNoneReturn(self, sqlstr):
        self.cur.execute(sqlstr)
        self.connector.commit()

    def queryWithReturn(self, sqlstr):
        self.cur.execute(sqlstr)
        self.connector.commit()
        data = self.cur.fetchall()
        return data

    def close(self):
        self.connector.close()




