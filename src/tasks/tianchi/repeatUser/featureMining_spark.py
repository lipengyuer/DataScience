#寻找好的特征
from  pyspark import SparkContext, SQLContext
from pyspark import SparkConf
from pyspark.ml.feature import Word2Vec,CountVectorizer  
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors, VectorUDT
import pyspark.ml.evaluation as ev
from pyspark.sql.functions import rand, udf

conf = SparkConf().setAppName("tianchi_repeat_buyers")  
sc = SparkContext(conf=conf)   
sqlContext=SQLContext(sc) 

def log(line):
    with open('log.txt', 'a+') as f:
        f.write(line + '\n')
        
def processLog(logStr):
    
    if logStr:
        actions = logStr.split('#')
    else:
        actions = ''
    idMap = {'item_id': [], 'category_id': [], 'brand_id': []}
    for action in actions:
        s = action.split(':')
        if len(s)<5: continue
        [item_id, category_id, brand_id, _, _] = s
        idMap['item_id'].append(item_id)
        idMap['category_id'].append(category_id)
        idMap['brand_id'].append(brand_id)

    freqData = [list(idMap['item_id']), list(idMap['category_id']), list(idMap['brand_id'])]
    return freqData

def countIDFreq(df):
    def addOne(id_list):
        res = []
        for ID in id_list: res.append([ID, 1])
        return res
    
    log(str(df.take(2)))
    log_rdd = df.select('activity_log').rdd.map(lambda x: processLog(x[0]))
    item_id_term_freq = log_rdd.flatMap(lambda x:addOne(x[0])).reduceByKey(lambda x,y:x+y).collect()
    item_id_term_freq = dict(item_id_term_freq)
    item_id_doc_freq = log_rdd.flatMap(lambda x:addOne(x[0])).reduceByKey(lambda x,y:x).collect()
    item_id_doc_freq = dict(item_id_doc_freq)
    category_id_term_freq = log_rdd.flatMap(lambda x:addOne(x[1])).reduceByKey(lambda x,y:x+y).collect()
    category_id_term_freq = dict(category_id_term_freq)
    category_id_doc_freq = log_rdd.flatMap(lambda x:addOne(x[1])).reduceByKey(lambda x,y:x).collect()
    category_id_doc_freq = dict(category_id_doc_freq)
    brand_id_term_freq = log_rdd.flatMap(lambda x:addOne(x[2])).reduceByKey(lambda x,y:x+y).collect()
    brand_id_term_freq = dict(brand_id_term_freq)
    brand_id_doc_freq = log_rdd.flatMap(lambda x:addOne(x[2])).reduceByKey(lambda x,y:x).collect()
    brand_id_doc_freq = dict(brand_id_doc_freq)

    item_id_TFIDF = {}
    for id in item_id_term_freq:
        item_id_TFIDF[id] = item_id_term_freq[id]/item_id_doc_freq[id]

    category_id_TFIDF = {}
    for id in category_id_doc_freq:
        category_id_TFIDF[id] = category_id_term_freq[id]/category_id_doc_freq[id]
        
    brand_id_TFIDF = {}
    for id in brand_id_term_freq:
        brand_id_TFIDF[id] = brand_id_term_freq[id]/brand_id_doc_freq[id]
    
    def addIndex(lines):
        res = {}
        for i in range(len(lines)):
            res[lines[i][0]] = i
        return res
    
    with open('id_freq_data/item_id_freq.txt', 'w') as f:
        item_id_freq = sorted(item_id_TFIDF.items(), key=lambda x:x[1], reverse=True)
        first_line = addIndex(item_id_freq[:1000])
        f.write(str(first_line) + '\n')
        for line in item_id_freq:
            f.write(str(line[0]) + '\t' + str(line[1]) + '\n')
    
    with open('id_freq_data/category_id_freq.txt', 'w') as f:
        category_id_freq = sorted(category_id_TFIDF.items(), key=lambda x:x[1], reverse=True)
        first_line = addIndex(category_id_freq[:1000])
        f.write(str(first_line) + '\n')
        for line in category_id_freq:
            f.write(str(line[0]) + '\t' + str(line[1]) + '\n')
            
    with open('id_freq_data/brand_id_frq.txt', 'w') as f:
        brand_id_frq = sorted(brand_id_TFIDF.items(), key=lambda x:x[1], reverse=True)
        first_line = addIndex(brand_id_frq[:1000])
        f.write(str(first_line) + '\n')
        for line in brand_id_frq:
            f.write(str(line[0]) + '\t' + str(line[1]) + '\n')  
             
def loadData():
    trainDataFile = '/user/mydata/tianchi/repeat_buyers_format2/train_format2.csv'
    #trainDataFile = '/user/mydata/tianchi/repeat_buyers_format2/train_format2_sub.csv'
    df = sqlContext.read.csv(trainDataFile, header='true',inferSchema='true',sep=',').repartition(100)
    df = df.withColumn('random', rand())
    #df = df.filter("random<=0.3")

    countIDFreq(df)#统计商品id,类型id,品牌id出现的频次

if __name__ == '__main__':
#     loadData()
    
    
    
    
    