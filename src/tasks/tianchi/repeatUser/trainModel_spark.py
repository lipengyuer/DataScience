# coding=utf-8
#使用spark来计算。
from  pyspark import SparkContext, SQLContext
from pyspark import SparkConf
from pyspark.ml.feature import Word2Vec,CountVectorizer  
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors, VectorUDT
import pyspark.ml.evaluation as ev
from pyspark.sql.functions import rand, udf
import numpy as np

conf = SparkConf().setAppName("tianchi_repeat_buyers")  
sc = SparkContext(conf=conf)   
sqlContext=SQLContext(sc)  

def processLog(logStr, mode='work'):
    freq_brand_id_index_map = {'3738': 0, '1446': 1, '1360': 2, '5376': 3, '1214': 4, '82': 5, '2276': 6, '8235': 7, '4705': 8, '1662': 9, '3969': 10, '5434': 11, '2104': 12, '4073': 13, '7069': 14, '376': 15, '6143': 16, '4874': 17, '6065': 18, '6938': 19, '3650': 20, '7749': 21, '4290': 22, '3700': 23, '6585': 24, '6742': 25, '4509': 26, '1573': 27, '1859': 28, '1128': 29, '1866': 30, '3929': 31, '5818': 32, '3535': 33, '6762': 34, '5683': 35, '3345': 36, '2651': 37, '3489': 38, '7164': 39, '99': 40, '4953': 41, '4120': 42, '3997': 43, '1392': 44, '856': 45, '385': 46, '7892': 47, '1579': 48, '4801': 49, '7394': 50, '1552': 51, '1422': 52, '5800': 53, '4104': 54, '7995': 55, '1164': 56, '4382': 57, '4276': 58, '968': 59, '4360': 60, '3614': 61, '2045': 62, '4014': 63, '4058': 64, '4446': 65, '2031': 66, '6326': 67, '795': 68, '5795': 69, '2337': 70, '1954': 71, '247': 72, '4094': 73, '7279': 74, '6236': 75, '6320': 76, '2184': 77, '3273': 78, '2197': 79, '2459': 80, '8351': 81, '4365': 82, '6215': 83, '7819': 84, '4594': 85, '2116': 86, '6109': 87, '8411': 88, '5644': 89, '5735': 90, '1605': 91, '1981': 92, '3791': 93, '8353': 94, '7703': 95, '4623': 96, '6443': 97, '1661': 98, '6455': 99, '7169': 100}
    freq_category_id_index_map = {'662': 0, '737': 1, '1505': 2, '389': 3, '656': 4, '1349': 5, '1142': 6, '602': 7, '1577': 8, '1095': 9, '1438': 10, '177': 11, '407': 12, '821': 13, '1553': 14, '1467': 15, '302': 16, '1075': 17, '1208': 18, '1238': 19, '1188': 20, '664': 21, '1271': 22, '1213': 23, '1389': 24, '1397': 25, '1023': 26, '420': 27, '351': 28, '267': 29, '1611': 30, '464': 31, '154': 32, '1401': 33, '614': 34, '946': 35, '387': 36, '276': 37, '748': 38, '898': 39, '1181': 40, '1326': 41, '1591': 42, '1252': 43, '35': 44, '1112': 45, '184': 46, '992': 47, '612': 48, '962': 49, '1028': 50, '993': 51, '1174': 52, '812': 53, '883': 54, '629': 55, '451': 56, '384': 57, '530': 58, '229': 59, '120': 60, '295': 61, '1130': 62, '555': 63, '500': 64, '119': 65, '1118': 66, '683': 67, '247': 68, '1604': 69, '707': 70, '300': 71, '1429': 72, '178': 73, '180': 74, '786': 75, '1228': 76, '1280': 77, '756': 78, '825': 79, '320': 80, '2': 81, '611': 82, '11': 83, '766': 84, '834': 85, '369': 86, '933': 87, '1528': 88, '1147': 89, '598': 90, '776': 91, '833': 92, '1329': 93, '1197': 94, '559': 95, '1661': 96, '43': 97, '1344': 98, '1620': 99, '601': 100}
    brand_id_freq_vec = np.zeros(len(freq_brand_id_index_map))
    category_id_freq_vec = np.zeros(len(freq_category_id_index_map))
    if logStr:
        actions = logStr.split('#')
    else:
        actions = ''
    actionsNum = len(actions)#行为个数
    freqMap = {'item_id_num':set() , 'category_id_num': set() , 'brand_id_num':set()  ,
                'time_stamp_num':set() , 'action_type_total_num':set() }
    actionTypeFreq = {'0_freq': 0, '1_freq': 0, '2_freq': 0, '3_freq': 0}
    nearBuyNum = 0#用户有几次购买或者接近购买
    time_stamp_set = set()
    for action in actions:
        s = action.split(':')
        if len(s)<5:
            continue
        [item_id, category_id, brand_id, time_stamp, action_type] = s
        if action_type in [1, 2, 3] and time_stamp not in time_stamp_set:
            nearBuyNum += 1
            time_stamp_set.add(time_stamp)
            
        freqMap['item_id_num'].add(item_id)
        freqMap['category_id_num'].add(category_id)
        freqMap['brand_id_num'].add(brand_id)
        freqMap['action_type_total_num'].add(action_type)
        freqMap['time_stamp_num'].add(time_stamp)
        actionTypeFreq[action_type + '_freq'] = actionTypeFreq.get(action_type+ '_freq', 0) + 1
        
        if brand_id in freq_brand_id_index_map:
            brand_id_freq_vec[freq_brand_id_index_map[brand_id]] = 1
        if category_id in freq_category_id_index_map:
            category_id_freq_vec[freq_category_id_index_map[category_id]] = 1
        
    features = [actionsNum]
    names = ['actionsNum']
    for key in freqMap: 
        features.append(len(freqMap[key])/(actionsNum + 0.00000001))
        names.append(key)
    for key in actionTypeFreq: 
        features.append(actionTypeFreq[key]/(actionsNum + 0.00000001))
        names.append(key)
    features.append(nearBuyNum)
    names.append('nearBuyNum')
    print("features are ",names)
    features += list(brand_id_freq_vec) + list(category_id_freq_vec)
    if mode=='work':
        return features
    else:
        return features
    
def processLog4UserPersonas(logLines):
    featureValues, featureNames = [], []
    logs = logLines
    data = []
    for log in logs:
        if type(log)==str: data += log.split('#')
    data = list(map(lambda x: x.split(':'), data))
    
    #是否购买#####################################
    actionNum = len(data)
    timestamp_set =set()
    buyingNum, cartNum, click_num, favor_num = 0, 0, 0, 0
    for line in data: 
        if line[4] =='2': buyingNum += 1
        if line[4]=='1': cartNum+=1
        if line[4]=='0': click_num+=1
        if line[4]=='3': favor_num+=1
        timestamp_set.add(line[-2])
    day_num = len(timestamp_set)
    buy_favor_rate = buyingNum/(favor_num + 0.0000001)
    action_num_per_day = actionNum/(day_num + 0.0000001)
    click_num_per_day = click_num/(day_num + 0.0000001)
    buy_num_per_day = buyingNum/(day_num + 0.0000001)
    buying_prob = buyingNum/(click_num + 0.0000001)
    buying_prob_after_cart = buyingNum/(cartNum+0.0000001)
    buying_prob_after_click = buyingNum/(click_num+0.0000001)
    featureValues += [click_num_per_day, favor_num, cartNum, buyingNum, buying_prob, buying_prob_after_cart, 
                      buying_prob_after_click, action_num_per_day, buy_num_per_day,
            buy_favor_rate]
    featureNames += ['click_num_per_day', 'favor_num','cartNum','buyingNum', 'buying_prob', 'buying_prob_after_cart', 
                     'buying_prob_after_click', 'action_num_per_day'
                , 'buy_num_per_day', 'buy_favor_rate']
    
    
    
    #######商品类型####################################
    cat_num, brand_num,  cat_set, brand_set = 0, 0, set(), set()
    
    for line in data:
        if line[4] in ['2', '3']:
            cat_set.add(line[1])
            brand_set.add(line[2])
    cat_num, brand_num = len(cat_set), len(brand_set)
    featureValues += [cat_num, brand_num]
    featureNames += ['cat_num', 'brand_num']
    ##################
    return featureValues, featureNames

def addFeaturesInMapItems(dataItems, features, featureNames, user_num_came_merchant):
    if type(dataItems)==dict: dataItems = dataItems.items()
    for name, value in dataItems:
        features.append(value)
        featureNames.append(name)
        features.append(value/(user_num_came_merchant + 0.0000001))
        featureNames.append(name + '_user_buy_rate')
    return features, featureNames
    
def processLog4MerchantPersonas(logLines):
    featureValues, featureNames = [], []
    user_id_set, buyer_id_set, click_id_set =set({}), set({}), set({})
    logs = logLines
    data = []
    for log in logs:
        if type(log)==str: data += log.split('#')
    data = list(map(lambda x: x.split(':'), data))
    
    #是否购买#####################################
    actionNum = len(data)
    timestamp_set =set()
    buyingNum, cartNum, click_num, favor_num = 0, 0, 0, 0
    catSaleMap, brandSaleMap = {}, {}
    for line in data: 
        if line[4] =='2': 
            buyingNum += 1
            click_id_set.add(line[0])
        if line[4]=='1': cartNum+=1
        if line[4]=='0': click_num+=1
        if line[4]=='3': favor_num+=1
        timestamp_set.add(line[-2])
        user_id_set.add(line[0])
        if line[4] in ['2', '3']: buyer_id_set.add(line[0])
        catName = 'catagory_' + line[1]
        brandName = 'brand_' + line[2]
        catSaleMap[catName] = catSaleMap.get(catName, 0) + 1
        brandSaleMap[brandName] = brandSaleMap.get(brandName, 0) + 1

    day_num = len(timestamp_set)
    buy_favor_rate = buyingNum/(favor_num + 0.0000001)
    action_num_per_day = actionNum/(day_num + 0.0000001)
    click_num_per_day = click_num/(day_num + 0.0000001)
    buy_num_per_day = buyingNum/(day_num + 0.0000001)
    buying_prob = buyingNum/(click_num + 0.0000001)
    buying_prob_after_cart = buyingNum/(cartNum+0.0000001)
    buying_prob_after_click = buyingNum/(click_num+0.0000001)
    
    #来过这家商铺的用户行为数据
    user_num_came_merchant = len(user_id_set)
    user_buy_merchant = len(buyer_id_set)
    user_click_merchant = len(click_id_set)
    user_to_buyer_rate = user_buy_merchant/(user_num_came_merchant+0.0000001)
    user_click_to_buy_rate = user_buy_merchant/(user_click_merchant+0.0000001)

    featureValues += [click_num_per_day, favor_num, cartNum, buyingNum, buying_prob, buying_prob_after_cart, 
                      buying_prob_after_click, action_num_per_day, buy_num_per_day,
            buy_favor_rate, user_num_came_merchant, user_buy_merchant, user_to_buyer_rate,user_click_to_buy_rate
            ]
    featureNames += ['click_num_per_day', 'favor_num','cartNum','buyingNum', 'buying_prob', 'buying_prob_after_cart', 
                     'buying_prob_after_click', 'action_num_per_day'
                , 'buy_num_per_day', 'buy_favor_rate', 'user_num_came_merchant', 'user_buy_merchant',
                'user_to_buyer_rate', 'user_click_to_buy_rate']
    
    
    
    #######商品类型####################################
    cat_num, brand_num,  cat_set, brand_set = 0, 0, set(), set()
    
    for line in data:
        if line[4] in ['2', '3']:
            cat_set.add(line[1])
            brand_set.add(line[2])
    cat_num, brand_num = len(cat_set), len(brand_set)
    featureValues += [cat_num, brand_num]
    featureNames += ['cat_num', 'brand_num']
    ##################
    
#     catSaleMap[catName] = catSaleMap.get(catName, 0) + 1
#     brandSaleMap[brandName] = brandSaleMap.get(brandName, 0) + 1
    def paddingList(oriList, L):
        length = len(oriList)
        if L>length:
            oriList = oriList + [['padding_name', 0] for _ in range(L-length)]
        return oriList
    
    catSaleItems = sorted(catSaleMap.items(), key=lambda x: x[1], reverse=True)[:10]
    brandSaItems = sorted(brandSaleMap.items(), key=lambda x: x[1], reverse=True)[:10]
    catSaleItems = paddingList(catSaleItems, 10)
    brandSaItems = paddingList(brandSaItems, 10)
    featureValues, featureNames = addFeaturesInMapItems(catSaleItems,featureValues, featureNames, user_num_came_merchant)
    featureValues, featureNames = addFeaturesInMapItems(brandSaItems,featureValues, featureNames, user_num_came_merchant)

           
    return featureValues, featureNames

#对用户进行画像，并把结果写到csv文件中，便于后续进行联表操作
def personas4User(personas_rdd):
    def stastic(data_list):
        merchant_num = len(data_list)
        logLines = list(map(lambda x: x[-1], data_list))
        featuresFromLog, logFeatureNames = processLog4UserPersonas(logLines)
        res = [merchant_num/10] + featuresFromLog
        return res
    print("开始分组。")
    res_rdd = personas_rdd.groupByKey()
    print("开始统计")
    res_rdd = res_rdd.mapValues(stastic)
    return res_rdd

#对商家进行画像，并把结果写到csv文件中，便于后续进行联表操作
def personas4Merchant(personas_merchant_rdd):
    def stastic(data_list):
        merchant_num = len(data_list)
        logLines = list(map(lambda x: x[-1], data_list))
        featuresFromLog, logFeatureNames = processLog4MerchantPersonas(logLines)
        res = [merchant_num/10] + featuresFromLog
        return res
    print("开始分组。")
    res_rdd = personas_merchant_rdd.groupByKey()
    print("开始统计")
    res_rdd = res_rdd.mapValues(stastic)
    return res_rdd

#特征工程
def featureEngineering(data_rdd, train_df):
    data_rdd_feature_from_log = data_rdd.mapValues(lambda x: processLog(x[-1]))
    log('number of data_rdd_feature_from_log is ' + str(data_rdd_feature_from_log.count()) + '\n')
    data_train = data_rdd.join(data_rdd_feature_from_log)#add log features to traning data
    #log('number of data_train is ' + str(data_train.count()) + '\n')
    #let user_id be the key for adding user personas data
    data_train_ = data_train.map(lambda x: (x[0][0], [x[0][0], x[0][1]] + getFeaturesAndLabel(x[1])))
    
    #读取用户画像
    log('processing users personas.\n')
    personas_user_rdd= train_df.rdd.map(lambda x: (x[0], x[1:]))
    personas_user_rdd = personas4User(personas_user_rdd)
    data_train_ = data_train_.join(personas_user_rdd)#add user personas features to traning data
    #log('data_train_ is ' + str(data_train_.first()) + '\n#########################\n')
    data_train_ = data_train_.mapValues(lambda x:x[0][:3] + [x[0][3] + list(x[1])])
    
    #'''
    #读取商家用户画像
    log('processing merchants personas.\n')
    #let merchant_id be the key for adding merchant personas data
    data_train_ = data_train_.map(lambda x: x[1]).map(lambda x: (x[1], x))
    personas_merchant_rdd= train_df.rdd.map(lambda x: (x[3], [x[1], x[2], x[3], -1, x[5]]))
    #log("画像原始数据 " + str(personas_merchant_rdd.take(10)) + "\n###################\n")
    personas_merchant_rdd = personas4Merchant(personas_merchant_rdd)
    #log("画像数据 " + str(personas_merchant_rdd.take(10)) + "\n###################\n")
    data_train_ = data_train_.join(personas_merchant_rdd)
    #log("加上商家画像的数据 " + str(data_train_.take(10)) + "\n###################\n")
    data_train_ = data_train_.mapValues(lambda x:x[0][:3] + [Vectors.dense(x[0][3] + list(x[1]))])
    #data_train_ = data_train_.mapValues(lambda x:x[0][:3] + [Vectors.dense(x[0][3])])
    #'''
    data_train_ = data_train_.map(lambda x: x[1])
    columns = ['user_id', 'merchant_id', 'label', 'features']
    df_data = sqlContext.createDataFrame(data_train_, schema=columns)
    
    return df_data

def getFeaturesAndLabel(data):
    res = [data[0][3], data[1]]
    return res

from sklearn import metrics
def calAccuracy(labels, predictions):
    auc = metrics.roc_auc_score(labels,predictions)#验证集上的auc值
    print('AUC is', auc)
    return auc
import random
def getRandom():
    random.uniform(0, 1)

def log(line):
    with open('log.txt', 'a+') as f:
        f.write(line)

def process01(value):
    if value>1.0: return 1.0
    elif value<0.0: return 0.0
    else: return float(value)

def dataProcess(df, mode='work'):
    df = df.fillna(-666)
    data_rdd = df.rdd.map(lambda x: [x[i] for i in range(6)])
    if mode!='work':
        data_useful_user_rdd = data_rdd.map(lambda x: ((x[0], x[3]), x[1:]))
        data_rdd_1 = data_useful_user_rdd.filter(lambda x: -666 not in x[1][:-1] and x[1][3]>=0).\
                     mapValues(lambda x: x[:-1] + [''] if x[-1]==-666 else x).repartition(1000)
    else:                
        data_useful_user_rdd = data_rdd.map(lambda x: ((x[0], x[3]), x[1:]))
        data_rdd_1 = data_useful_user_rdd.mapValues(lambda x: x[:-1] + [''] if x[-1]==-666 else x).repartition(1000)
    
    log("extracting training features.\n")
    trainData = featureEngineering(data_rdd_1, df)
    trainData = trainData
    return trainData
    
from pyspark.ml.regression import *
import time
#加载用于做交叉验证的数据
def loadData4Validation(K=5):
    trainDataFile = '/user/mydata/tianchi/repeat_buyers_format2/train_format2.csv'
    #trainDataFile = '/user/mydata/tianchi/repeat_buyers_format2/train_format2_sub.csv'

    log("start to validate.\n")
    aucTotal = 0
    for i in range(5):
        t1 = time.time()
        #读取数据，分割为训练集和测试集
        df = sqlContext.read.csv(trainDataFile, header='true',inferSchema='true',sep=',').repartition(1000)
        df = df.withColumn('random', rand())
        train_df, test_df = df.filter("random<=0.8").repartition(1000), df.filter("random>0.2").repartition(1000)
        
        #训练阶段
        trainData = dataProcess(train_df, mode='validation')
#         trainData = trainData.withColumn('random', rand())
#         trainData = trainData.where("(label==0 and random>0.5) or label=1")
        log("data preprocessing costs " + str(time.time() - t1) + ".\n")
        #clf = LogisticRegression(featuresCol='features', labelCol='label', predictionCol='prediction')
        log('training it\n')
        #clf = GBTRegressor(maxIter=200, maxDepth=6, seed=42,subsamplingRate=0.7)
        clf = RandomForestRegressor(subsamplingRate=0.7, numTrees=50, featureSubsetStrategy='0.5')
        model = clf.fit(trainData)
        log('training cost ' + str(time.time()-t1) + 's\n')
        
        #测试阶段
        testData = dataProcess(test_df, mode='validation')
        log('transforming them\n')
        train_prediction = model.transform(trainData)
        test_prediction = model.transform(testData)
        udfProcessFloat201 = udf(process01, DoubleType())
        #log(str(train_prediction.rdd.take(2)) + '\n')
        train_prediction = train_prediction.withColumn('prediction_final', udfProcessFloat201(train_prediction.prediction))
        
        test_prediction = test_prediction.withColumn('prediction_final',  udfProcessFloat201(test_prediction.prediction))
    
        #log(str(train_prediction.rdd.take(20)) + '\n')
        #"""
        #use spark to evaluate model
        print("#####evaluating#######\n\n\n\n\n\n\n\n\n\n\n")
        evaluator = ev.BinaryClassificationEvaluator(
                rawPredictionCol='prediction_final',
                labelCol='label')
        print("############\n\n\n\n\n\n\n\n\n\n\n")
        train_auc = evaluator.evaluate(train_prediction,{evaluator.metricName: 'areaUnderROC'})
        test_auc = evaluator.evaluate(test_prediction,{evaluator.metricName: 'areaUnderROC'})
        log(str(i) + " epoch auc is " + str(test_auc) + ', training auc is ' +str(train_auc) +  '\n')
        t2 = time.time()
        print("############time cost is " + str(t2-t1) + "\n\n\n\n\n\n\n\n\n\n\n")
        #use spark to evaluate model
        #"""
        
        '''
        #use sklearn to evaluate model
        print("############\n\n\n\n\n\n\n\n\n\n\n")
        from sklearn.metrics import auc
        from sklearn.metrics import roc_auc_score
        print("############\n\n\n\n\n\n\n\n\n\n\n")
        train_prediction, test_prediction = \
                train_prediction.select('label', 'prediction_final').toPandas(), \
                test_prediction.select('label', 'prediction_final').toPandas()
        trainLables, trainPredictions = \
                 train_prediction['label'].values, train_prediction['prediction_final'].values
        testLables, testPredictions = \
                 test_prediction['label'].values, test_prediction['prediction_final'].values
        train_auc, test_auc = roc_auc_score(trainLables, trainPredictions), roc_auc_score(testLables, testPredictions)
        log(str(i) + " epoch auc is " + str(test_auc) + ', training auc is ' +str(train_auc) +  '\n')
        print("############\n\n\n\n\n\n\n\n\n\n\n")
        #use sklearn to evaluate model
        '''
        aucTotal += test_auc
    print("av auc is ", aucTotal/5)
    log(str(aucTotal/5) + '\n')
    print("############\n\n\n\n\n\n\n\n\n\n\n")
    

def getIDs():
    fileName = r"/user/mydata/tianchi/repeat_buyers_format2/sample_submission.csv"
    df = sqlContext.read.csv(fileName, header='true',inferSchema='true',sep=',')
    df = df.select(df['user_id'], df['merchant_id'])
    return df

def loadData2Submit():
    ID_df = getIDs()#加载id数据
    ID_df = ID_df.withColumnRenamed('user_id', 'user_id_set')
    ID_df = ID_df.withColumnRenamed('merchant_id', 'merchant_id_set')
    log('number of id pair is ' + str(ID_df.count()) + '\n')
    trainDataFile = '/user/mydata/tianchi/repeat_buyers_format2/test_format2.csv'
    df = sqlContext.read.csv(trainDataFile, header='true',inferSchema='true',sep=',').repartition(1000)
    df = df.withColumn('random', rand())
    log('data final' + str(df.take(2)) + '\n')
    df1 = dataProcess(df, mode='work')
#     log('ori data is ' + str(df1.take(200)) + '/n#####################################\n')
    data4Valaidation = ID_df.join(df1, [ID_df.user_id_set==df1.user_id, ID_df.merchant_id_set==df1.merchant_id], 'left').repartition(1000)
    log('data size after joining is ' + str(data4Valaidation.take(2)) + '\n')
    return data4Valaidation

def getFinalResult():
    log("start to process final task.\n")
    trainDataFile = '/user/mydata/tianchi/repeat_buyers_format2/train_format2.csv'
    #trainDataFile = '/user/mydata/tianchi/repeat_buyers_format2/train_format2_sub.csv'
    df = sqlContext.read.csv(trainDataFile, header='true',inferSchema='true',sep=',').repartition(1000)    
    data4training = dataProcess(df, mode='validation')
    data4training = data4training.repartition(2000) 
    #data4training = data4training.filter(data4training.label==0 & data4training.random>0.5)
    #data4training = data4training.where("(label==0 and random>0.5) or label=1")
    clf = RandomForestRegressor(subsamplingRate=0.7, numTrees=50, featureSubsetStrategy='0.5')
    #clf = GBTRegressor(maxIter=10, maxDepth=6, seed=42,subsamplingRate=0.7)
    log("traning.\n")
    model = clf.fit(data4training)
    log("finish traning.\n")
    log("loading data.\n")
    df2Submit = loadData2Submit()
    log("transforming data.\n")
    res = model.transform(df2Submit)
    udfProcessFloat201 = udf(process01, DoubleType())
    res = res.withColumn('prob', udfProcessFloat201(res.prediction))

    res = res.selectExpr('user_id', 'merchant_id','prob')#'prediction as prob')
    log("storing data.\n")
    res = res.toPandas()
    res.to_csv('myRes_spark.csv', index=0)

if __name__ == '__main__':
    #data4training = loadData4Validation()
    getFinalResult()

"""
pyspark2 --master yarn  --driver-memory 2G --executor-memory 2G --num-executors 10 --conf spark.pyspark.python=/opt/anaconda2/envs/python36/bin/python --conf spark.executorEnv.PYTHONHASHSEED=321
spark2-submit --master yarn  --driver-memory 3G --executor-memory 1500M  --conf spark.pyspark.python=/opt/anaconda2/envs/python36/bin/python --conf spark.executorEnv.PYTHONHASHSEED=321 trainModel_spark.py
spark2-submit --master yarn --num-executors 20 --executor-memory 5G --conf spark.pyspark.python=/opt/anaconda2/envs/python36/bin/python --conf spark.executorEnv.PYTHONHASHSEED=321 trainModel_spark.py
spark2-submit --master yarn --num-executors 20 --executor-memory 4G --conf spark.pyspark.python=/opt/anaconda2/envs/python36/bin/python --conf spark.executorEnv.PYTHONHASHSEED=321 trainModel_spark.py

"""


    
    