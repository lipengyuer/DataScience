#聚类分析虎扑用户，特征是用户的个人资料
import sys, os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
from dbconnection.getMongo import getConnectionMongo
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from  matplotlib import pyplot as plt
import copy

class clusterAnnalysisThem():
    def __init__(self):
        self.data = None
        self.features = None
        self.featuresPro = None
        self.bandwidth = None
        self.cluster = None
        self.IDs = []
        self.IDLabelMap = {}
        self.IDDataMap = {}
    def getData(self):
        print('正在获取数据。')
        conn = getConnectionMongo()
        cur = conn.find({}, {'_id': 1, 'follow': 1, 'fans': 1, 'onlineTime': 1, 'homeTeams': 1, \
                             'communityRPScore': 1, 'HPLevel': 1})
        data = []
        count = 0
        for line in cur:
            count += 1
            if count>20000:
                break
            if len(line.keys())<=2:
                continue

            line['HPLevel'] = line['HPLevel'] if 'HPLevel' in line else 0
            if line['HPLevel']==0:
                continue
            self.IDs.append(line['_id'])
            # 处理虎扑等级

            # 处理粉丝数
            line['fans'] = len(line['fans']) if 'fans' in line else 0
            # 处理关注数
            line['follow'] = len(line['follow']) if 'follow' in line else 0
            # 处理主队数
            line['homeTeams'] = len(line['homeTeams']) if 'homeTeams' in line else 0
            # 处理在线时间
            line['onlineTime'] = int(line['onlineTime']) if 'onlineTime' in line else 0
            #处理声望值
            line['communityRPScore'] = int(line['communityRPScore']) if 'communityRPScore' in line else 0
            line['communityRPScore'] = 0.1 if line['communityRPScore'] <= 0 else line['communityRPScore']
            # line['fans'] = len(line['fans'])
            data.append([line['_id'], line['communityRPScore'], line['HPLevel'], line['homeTeams']
                            , line['fans'], line['follow'], line['onlineTime']])
            self.IDDataMap[line['_id']] = [line['communityRPScore'], line['HPLevel'], line['homeTeams']
                            , line['fans'], line['follow'], line['onlineTime']]
        self.data = pd.DataFrame(data, columns=['_id', 'communityRPScore', 'HPLevel',
                                         'homeTeams', 'fans', 'follow', 'onlineTime'])

    def getFeatures(self):
        print('正在获取特征。')
        self.features = self.data[['communityRPScore', 'HPLevel',
                       'homeTeams', 'fans', 'follow', 'onlineTime']]

    def normalIt(self, f):
        m = np.mean(f)
        s = np.mean(np.abs(f-m))
        return (f - m)/s

    def featureEngineering(self):
        self.featuresPro = copy.deepcopy(self.features)
        self.featuresPro['communityRPScore'] = self.normalIt(self.featuresPro['communityRPScore'])
        self.featuresPro['HPLevel'] = self.normalIt(self.featuresPro['HPLevel'] + 0.1)
        self.featuresPro['fans'] = self.normalIt(self.featuresPro['fans'] + 0.1)
        self.featuresPro['follow'] = self.normalIt(self.featuresPro['follow'] + 0.1)
        self.featuresPro['onlineTime'] = self.normalIt(self.featuresPro['onlineTime'] + 0.1)

    def estimateBandwidth(self):
        print('正在估计簇半径')
        self.bandwidth = estimate_bandwidth(self.featuresPro, quantile=0.7, n_samples=500)

    # meanshift聚类
    def clusterThem(self):
        print('正在聚类。')
        cluster = MeanShift(bandwidth=self.bandwidth)
        cluster.fit(self.featuresPro)
        self.pred = cluster.predict(self.featuresPro)

    def addLabel2IDs(self):
        print('给每个样本分配标签。',self.pred)
        for i in range(len(self.pred)):
            label = self.pred[i]
            if label in self.IDLabelMap:
                self.IDLabelMap[label].append(self.IDs[i])
            else:
                self.IDLabelMap[label] = [self.IDs[i]]

    def showHist(self, aCol):
        plt.hist(aCol)
        plt.show()

    def stasticsOnClusters(self, minMenberNum):
        self.usefulClusters = {}
        #提取足够大的簇
        for key in a.IDLabelMap:
            if len(a.IDLabelMap[key]) > minMenberNum:
                print(key, len(a.IDLabelMap[key]), a.IDLabelMap[key])
                self.usefulClusters[key] = a.IDLabelMap[key]

        #获取每个uid的数据
        self.usefulClustersData = {}
        for clusterId in self.usefulClusters:
            uids = self.usefulClusters[clusterId]
            dataTemp = []
            for uid in uids:
                dataTemp.append(self.IDDataMap[uid])
            dataTemp = np.array(dataTemp)
            meanValue = np.mean(dataTemp, axis=0)
            medianValue = np.median(dataTemp, axis=0)
            stdVallue = np.std(dataTemp, axis=0)
            self.usefulClustersData[clusterId] = {'popularity': len(dataTemp),
                                                  'meanValue': meanValue,
                                                  'medianValue': medianValue,
                                                  'stdVallue': stdVallue}

    def showRadar(self, datas, featureLabels, sampleLabels):
        labels = featureLabels
        # 数据个数
        dataLenth = len(datas[0])
        # 数据
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)  # polar参数！！
        for data in datas:
            # ========自己设置结束============
            angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)
            data = np.concatenate((data, [data[0]]))  # 闭合
            angles = np.concatenate((angles, [angles[0]]))  # 闭合
            ax.plot(angles, data, 'o-', linewidth=2)  # 画线
        # ax.fill(angles, data, facecolor='r', alpha=0.25)# 填充
        ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
        ax.set_title("虎扑主要用户类型对比", va='bottom', fontproperties="SimHei")
        # ax.set_rlim(0, 10)
        ax.grid(True)
        plt.legend(sampleLabels)
        plt.show()

if __name__ == '__main__':
    a = clusterAnnalysisThem()
    a.getData()
    a.getFeatures()
    a.featureEngineering()
    # a.showHist(a.featuresPro['communityRPScore'])
    a.estimateBandwidth()
    a.clusterThem()
    a.addLabel2IDs()
    import pickle
    pickle.dump(a, open('a.pkl', 'wb'))
    a = pickle.load(open('a.pkl','rb'))
    a.stasticsOnClusters(10)
    datas = []
    clusterIDs = []
    features = a.data.columns.values.tolist()
    print(features)
    for key in a.usefulClustersData:
        clusterIDs.append(key)
        datas.append(np.log(0.1 + a.usefulClustersData[key]['meanValue']))
        print(key,
              # '数量', a.usefulClustersData[key]['popularity'],
              # '均值', a.usefulClustersData[key]['meanValue'],
              '中位数', a.usefulClustersData[key]['medianValue'],
              # '标准差', a.usefulClustersData[key]['stdVallue']
              )

    a.showRadar(datas, features, clusterIDs)


