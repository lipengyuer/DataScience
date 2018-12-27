'''
Created on 2018年12月27日

@author: pyli
'''
#实现一个用于求篱笆网络里最短路径的维特比算法
import itertools


stationsInEachCity = [['a1', 'a2'], ['b1', 'b2'], ['c1', 'c2', 'c3']]#三个城市的车站名

def init():
    network = """a1 b1 3
a1 b2 4
a2 b1 1
a2 b2 1.4
b1 c1 1.4
b1 c2 0.3
b1 c3 2
b2 c1 1.2
b2 c2 2.3
b2 c3 1.1"""
    #这是城市A-B-C的车站之间的距离关系，求一个最短的车站路线
    network = network.split('\n')
    network = list(map(lambda x: x.split(' '), network))
    network = list(map(lambda x: [x[0], x[1], float(x[2])], network))
    distanceMap = {}
    for line in network:
        distanceMap[tuple(line[:2])] = line[2]
    return distanceMap
stationDistMap = init()

#暴力求最短路径: 计算出所有路径的距离，然后挑选最短的那一个
def calDistanceAllPath(stationsInEachCity):
    #罗列出所有可能的路径
    allPaths = stationsInEachCity[0]
    for i in range(1, len(stationsInEachCity)):
        allPaths = itertools.product(allPaths, stationsInEachCity[i])#
        allPaths = list(map(lambda x: list(x[0])+ [x[1]] if type(x[0])==tuple else x, allPaths))
    
    pathDistMap = {}
    for aPath in allPaths:
        dist = 0
        for i in range(1, len(aPath)):
            nodePair = (aPath[i-1], aPath[i])#这是两个车站的名字组成的tuple
            dist += stationDistMap[nodePair]#取出这两个车站之间的距离，累加起来，就是这条路径的长度
        pathDistMap[tuple(aPath)] = dist
    pathDistList = sorted(pathDistMap.items(), key=lambda x: x[1])#按照路径从短到长排列各个路径
    print(pathDistList)
    return pathDistList[0][0]#返回长度最短的路径
    
#基于维特比算法，求出最短路径
def getShortestPathByViterbi(stationsInEachCity):
    """输入篱笆网络数据，包括每一个城市的车站列表，以及两个车站之间的距离。返回从第一个城市大最后一个城市的最短路径"""
    candShortestPathList = [[] for _ in range(len(stationsInEachCity[0]))]#第一个城市的车站数量，就是候选最短路径的个数
    #我们会以这几个车站为出发点，分别找到一条最短路径，然后选取其中最短的那一条，作为真正的最短路径。
    candPathAndDistanceMap = {}#存储两个节点之间的路径长度
    candPathAndDistanceList = []#用于存储各个初始车站出发的最短路径
    initStations = stationsInEachCity[0]#取出初始车站
    for i in range(len(initStations)):#求每一个初始车站对应的最短路径的长度
        candShortestPathList = [initStations[i]]
        tempStationDistMap = {}#暂时存储已经找到的最短路径的末尾节点，与下一个城市的所有节点的距离
        for j in range(1, len(stationsInEachCity)):#向后遍历每一个城市
            for n in range(len(stationsInEachCity[j])):#遍历这个城市的所有车站
                stationPair = (candShortestPathList[-1], stationsInEachCity[j][n])#已经找到的最短路径的最后一个节点，与当前节点组成一个肯可能的路径
                currentShortestPath = tuple(candShortestPathList)
                tempPath = tuple(candShortestPathList + [stationsInEachCity[j][n]])#已经找到的最短路径，加上当前车站
                #已有最短路径的长度，加上末尾节车站到当前车站的距离，就是一个候选路径的长度
#                 print(tempPath,currentShortestPath, stationPair)
                tempStationDistMap[tempPath] = tempStationDistMap.get(currentShortestPath, 0) + stationDistMap[stationPair]
            #把路径上节点个数小于j,也就是上一次得到的所有路径全都删掉
            tempStationDistList = list(filter(lambda x: len(x[0])> j, tempStationDistMap.items()))  
            tempStationDistList = list(sorted(tempStationDistList, key=lambda x: x[1]))
#             print(tempStationDistList)
            candShortestPathList = list(tempStationDistList[0][0])#把起点到当前城市的最短路径的
            candPathAndDistanceMap[tuple(candShortestPathList)] = tempStationDistList[0][1]

        aCandShortestPath = candShortestPathList
        candPathAndDistanceList.append([aCandShortestPath, candPathAndDistanceMap[tuple(aCandShortestPath)]])#把这个初始车站为起点的最短路径缓存起来
    candPathAndDistanceList = sorted(candPathAndDistanceList, key=lambda x: x[1])
    result = candPathAndDistanceList[0][0]#最短路径
    print(result, candPathAndDistanceList)
    
if __name__ == '__main__':
    calDistanceAllPath(stationsInEachCity)
    getShortestPathByViterbi(stationsInEachCity)

    
    
    
    
    
    
    
    
