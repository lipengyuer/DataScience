#分析虎扑用户的社交网络
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
import networkx as nx
# from networkx import find_com
import community

class clusterAnnalysisThem():
    def __init__(self):
        self.data = None
        self.G = nx.Graph()

    def getData(self, dataNum = 100):
        print('正在获取数据。')
        conn = getConnectionMongo()
        cur = conn.find({}, {'_id': 1, 'follow': 1, 'fans': 1})
        count = 0
        for line in cur:
            count += 1
            if count == dataNum:
                break
            if len(line.keys())<=2:
                continue
            # print('正在加载用户', line['_id'],'的关注和粉丝，数量是', len(line['follow']))
            for fol in line['follow']:
                self.G.add_edge(line['_id'], fol)
            for fan in line['fans']:
                self.G.add_edge(fan, line['_id'])

def find_community(graph):
    return list(nx.find_cores(graph))

if __name__ == '__main__':
    a = clusterAnnalysisThem()
    a.getData()
    path = nx.all_pairs_shortest_path(a.G)
    print(a.G.number_of_nodes())
    print(a.G.number_of_edges())
    # nx.draw_spring(a.G,node_color = 'b', edge_color = 'r', with_labels = True,font_size = 2, node_size = 2)
    #
    # plt.show()
    print(find_community(a.G))