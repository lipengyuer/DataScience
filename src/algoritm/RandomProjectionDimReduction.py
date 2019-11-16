'''
Created on 2019年11月16日

@author: Administrator
'''
#基于随机投影的数据降维

import numpy as np
from nltk.probability import RandomProbDist

class DimentionalityReductionByRP():
    
    def __init__(self, ori_dim, new_dim=10):
        self.ori_dim = ori_dim
        self.new_dim = new_dim
        #生成参数，即随机超平面。一般需要保存起来。
        self.random_vectors = self.get_random_vectors()
    
    #为了每次加载后，得到同一份随机向量，这里使用设定随机种子的方式
    def get_random_vectors(self):
        random_vectors = np.zeros((self.ori_dim, self.new_dim))#行对应词向量空间，列对应新空间
        for i in range(self.ori_dim):
            np.random.seed(i)#一个随种子，对应一个唯一的随机数序列(生成器的算法实际上是伪随机，当然效果挺不错)。
#             for j in range(self.new_dim):#为各个随机向量的这个维度生成随机数。
#                 random_vectors[i, j] = np.random.normal(0,1)
            random_vectors[i, :] = np.random.normal(0,1, (1, self.new_dim))#更快一点
        return random_vectors
    
    #映射一条数据到新的空间里
    def f(self, ori_feature):
        ori_feature = np.array(ori_feature)
        new_point = np.dot(ori_feature.reshape((1, self.ori_dim)),self.random_vectors)
        return new_point  
if __name__ == '__main__':
    a_point = [1,3, 1, 1.5, 6,6,6,10]
    model = DimentionalityReductionByRP(len(a_point), new_dim=3)
    print(model.f(a_point)) 
        
        
        