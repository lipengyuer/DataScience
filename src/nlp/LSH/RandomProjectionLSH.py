'''
Created on 2019年11月16日

@author: Administrator
'''
#基于随机投影的hash

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
from pyhanlp import HanLP
import hashlib
import numpy as np
from LSH import LSH
class RPLSH(LSH):
    
    def __init__(self, K=10):
        self.word_id_map = self.load_vocab()
        self.vocab_size = len(self.word_id_map)
        self.new_dim = K
        #生成参数，即随机超平面。一般需要保存起来。
        self.random_vectors = self.get_random_vectors()
    
    #为了每次加载后，得到同一份随机向量，这里使用设定随机种子的方式
    def get_random_vectors(self):
        random_vectors = np.zeros((self.vocab_size, self.new_dim))#行对应词向量空间，列对应新空间
        for i in range(self.vocab_size):
            np.random.seed(i)#一个随种子，对应一个唯一的随机数序列(生成器的算法实际上是伪随机，当然效果挺不错)。
            for j in range(self.new_dim):#为各个随机向量的这个维度生成随机数。
                random_vectors[i, j] = np.random.normal(0,1)
#             random_vectors[i, :] = np.random.normal(0,1, (1, self.new_dim))#更快一点
        return random_vectors
        
    #词典来自https://github.com/fxsjy/jieba/tree/master/jieba
    def load_vocab(self):
        with open("./dict.txt", 'r', encoding='utf8') as f:
            lines = f.readlines()
        word_freq_map = {}
        word_id_map = { }
        for word_info in lines:
            word, freq, postag = word_info.split(" ")
            word_freq_map[word] = int(freq)
        for word,freq in sorted(word_freq_map.items(), key=lambda x: x[1])[-10000:]:#基于词频做一个过滤，算是初步降维
            word_id_map[word] = len(word_id_map)
        return word_id_map
        
    def get_TF(self, text):
        words = self.segment(text)
        TF = np.zeros(self.vocab_size)
        for word in words:
            word_onehot_code = np.zeros(self.vocab_size)
            if word in self.word_id_map:
                word_onehot_code[self.word_id_map[word]] = 1
            TF += word_onehot_code
        return TF
    
    #计算文档的hash编码
    def hash(self, text):
        TF = self.get_TF(text)
        new_point = np.dot(TF.reshape((1, self.vocab_size)),self.random_vectors)
        new_point = new_point[0]
        binary_code = np.zeros(self.new_dim)
        for i in range(new_point.shape[0]):
            if new_point[i]>=0: binary_code[i] = 1
            else: binary_code[i] = 0
        return binary_code


        
if __name__ == '__main__':
    corpus = ["我要上厕所", "我要上个厕所", "我要去上厕所了"]
    rp = RPLSH()
    for s in corpus:
        print(rp.hash(s) ) 
        
        
        