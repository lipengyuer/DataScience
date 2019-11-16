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
from RandomProjectionLSH import RPLSH

class RHLSH(RPLSH):
    
        #计算文档的hash编码
    def hash(self, text):
        TF = self.get_TF(text)
        new_point = np.dot(TF.reshape((1, self.vocab_size)),self.random_vectors)
        new_point = new_point[0]
        binary_code = np.zeros(self.new_dim)
        for i in range(new_point.shape[0]):
            if new_point[i]>=0: binary_code[i] = 1
            else: binary_code[i] = -1#这是随机超平面hsh和随机投影hash的唯一区别
        return binary_code
            
if __name__ == '__main__':
    corpus = ["我要上厕所", "我要上个厕所", "我要去上厕所了"]
    rp = RHLSH()
    for s in corpus:
        print(rp.hash(s) ) 