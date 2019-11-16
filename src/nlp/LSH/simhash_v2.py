'''
Created on 2019年11月4日
@author: Administrator
'''
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
from pyhanlp import HanLP
import hashlib
import numpy as np
from simhash_v1 import SimHashV1
#随机超平面hash视角下的simhash
class SimHashV2(SimHashV1):
    
    def __init__(self, K=64):
        self.K = K
    
    def hash(self, text):
        words = self.segment(text)
        word_code_sum = np.zeros(self.K)#用来计算各个维度的得分
        simhash_code_str = ""#用来存储各个维度的最终取值，是string
        for word in words:
            word_code_sum += self.word_not_random_hyperplane_hash(word)
        for i in range(self.K):
            if word_code_sum[i]>=0:
                simhash_code_str += "1"
            else:
                simhash_code_str += "0"
        simhash_code = int("0b" + simhash_code_str, 2)#将str形式的编码转为二进制编码
        return simhash_code
    
    #计算两篇文档的距离
    def get_distance(self, text1, text2):
        simhash_code1 = self.hash(text1)
        simhash_code2 = self.hash(text2)
        distance = self.get_hamming_distance_bit(simhash_code1, simhash_code2)#使用位运算版本的海明距离计算方法
        return distance
    
    #获取词语的hash码
    def word_not_random_hyperplane_hash(self, word):
        
        ###########词语的初步编码部分，随便一个hash函数即可################
        bin_code = ""
        md5_code = hashlib.md5(word.encode('utf8')).hexdigest()#获取词语的16禁进制hash码
        dec_code = int(md5_code, 16)#获取词语的十进制编码
        
        #获取词语的二进制编码
        for char in str(dec_code):
            binary_code = str(bin(int(char)))[2:]#编码开头的0b给去掉
            bin_code += binary_code
        ###########词语的初步编码部分########
        
        #用不随机的超平面分割空间，得到词语的最终编码
        code = np.ones(self.K)
        for i in range(self.K):
            if bin_code[i]=='0':
                code[i] = -1
        return code

             

if __name__ == '__main__':
    corpus = ["我要上厕所", "我要上个厕所", "我要去上厕所了"]
    corpus = list(open(r"C:\Users\Administrator\Desktop\简单任务\算法学习\simhash\corpus.txt", 'r', encoding='utf8'))
    a = SimHashV2()
    for i in range(len(corpus)):
        for j in range(i, len(corpus)):
            distance = a.get_distance(corpus[i], corpus[j])
            print(i+1, j+1, distance)
        
