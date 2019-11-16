'''
Created on 2019年11月4日
@author: Administrator
'''

from pyhanlp import HanLP
import hashlib
import numpy as np

#所有局部敏感哈希的基类
class LSH():
    
    #分词
    def segment(self, text):
        words = []
        for word_tag in HanLP.segment(text):
            words.append(word_tag.word)
        return words
    
    #计算两个数据串的海明距离
    def get_hamming_distance(self, hash_code1, hash_code2):
        if len(hash_code1)!=len(hash_code2): return -1#如果不等行长，返回异常信号
        diatance = 0
        for i in range(len(hash_code1)):
            if hash_code1[i]!=hash_code2[i]:
                diatance += 1
        return diatance

    #计算两个数据串的海明距离，要求输入是int值
    def get_hamming_distance_bit(self, hash_code1, hash_code2):
        res = hash_code1^hash_code2
        distance = 0
        while res!=0:
            distance += res&1#如果res的末尾一位是1,说明找到了一处不相等的地方，距离加1
            res>>=1#判断完末尾一位之后，将它删掉。删除操作可以用向右移动一位来表示
        return distance
         
    #计算文档之间的距离
    def get_distance(self, text1, text2):
        pass
    
if __name__ == '__main__':
    a = LSH()
    print(a.get_hamming_distance_bit(123, 1214))
    

