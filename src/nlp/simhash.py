'''
Created on 2019年11月4日

@author: Administrator
'''
#使用simhash计算文本相似度，并去重
from pyhanlp import HanLP
import hashlib
import numpy as np

class SimHash():
    
    def __init__(self):
        self.doc_id_simhash_map  =None#存储所有文章的id和simhash码
            
    def segment(self, text):
        words = []
        for word_tag in HanLP.segment(text):
            words.append(str(word_tag).split('/')[0])
        return words
    
    def get_simhash_code(self, text):
        words = self.segment(text)
        simhash_code = np.zeros(64)
        for word in words:
            simhash_code += self.get_word_code(word)
#         print(simhash_code)
        for i in range(simhash_code.shape[0]):
            if simhash_code[i]>=0:
                simhash_code[i]=1
            else:
                simhash_code[0]=0
                
        return simhash_code
    
    def get_hamming_distance(self, hash_code1, hash_code2):
        return np.sum(np.abs(hash_code1-hash_code2))
    
    def get_distance(self, text1, text2):
        simhash_code1 = self.get_simhash_code(text1)
        simhash_code2 = self.get_simhash_code(text2)
        distance = self.get_hamming_distance(simhash_code1, simhash_code2)
        return distance
    
    def word_binary_hash(self, word):
        bin_code = ""
        md5_code = hashlib.md5(word.encode('utf8')).hexdigest()
        dec_code = int(md5_code, 16)
        for char in str(dec_code):
            binary_code = str(bin(int(char)))[2:] 
#             print(binary_code)
            four_bit_code = binary_code#'0'*(4-len(binary_code)) + binary_code
            bin_code += four_bit_code
        return bin_code
    
    def get_word_code(self, word):
        word_code = np.ones(64)
        binary_code = self.word_binary_hash(word)
        for i in range(64):
            if binary_code[i]=="0":
                word_code[i] = -1
        return word_code
    
    
corpus = ["我要上厕所", "我要上个厕所", "我要去上厕所了"]
corpus = list(open(r"C:\Users\Administrator\Desktop\简单任务\算法学习\simhash\corpus.txt", 'r', encoding='utf8'))
if __name__ == '__main__':
    a = SimHash()
    for i in range(len(corpus)):
        for j in range(i, len(corpus)):
            distance = a.get_distance(corpus[i], corpus[j])
            print(i, j, distance)
        
        
        