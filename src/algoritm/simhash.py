'''
Created on 2019年11月4日
@author: Administrator
'''
#使用simhash计算文本相似度，并去重
from pyhanlp import HanLP
import hashlib
import numpy as np

class SimHash():
    
    def segment(self, text):
        words = []
        for word_tag in HanLP.segment(text):
            words.append(str(word_tag).split('/')[0])
        return words
    
    def get_simhash_code(self, text):
        words = self.segment(text)
        simhash_code = np.zeros(64)
        for word in words:
            simhash_code += self.word_binary_hash(word)
#             simhash_code += self.string_hash(word)
            
        for i in range(64):
            if simhash_code[i]>=0:
                simhash_code[i]=1
            else:
                simhash_code[i]=0
                
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
#         print(bin_code)
        code = np.ones(64)
        for i in range(64):
            if bin_code[i]=='0':
                code[i] = -1
        return code
    
    def get_word_code(self, word):
        word_code = np.ones(64)
        binary_code = self.string_hash(word)
        for i in range(64):
#             print(binary_code[binary_index])
            if binary_code[i]=="0":
                word_code[i] = -1
#         print(word_code)
        return word_code
    
    def string_hash(self,source):
        code = np.ones(64)
        if source == "":
            return code
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(64)[-64:]
#                 print(source,x)
        for i in range(len(x)):
            if x[i]=='0':
                code[i] = -1
#         print(code)  
        return code
             
corpus = ["我要上厕所", "我要上个厕所", "我要去上厕所了"]
corpus = list(open(r"c:\Users\lipy\Desktop\new 18", 'r', encoding='utf8'))
if __name__ == '__main__':
    a = SimHash()
#     print(a.get_simhash_code("我"))
    for i in range(len(corpus)):
        for j in range(i, len(corpus)):
            distance = a.get_distance(corpus[i], corpus[j])
            print(i+1, j+1, distance)
        
        
        