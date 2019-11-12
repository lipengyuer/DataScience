'''
Created on 2019年11月12日

@author: lipy
'''
#局部敏感哈希实验
import numpy as np

class RandomProjectionHash():
    
    def __init__(self, K=64):
        
        self.K = K
        chars = "abcdefghijklmnopqrstucwxyz "
        self.vocab_size = len(chars)
        self.random_vector = np.random.normal(0, 1, (K, self.vocab_size))#随机向量是参数，初始化之后就需要冻结，并存储起来。
        #当有新文本需要处理时，使用这个参数.
        #,以字或词的assic码或者hash码为随机种子，生成K个长度为vocab_size的一维向量，元素服从标准正态分布。
        #用K个随机向量分别与原始向量点积，就得到了一个长度为K的一维向量。接着进行点积是否大于0的判断和处理.
        self.char_id_map = {}
        for i in range(len(chars)): self.char_id_map[chars[i]] = i

    def random_projection_hash(self, text):
        lsh_code = np.zeros(self.K)
        for i in range(self.K):
            tf = np.zeros(self.vocab_size)
            for char in text:
                index = self.char_id_map[char]
                tf[index] += 1
            dot_res = np.dot(tf, self.random_vector[i, :])
            if dot_res>0:
                lsh_code[i] = 1
            else:
                lsh_code[i] = 0
        return lsh_code

    def random_hyperplane_hash(self, text):
        lsh_code = np.zeros(self.K)
        for i in range(self.K):
            tf = np.zeros(self.vocab_size)
            for char in text:
                index = self.char_id_map[char]
                tf[index] += 1
            dot_res = np.dot(tf, self.random_vector[i, :])
            if dot_res>0:
                lsh_code[i] = 1
            else:
                lsh_code[i] = -1
        return lsh_code
    
        
    def simlarity(self, code1, code2):
        score = 0
        for i in range(len(code1)):
            if code1[i]==code2[i]:
                score += 1
        return score/len(code1)
    
if __name__ == '__main__':
    RPHash = RandomProjectionHash()
    s = 'i am chinese asd asd  asd asd asd asd '
    code1 = RPHash.random_hyperplane_hash(s)
    print(code1)
    s = 'i am a chinese asd asd  asd asd asd asd '
    code2 = RPHash.random_hyperplane_hash(s)
    print(code2)
    print("相似度是", RPHash.simlarity(code1, code2))
    print("http://www.apabi.cn/ontologies/xisixiang#中国\u2014委内瑞拉高级混合委员会")
    np.random.seed(1)
    for i in range(10): print(np.random.normal())

    
    