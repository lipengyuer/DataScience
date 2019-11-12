'''
Created on 2019年11月12日

@author: lipy
'''
#局部敏感哈希实验
import numpy as np

class RandomProjectionHash():
    
    def __init__(self, K=10):
        
        self.K = K
        chars = "abcdefghijklmnopqrstucwxyz "
        self.vocab_size = len(chars)
        self.random_vector = np.random.normal(0, 1, (K, self.vocab_size))#随机向量是参数，初始化之后就需要冻结，并存储起来。
        #当有新文本需要处理时，使用这个参数
        self.char_id_map = {}
        for i in range(len(chars)): self.char_id_map[chars[i]] = i

    def hash(self, text):
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

if __name__ == '__main__':
    RPHash = RandomProjectionHash()
    s = 'i am chinese'
    res = RPHash.hash(s)
    print(res)
    s = 'i am a chinese'
    res = RPHash.hash(s)
    print(res)
    
    