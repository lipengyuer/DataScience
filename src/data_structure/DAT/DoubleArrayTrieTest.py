'''
Created on 2020年1月15日

@author: lipy
'''
#基于双数组实现前缀树
import numpy as np
#https://segmentfault.com/a/1190000008877595
char_id_map = {"哈": 0, "工": 1, "大":2, "人": 3, "民": 4, "力": 5, "量": 6, "伟": 7}

class DoubleArrayTrie():
    
    def __init__(self):
        base = "1    1    1    2    1    -1    3    -1    8    1    -1"
        check = "0    0    1    0    0    4    0    3    0    6    8"
        
        self.base = self.str2ints(base)
        self.check = self.str2ints(check)
        self.size = 1
        self.max_ascii_id = 0
        self.min_ascii_id = 0
    
    def str2ints(self, a_str):
        slices = a_str.split("    ")
        slices = list(map(int, slices))
        return slices
    
    def containsKey(self, term):
        start_status = 0
        for a_char in term:            
            new_index = np.abs(self.base[start_status]) + char_id_map[a_char]
            print(start_status, new_index, a_char)
            if self.base[new_index]==0:#如果位置是空的
                return False
            else:#如果位置不为空，需要处理冲突
                if np.abs(self.check[new_index]) == start_status:#如果当前节已经收录，不需要插入，开始考虑下一个状态
                    start_status = new_index
                    continue
                else:
                    return False
                start_status = new_index
        if self.base[new_index] < 0:
            return True
        else:
            return False
        
if __name__ == '__main__':
    dat = DoubleArrayTrie()
    
    print(dat.containsKey("人民"))

