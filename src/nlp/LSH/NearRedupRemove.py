'''
Created on 2019年11月16日

@author: Administrator
'''
#基于simhash的文本去重框架
from simhash_v2 import SimHashV2

class NearDuplicateRemove():
    
    def __init__(self, simhash_dim=64, sub_code_num=4):
        if simhash_dim%sub_code_num!=0:
            print("你这是干啥？")
            return
        self.simhash_dim = simhash_dim#simhash编码的长度
        self.sub_code_num = sub_code_num#在构建倒查索引的时候，使用的key长度
        self.iindex_key_len = int(simhash_dim/sub_code_num)
        self.iindex_list = [{} for _ in range(self.iindex_key_len)]
        self.doc_id_simhash_code_map = {}#存储每篇文档的simhash编码
        self.simhasher = SimHashV2()
    
    #处理一个文档，如果不是重复文档，添加到倒查索引中；如果是重复的，提醒即可    
    def process_a_doc(self, doc_content, doc_id):
        if_dup, simhash_code = self.if_a_dup_doc(doc_content)
        if if_dup:
            print("这篇文档是重复的", doc_id)
        else:
            self.insert_into_iindex(simhash_code, doc_id)
    
    def if_a_dup_doc(self, text):
        simhash_code = self.simhasher.hash(text)
        cand_similar_doc_ids = self.recall_sub_set(simhash_code)
        for doc_id in cand_similar_doc_ids:
            if self.near_duplicate(simhash_code, self.doc_id_simhash_code_map[doc_id]):
                return True, simhash_code
        return False, simhash_code
    
    def near_duplicate(self, code1, code2):
        if self.simhasher.get_hamming_distance_bit(code1, code2)<=3:
            return True
        else:
            return False
        
    def recall_sub_set(self, a_simhash_code):
        cand_doc_ids = []
        a_simhash_code = str(a_simhash_code)
        for i in range(self.sub_code_num):
            key = a_simhash_code[i*self.iindex_key_len: (i+1)*self.iindex_key_len]
            cand_doc_ids += self.iindex_list[i].get(key, [])
        return cand_doc_ids
    
    def insert_into_iindex(self,a_simhash_code, doc_id):
        simhash_code_str = str(a_simhash_code)
        for i in range(self.sub_code_num):
            key = simhash_code_str[i*self.iindex_key_len: (i+1)*self.iindex_key_len]
            if key not in self.iindex_list[i]:
                self.iindex_list[i][key] = []
            self.iindex_list[i][key].append(doc_id)
        self.doc_id_simhash_code_map[doc_id] = a_simhash_code
        
    
if __name__ == '__main__':
    corpus = list(open(r"C:\Users\Administrator\Desktop\简单任务\算法学习\simhash\corpus.txt", 'r', encoding='utf8'))
    remover = NearDuplicateRemove()
    for i in range(len(corpus)):
        remover.process_a_doc(corpus[i], i)
        
        