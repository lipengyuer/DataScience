#用gensim训练字向量

import gensim as gs
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

'''
Created on 2019年6月7日

@author: Administrator
用cnn实现一个文本分类器
'''

import tensorflow as tf
import utils
import json, pickle
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
class EmbenddingTrainer():
    
    def __init__(self):
        self.load_parameters()
        self.load_char_id_data()
    
    def load_parameters(self):
        self.parameters = json.load(open('parameters.json', 'r', encoding='utf8'))
    
    def load_char_id_data(self):
        self.char_id_map = pickle.load(open(self.parameters['char_id_map_file'], 'rb'))
        self.id_char_map = pickle.load(open(self.parameters['id_char_map_file'], 'rb'))
    #直接读取训练语料，并转换为id,然后转换为字向量。默认是静态字向量；可以选择微调
    def fit(self, if_static_embeding=True):
        model = None
        file_list = utils.find_all_files(self.parameters['train_corpus_for_embedding'], [])
        print("训练数据的文件数量是", len(file_list))
        count = 0
        x_batch = []
        step = 0
        
        random.shuffle(file_list)
        for file_name in file_list:
            lines = utils.read_lines_small_file(file_name)
#                 text = self.get_title_content(lines)
            lines = list(map(lambda x: x.split('#'), lines))
            lines = list(filter(lambda x: len(x)==8 and len(x[6])>50, lines))
            lines = list(map(lambda x: x[6].split('kabukabu')[1].\
                             replace('d_post_content j_d_post_content  clearfix"> ', ''), lines))
            text = ''.join(lines).replace(' ', '')
            if len(text)==0: continue
#             print("文档字数 是", len(text))
            text = list(text)
            if len(text)==0: continue
            
            count += 1

            x_batch.append(text)
            if len(x_batch)==10:
                #打乱顺序
                random_index = list(range(10))
                random.shuffle(random_index)
                #print(x_batch)
                x_batch = np.array(x_batch)[random_index]
                #训练
                print("这是第", step)
                if model==None:
                    model = Word2Vec(x_batch, size=200, window=5, min_count=5, workers=8, iter=200)
                    step += 1
                else:
                    model.build_vocab(x_batch,update=True)
                    model.train(x_batch, total_examples=x_batch.shape[0],epochs=200)
                    step += 1
                x_batch = []
            if count%50==0:
                model.save("./model/word2vec.model")
        
    def trans_char2id(self, text):
        id_list = [self.char_id_map.get(char, 0) for char in text]
        length_dif = self.parameters['max_text_length'] - len(id_list)
        if length_dif>0: id_list += [1]*length_dif
        return id_list
            
    def get_title_content(self, lines):
        title_index, content_start_index = 0, 0
        for i in range(len(lines)):
            line = lines[i]
            if '标  题' in line:
                title_index = i
            if '正  文' in line:
                content_start_index = i
                break
#         print(lines)
        if title_index==0 or content_start_index==0:
            text = ''
        else:
            text = lines[title_index].split('】')[1] + '。' 
        for i in range(content_start_index + 1, len(lines)):
            text += lines[i]
        text = text.replace('\n', '').replace(' ', '')
        return text
                



if __name__ == '__main__':
    embedding_trainer = EmbenddingTrainer()
    embedding_trainer.fit()
    
#     model = Word2Vec.load("./model/word2vec.model")
#     print(model.similar_by_word("蓝", topn=10))
#     print(len(model.wv.__dict__['vocab'].keys()), model.wv.__dict__['vocab'].keys())

    
    

    