'''
Created on 2019年6月7日

@author: Administrator
数据预处理部分
'''
import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
import utils
import pickle, json
import preProcess

stop_chars = {"的"}
class PreprocessCorpus():
    
    def __init__(self):
        self.parameters = None
        self.load_parameters()
    
    def load_parameters(self):
        self.parameters = json.load(open('parameters.json', 'r', encoding='utf8'))
        #print(self.parameters)

    def get_char_id_map(self):
        class_label_id_map = {}
        char_freq_map = {}
        all_files = []
        print("正字读取语料，并统计字的频率")
        utils.find_all_files(self.parameters['train_corpus_dir'], all_files)
        count = 0
        for file_name in all_files:
            
            class_name = file_name.split("/")[-2]
            if class_name not in class_label_id_map:
                class_label_id_map[class_name] = len(class_label_id_map)
                
            this_class_sample_size = 0
            for lines in utils.read_lines_small_file(file_name):
                for line in lines:
                    line = preProcess.filtUrl(line)
                    line = line.replace(' ', '').replace('\n', '')
                    if len(line)<10: continue
                    count += 1
                    this_class_sample_size += 1
    #                 if this_class_sample_size==100: break
                    if count%10000==0: print("已经读取了", count, '行。', "字符数量是", len(char_freq_map))
                    for char in line:
                        char_freq_map[char] = char_freq_map.get(char, 0) + 1
        
        print("正在为每一个字分配一个id")
        char_id_map = {'unk':0, 'pad_char': 1, 'stop_char': 2}
        id_char_map = {0: 'unk', 1: 'pad_char', 2:'stop_char'}
        init_char_id_map_size = len(char_id_map)
        char_freq_list = sorted(char_freq_map.items(),\
                                 key=lambda x: x[1], reverse=True)\
                                 [:self.parameters['char_set_size']-len(id_char_map)]
        for i in range(len(char_freq_list)):
            
            [char, _] = char_freq_list[i]
            if char not in stop_chars:
                char_id_map[char] = i + init_char_id_map_size
                id_char_map[i + init_char_id_map_size] = char

        pickle.dump(char_id_map, open(self.parameters['char_id_map_file'], 'wb'))
        print(char_id_map.keys())
        pickle.dump(id_char_map, open(self.parameters['id_char_map_file'], 'wb'))
        pickle.dump(class_label_id_map, open(self.parameters['class_label_id_map_file'], 'wb'))
        
if __name__ == '__main__':
    pc = PreprocessCorpus()
    pc.get_char_id_map()
        
        
        
            
        
            
                    
