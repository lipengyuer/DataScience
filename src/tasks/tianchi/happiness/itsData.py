import  pandas as pd
import numpy as np

rootPath = './data/'


class property_other_data():
    
    def __init__(self):
        with open(rootPath + 'property_other_data.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
        train_data = []
        for line in lines:
            if '#' in line:
                train_data.append([])
            else:
                train_data[-1].append(line.replace('\n', ''))
        self.clf_data = train_data
        
    def get_n_grams(self, text, n=2):
        res = []
        for i in range(len(text)-n):
            res.append(text[i:i+n])
        return set(res)
    
    def cal_simi_score(self, string1, string2):
        ngrams1 = self.get_n_grams(string1)
        ngrams2 = self.get_n_grams(string2)
        score = len(ngrams2&ngrams1)
        return score
    
    def predict(self, text):
        if not text or text==-1: return 0
        max_score = 0
        label = 0
        for class_no in range(len(self.clf_data)):
            for clf_text in self.clf_data[class_no]:
                temp_score = self.cal_simi_score(text, clf_text)
                if temp_score>max_score:
                    label = class_no
                    max_score = temp_score
        return label+1#让标签的取值为1,2,3,4
    
 
    
if __name__ == '__main__':
    clf = property_other_data()
    print(clf.predict("没有房产证"))
    
    
    