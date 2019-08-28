#基于ngrams来度量语言的多样性
#统计随着时间推移，语言的多样性的变化
from data_base import get_connection
import pandas as pd

class StasticsOnLanguageVariety():
    
    def __init__(self, max_N=2):
        self.max_N = max_N
    
    def basic_stastic_texts(self, text_list):
        ngram_freq = {}
        for text in text_list:
            for i in range(len(text)):
                for j in range(i, min(i + self.max_N, len(text))):
                    a_gram = text[i: j]
                    ngram_freq[a_gram] = ngram_freq.get(a_gram, 0) + 1
        return ngram_freq
    
    #获取文本片段的种类
    def get_gram_number(self, text_list):
        return len(self.basic_stastic_texts(text_list))
    

#对学习社项目的语料进行分析
def test_on_daogu():
    conn = get_connection.get_connection_mysql(db='daogu_test')
    sql_str = "select doc_content, doc_publish_time from resource_document"
    print("开始获取数据")
    data = pd.read_sql(sql_str, conn)
    data['date'] = data['doc_publish_time'].apply(lambda x: str(x)[:4])
    data = data.drop('doc_publish_time', axis=1)
    print(data.columns)
    counter = StasticsOnLanguageVariety(max_N=5)
    print("开始统计")
    data = data.groupby(by=['date']).apply(lambda x: counter.get_gram_number(x['doc_content'].tolist()))
    print("各个阶段，文本片段的多样性为:")
    print(data)
    
if __name__ == '__main__':
    test_on_daogu()
    
    
        