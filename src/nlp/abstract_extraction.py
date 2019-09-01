import pickle, json, requests
import numpy as np

#调用分词工具得到分词结果
def get_words(content):
    restfulUrl = 'http://192.168.1.201:1240/splitWord'
    data = json.dumps([{'title':'','content':content }])
    headers = {"Content-Type": "application/json"}
    resp = requests.post(restfulUrl, headers=headers, data=data.encode('utf-8'))
    return json.loads(resp.text)[0]#['林志玲', '结婚', '了', '。', '对象', '不是', '言承旭', '。']

class KeyWordsBasesOnTermFreq():
    
    def get_freq(self, words):#统计一个词语列表中的term freq
        word_freq_map = {}
        for word in words:
            word_freq_map[word] = word_freq_map.get(word, 0) + 1
        return word_freq_map
    
    #基于词语频数/term freq抽取关键词
    def get_keywords_based_on_freq(self, content, topN=10):
        words = get_words(content)#分词
        word_freq_map = self.get_freq(words)#获取词频
        keywords = sorted(word_freq_map.items(), key=lambda x: x[1], reverse=True)#按照词频倒序排列
        keywords = keywords[:topN]#挑选关键词
        keywords = list(map(lambda x: x[0],keywords))
        return keywords
    
class KeyWOrdsBasedOnTFIDF(KeyWordsBasesOnTermFreq):
    
    def __init__(self, corpus_path=None, idf_path=None):
        self.init_idf(corpus_path, idf_path)
    
    def init_idf(self, corpus_path, idf_path):
        #初始化idf.如果制定了文件路径就直接加载；如果没有指定就训练一个
        if idf_path==None:
            if corpus_path==None:
                corpus =["林志玲结婚了。对象不是言承旭。不久前林志玲在微博发布了自己结婚的消息，并附上了和丈夫的甜蜜合照。丈夫是日本艺人，两人8年前共同主演舞台剧《赤壁 爱》时相识、去年底开始的交往、6月6日结婚！",
                              "一直对婚姻、爱情淡定闭口不谈的林志玲，第一次说：我准备好面对未知的未来。而林志玲的老公黑泽良平（EXILE-AKIRA）也甜蜜承诺：会照顾她一辈子。"
                              ,"气场很足的夫妇。我先找到了黑泽良平组合的歌，听着看一下他的故事吧。林志玲大家都很熟悉——一直以来都是漂亮、情商高的代表。44岁可以清纯可爱。"
                              ,"想起早前林志玲被炮轰演技为零，她笑着说：“是不是因为我名字里有个玲字？不过不管怎么样，既然说演技，那说明大家都已经承认我是演员了，这也是进步呀"
                              ,"Akira黑泽良平今年37岁，小林志玲七岁。是日本组合Exile（人称民工团）的成员。"]
            else:
                corpus = list(open(corpus_path, 'r').readlines())
                corpus = list(map(lambda x: x.replace('\n', ''), corpus))
            words_list = list(map(get_words, corpus))
            doc_num = len(words_list)
            doc_freq_map = {}
            for words in words_list:
                for word in set(words):
                    doc_freq_map[word] = doc_freq_map.get(word, 0) + 1
            self.idf_map = {}
            for word in doc_freq_map:
                self.idf_map[word] = np.log(doc_num/(doc_freq_map[word] + 1)**0.6)
            print(self.idf_map)
            pickle.dump(self.idf_map, open('idf.pkl', 'wb'))
        else:
            self.idf_map = pickle.load(open(idf_path, 'rb'))
    
    def get_tfidf(self, content):
        words = get_words(content)
        tf_map = self.get_freq(words)
        tfidf_map = {}
        for word in tf_map:
            if word in self.idf_map:#训练语料中没有出现的词语就干掉了
                tfidf_map[word] = tf_map[word]*self.idf_map[word]
        return tfidf_map
    
    def get_keywords_based_on_tfidf(self, content, topN=10):
        tfidf_map = self.get_tfidf(content)#获取tfidf值
        keywords = sorted(tfidf_map.items(), key=lambda x: x[1], reverse=True)
        keywords = keywords[:topN]
        keywords = list(map(lambda x: x[0],keywords))
        return keywords
    
            
    
    
s = ['林志玲', '结婚', '了', '。', '对象', '不是', '言承旭', '。', '不久前', '林志玲', '在', '微博', 
     '发布', '了', '自己', '结婚', '的', '消息', '，', '并', '附上', '了', '和', '丈夫', '的', '甜蜜',
      '合照', '。', '丈夫', '是', '日本', '艺人', '，', '两', '人', '8', '年前', '共同', '主演', 
      '舞台剧', '《', '赤壁', '爱', '》', '时相识', '、', '去年底', '开始', '的', '交往', '、', '6月',
       '6', '日', '结婚', '！']
c = "林志玲结婚了。对象不是言承旭。不久前林志玲在微博发布了自己结婚的消息，并附上了和丈夫的甜蜜合照。丈夫是日本艺人，两人8年前共同主演舞台剧《赤壁 爱》时相识、去年底开始的交往、6月6日结婚！"
if __name__ == '__main__':
    kw_extractor_tf = KeyWordsBasesOnTermFreq()
    print(s)
    print(kw_extractor_tf.get_freq(s))
    print(kw_extractor_tf.get_keywords_based_on_freq(c))
    
    kw_extractor_tfidf = KeyWOrdsBasedOnTFIDF()
    print(kw_extractor_tfidf.get_keywords_based_on_tfidf(c))
    
    
    
    
