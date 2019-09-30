#基于概率的隐语义分析模型pLSA
#使用EM算法球模型参数
from pyhanlp import HanLP

class PLSA():
    
    def __init__(self):
        pass
    
    def fit(self):
        pass
    
    def have_a_look(self):
        pass
    
    def segment(self, text):
        word_tag_list = HanLP.segment(text)
        word_list = []
        for word_tag in word_tag_list:
            word, tag = str(word_tag).split('/')
            if tag=='n':
                word_list.append(word)
        return word_list

document_list = ["明天就是国庆节了，祝愿祖国永远繁荣昌盛。",
          "明天就是国庆节了，可以画家看孩子，真开心。",
          "明天可以看国庆大阅兵，真开心。",
          "国庆大阅兵的时候，我在火车上，信号不好，完犊子。",
          "看回放也是极好的，NBA决赛的回放仍然刺激。",
          "国庆大阅兵中，我们可以看到好多新装备。",
          "我的孩子是个调皮鬼，她自己也承认。"]

if __name__ == '__main__':
    topic_model = PLSA()
    topic_model.fit(document_list)