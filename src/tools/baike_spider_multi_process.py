import multiprocessing
import time


import os
import re
import multiprocessing
import time

import urllib.request
import urllib.parse
from lxml import etree

from src.data_base import get_connection
from src.data_preprocess.excel import open_xlsx, save_xlsx


class BaikeSpider(object):
    """
    百度百科爬取
    """

    def __init__(self):
        self.conn = get_connection.get_mysql()
        self.cur = self.conn.cursor()

    def get_html(self, url):
        """
        根据url获取html内容
        :param url: 请求的url地址
        :return: url对应的html内容
        """
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                 'Chrome/51.0.2704.63 Safari/537.36'}

        try:
            req = urllib.request.Request(url=url, headers=headers)
            res = urllib.request.urlopen(req)
            html = res.read().decode('utf-8')
        except:
            html = res.read().decode('gbk')
        return html

    def phrase_url(self, phrase):
        """
        获取短语对应的html内容
        :param phrase: 需要查询的短语  type=str
        :return: 该短语对应的html内容
        """
        url = 'https://baike.baidu.com/item/' + urllib.request.quote(phrase)
        phrase_html = self.get_html(url=url)
        return phrase_html

    def urls_extract(self, content, xpath, url_prefix=''):
        """
        从html内容中抽取url列表
        :param content: html内容，即get_html返回的内容
        :param xpath: html内容中对应的url的xpath
        :param url_prefix: 解析的url可能不是完整的http地址，需要在解析的url前加上前缀字段
        :return: 全字段(可直接请求获取html)的url列表
        """
        selector = etree.HTML(content)
        urls = [url_prefix + i for i in selector.xpath(xpath)]
        return urls

    def text_extract(self, content, xpaths):
        """
        从html内容中抽取str
        :param content: html内容，即get_html返回的内容
        :param xpath: 一个html内容中对应的所有str的xpath的列表    type=list
        :return: 从一个html中抽取的文本内容     type=str
        """
        selector = etree.HTML(content)
        texts = []
        for xpath in xpaths:
            text = selector.xpath(xpath)
            texts.append(text)
        return texts

    def get_synonym(self, phrase, type, status='all', connect_db=False):
        """
        获取同义词
        :param phrase: 需要获取同义词的短语
        :param status: 'full'为获取全称，'abb'为获取简称
        :param connect_db: 是否连接数据库
        """
        phrase_html = self.phrase_url(phrase=phrase)
        # 简称获取全称
        if status == 'full':
            full = self.text_extract(content=phrase_html, xpaths=[
                '//div[@class="main-content"]//dd[@class="lemmaWgt-lemmaTitle-title"]/h1/text()'])
            synonym = {'full': full[0][0], 'abb': [phrase]}
        # 全称获取简称
        elif status == 'abb':
            abb = []
            # 1.标题旁的同义词 2.正文中的"简称" 3.半结构化文本处
            pre_abb = self.text_extract(content=phrase_html, xpaths=[
                '//div[@class="main-content"]//dd[@class="lemmaWgt-lemmaTitle-title"]/h2/text()',
                '//div[@class="main-content"]//div[@class="lemma-summary"]/div[contains(text(),"简称")]/text()[1]',
                '//div[@class="main-content"]//div[@class="basic-info cmn-clearfix"]/dl/dt[contains(text(),"简称")]/following-sibling::dd[1]/text()'])
            for i in range(len(pre_abb)):
                if len(pre_abb[i]) != 0:
                    if i == 1:
                        text_split = re.findall(pattern='[\u4e00-\u9fa5]+', string=pre_abb[i])
                        print(text_split)
                        # loc = [text_split.index(j) for j in text_split if '简称' in j][0]
                        # if text_split[loc] == '简称':
                        #     abb.append(text_split[loc + 1])
                        # elif '的' in text_split[loc].split('简称')[0]:
                        #     pass
                    else:
                        abb.append(pre_abb[i][0].replace('（', '').replace('）', '').replace('\n', ''))
            synonym = {'full': phrase, 'abb': list(set(abb))}
            # synonym = pre_abb
        elif status == 'all':
            full = self.text_extract(content=phrase_html, xpaths=[
                '//div[@class="main-content"]//dd[@class="lemmaWgt-lemmaTitle-title"]/h1/text()'])
            full = full[0][0] + '1' if len(full[0]) > 0 else phrase + '0'
            abb = []
            pre_abb = self.text_extract(content=phrase_html, xpaths=[
                '//div[@class="main-content"]/span[@class="view-tip-panel"]/span/text()',
                '//div[@class="main-content"]//dd[@class="lemmaWgt-lemmaTitle-title"]/h2/text()',
                '//div[@class="main-content"]//div[@class="basic-info cmn-clearfix"]/dl/dt[contains(text(),"简")]/following-sibling::dd[1]/text()'])
            print(self.text_extract(content=phrase_html, xpaths=[
                '//div[@class="main-content"]//div[@class="lemma-summary"]/div[contains(text(),"简称")]/text()[1]']))
            try:
                abb.extend([i[0].split('（')[1].split('）')[0].replace('\n', '') for i in pre_abb if len(i) > 0])
            except:
                abb.extend([i[0].replace('（', '').replace('）', '').replace('\n', '') for i in pre_abb if len(i) > 0])
            synonym = {'full': full[:-1], 'abb': list(set(abb)), 'status': full[-1]}

        if connect_db == True:
            print(synonym)
            self.cur.execute(
                'INSERT INTO baike_entity_copy (full_entity,abb_entity,category,baike_status,create_datetime,is_deleted) VALUES (%s,NOW(),0);' % (
                        '\'' + synonym['full'] + '\'' + ',' + '\'' + str(synonym['abb']).replace('[', '').replace(']',
                                                                                                                  '').replace(
                    '\'', '') + '\'' + ',' + '\'' + type + '\'' + ',' + synonym['status']))
            self.conn.commit()
        return synonym
    
def worker(phrases):
    spider = BaikeSpider()
    select_sql=''
    l = []
    # i = 1
    l.append(['全称', '简称', '百度百科是否包含该词'])
    for phrase in phrases:
        # i += 1
        print(phrase[0])
        synonym = spider.get_synonym(phrase=phrase[0], type=file.split('.xlsx')[0], connect_db=True)
        print(synonym)
        # l.append([synonym['full'], synonym['abb'], synonym['status']])
    # save_xlsx(outfile='../补全结构/mini' + file, content=l)
    
    
if __name__ == '__main__':
    path = '../../data/语义词典百科补全/mini/'
    phrases_all = []
    for file in os.listdir(path):
        select_sql=''
        l = []
        l.append(['全称', '简称', '百度百科是否包含该词'])
        phrases = open_xlsx(os.path.join(path, file))
        phrases_all += phrases
    process_num = 3
    batch_size = int(phrases_all/process_num) + 1
    for i in range(process_num):
        a_process = multiprocessing.Process(target=worker, args=(i, phrases_all[i * batch_size : i * batch_size + batch_size]))#args默认要求输入一个tuple,如果我们给一个(i)
        #，python会吧(i)解释为i,导致类型不一致错误。因此当tuple只有一个元素的时候，一定要在这个元素后面添加一个逗号，表示这是一个tuple。
        a_process.start()
        
        
        