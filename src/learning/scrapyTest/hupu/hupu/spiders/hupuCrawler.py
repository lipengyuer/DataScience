import scrapy
from hupu.items import HupuItem
from scrapy.http import Request
from bs4 import BeautifulSoup
import re, time

class HupuSpider(scrapy.Spider):

    name = "hupu"# 爬虫名
    allowed_domains = ["hupu.com"]# 爬虫作用范围
    headers = {
        'Connection': 'keep - alive',  # 保持链接状态
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.82 Safari/537.36'
    }
    def loadCookie(self):
        with open('cookie.txt', 'r') as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.replace('\n', '').split('\t'), lines))
            cookieDict = {}
            for line in lines: cookieDict[line[0]] = line[1]
        return cookieDict

    def start_requests(self):
        self.hupuCookie = self.loadCookie()
        # print(self.hupCookie)
        count = 0
        t1 = time.time()
        for i in range(15000000 + 20000, 17000000):
        # for i in range(25401625, 25401625 + 1):#开放区的帖子
            count += 1
            url = "https://bbs.hupu.com/" + str(i) + '.html'
            t2 = time.time()
            tcost = t2 - t1
            print("第", i, "速度是：", int(count / (tcost + 1)), url)
            yield Request(url, self.parse, cookies=self.hupuCookie,
                          headers=self.headers, meta={'post_id': str(i)})
            # yield result

    #解析第一页
    def parse(self, response):
        # 初始化模型对象
        # print(response)
        pageNumber = int(response.xpath('//*[@id="j_data"]/@data-maxpage').extract()[0])
        # print(pageNumber)
        resList, post_id = self.getContentInAPost_1stPage(response)
        for item in resList: yield item
        # print("总页数是", pageNumber)
        for i in range(2, pageNumber + 1):
            aUrl_i = 'http://bbs.hupu.com/' + post_id + '-' + str(i) + '.html'
            # print(aUrl_i)
            yield Request(aUrl_i, callback=self.sparsePageN,cookies=self.hupuCookie,
                          headers=self.headers,  meta={'tid': post_id, 'pageNO': i})

    #解析第2页开始的跟帖
    def sparsePageN(self, response):
        # print(response.body)
        list_foll_posts = getContentInAPost_nstPage(response, response.meta['tid'])
        # print("这一页的回帖数是", len(list_foll_posts))
        for data in list_foll_posts:
            item = orgnizePost(data)
            yield item

    def getContentInAPost_1stPage(self, response):  # 提取帖子的一页中的内容，包括帖子标题，每个post的发布者名字、发布时间、内容。
        # 回复，亮了的跟帖，浏览
        # 回复总数
        advPostData = {'type': 'advPost'}
        num_reply = response.xpath('//*[@id="t_main"]/div[2]/div[2]/span/span[1]').extract()
        num_reply = '' if len(num_reply)==0 else num_reply[0]
        num_reply = re.findall('[0-9]+', str(num_reply))
        num_reply = int(num_reply[0]) if len(num_reply)>0 else 0
        advPostData['num_reply'] = num_reply
        #print("主贴的回复数是", num_reply)
        #亮了的回帖总数
        num_light_reposts = response.xpath('//*[@id="t_main"]/div[2]/div[2]/span/span[2]').extract()
        num_light_reposts = '' if len(num_light_reposts) == 0 else num_light_reposts[0]
        num_light_reposts = re.findall('[0-9]+', str(num_light_reposts))
        num_light_reposts = int(num_light_reposts[0]) if len(num_light_reposts)>0 else 0
        advPostData['num_light_reposts'] = num_light_reposts
        # print("这个主贴的亮帖总数是", num_light_reposts)
        #浏览数
        num_brows = -1
        advPostData['num_brows'] = num_brows

        # 用户名
        uname = response.xpath('//*[@id="tpc"]/div/div[2]/div[1]/div[1]/a/text()').extract()
        uname = '' if len(uname) == 0 else uname[0]
        advPostData['uname'] = uname[:20000]

        # print("发帖者的用户名是", uname)
        uid = response.xpath('//*[@id="tpc"]/div/div[2]/div[1]/div[1]/a/@href').extract()
        uid = 'kong' if len(uid)==0 else uid[0]
        uid = uid.split('/')[-1] if '/' in uid else uid
        advPostData['uid'] = uid

        # print("这个用户的uid是", uid)
        # 主贴内容
        ori_context = response.xpath('//*[@id="tpc"]/div/div[2]/table[1]/tbody/tr/td/div[2]').extract()
        ori_context = ori_context[0]
        advPostData['ori_context'] = ori_context[:20000]

        # print("主贴内容是", ori_context[:10])
        # 推荐数
        num_rec = response.xpath('//*[@id="Recers"]/a/text()').extract()
        num_rec = '' if len(num_rec) == 0 else num_rec[0]
        num_rec = re.findall('[0-9]+', num_rec)
        num_rec = int(num_rec[0]) if len(num_rec)>0 else 0
        advPostData['num_rec'] = num_rec
        # print("推荐数是", num_rec)
        # 赞赏情况gold-users
        num_gold_infor = response.xpath('//*[@id="showmjrs"]/div[2]')
        num_gold, num_gold_pro = 0, 0
        # print("初始化，这个主贴的赞赏",num_gold_infor)
        if len(num_gold_infor)>0:
            num_gold_infor = num_gold_infor[0]

            num_gold = num_gold_infor.xpath('text()').extract()

            num_gold = re.findall('[0-9]+', num_gold[1])
            num_gold = 0 if len(num_gold)==0 else int(num_gold[0])
            # print("赞赏的金币是", num_gold)
            num_gold_pro = num_gold_infor.xpath('a/text()').extract()
            num_gold_pro = num_gold_pro[0] if len(num_gold_pro)>0 else '0'
            num_gold_pro = re.findall('[0-9]+', num_gold_pro)
            num_gold_pro = int(num_gold_pro[0])
            # print("赞赏的金币人数是", num_gold_pro)
        advPostData['num_gold_pro'] = num_gold_pro
        advPostData['num_gold'] = num_gold

        # print("这个主贴的赞赏人数是", num_gold_pro, '收到的金币数是', num_gold)
        # 发帖时间
        post_time = response.xpath('//*[@id="tpc"]/div/div[2]/div[1]/div[1]/span[3]/text()').extract()
        post_time = '' if len(post_time) == 0 else post_time[0]
        if post_time!='':
            stat_date = post_time.split(" ")[0]  # 发帖日期，用于hive表分区
            timeArray = time.strptime(post_time, "%Y-%m-%d %H:%M")
            post_time = int(time.mktime(timeArray))
        else:
            stat_date = '2019-01-01'
            timeArray = time.strptime( '2019-01-01 00：00', "%Y-%m-%d %H:%M")
            post_time = int(time.mktime(timeArray))
        # print(stat_date, post_time)
        advPostData['post_time'] = post_time
        advPostData['stat_date'] = stat_date[:200]
        # print("主贴的时间是", stat_date, post_time)

        # 板块
        mainBlock = response.xpath('//*[@id="t_main"]/div[2]/div[1]/a[2]/text()').extract()
        mainBlock = 'kong' if len(mainBlock)==0 else mainBlock[0]
        detailBlock = response.xpath('//*[@id="t_main"]/div[2]/div[1]/a[3]/text()').extract()
        detailBlock = 'kong' if len(detailBlock)==0 else detailBlock[0]
        # print("主贴的板块信息是", mainBlock, detailBlock)
        advPostData['mainBlock'] = mainBlock[:20000]
        advPostData['detailBlock'] = detailBlock[:20000]

        # 标题
        title = response.xpath('//*[@id="tpc"]/div/div[2]/table[1]/tbody/tr/td/div[1]/span/text()').extract()
        title = 'kong' if len(title)==0 else title[0]
        advPostData['title'] = title[:20000]

        # print("主贴的标题是", title)
        post_id = response.meta['post_id']
        advPostData['post_id'] = post_id[:20000]

        item = orgnizePost(advPostData)
        resList = [item]
        # print("###############主贴信息打印完毕##################")
        ############主贴信息抽取完毕###############
        # 亮帖信息
        lightPostData = getLightedPostInfor(response)  # 获得所有亮帖信息
        resList += lightPostData
        # print("###############亮帖信息打印完毕##################")

        #######################
        # 跟帖信息
        list_foll_posts = getFollPostInfor(response, post_id)  # 获得第一页所有跟帖信息
        resList += list_foll_posts
        # print("###############跟帖信息打印完毕##################")
        #####################
        return resList, post_id

# 获取亮帖信息
def getLightedPostInfor(response):
    res = []
    lightPosts_ = response.xpath('//*[@id="readfloor"]/*')
    count = 0
    for mySelector in lightPosts_:
        count += 1
        lightPostData = {'type': 'lightPost'}
        # print("亮帖", count, "的信息。")
        # uname = mySelector.xpath('div/div/@uname')[0]#回帖者用户名
        post_time = mySelector.xpath('div/div[@class="author"]/div[@class="left"]/span[@class="stime"]/text()')[0].extract()# 发布时间
        post_time = '2019-01-01 00:00' if len(post_time)==0 else post_time
        stat_date = post_time.split(" ")[0]  # 发帖日期，用于hive表分区
        timeArray = time.strptime(post_time, "%Y-%m-%d %H:%M")
        post_time = int(time.mktime(timeArray))
        lightPostData['post_time'] = post_time
        lightPostData['stat_date'] = stat_date

        # print("亮帖", count, "的时间", stat_date, post_time)
        # 点亮次数
        num_light = mySelector.xpath('div/div/div/span/span[@class="ilike_icon_list"]/span[@class="stime"]/text()').extract()
        num_light = -1 if len(num_light)==0 else num_light[0]
        num_light = int(num_light)
        lightPostData['num_light'] = num_light

        # print("亮帖", count, "的点亮次数", num_light)
        uid = mySelector.xpath('div[2]/div[1]/div[1]/span[3]/@uid').extract()
        uid = 'kong' if len(uid)==0 else uid
        lightPostData['uid'] = uid

        # print("亮帖", count, "的发帖者uid是", uid)
        # 内容
        content = mySelector.xpath('div/table/tbody').extract()
        # print(content)
        content = 'kong' if len(content)==0 else content[0]
        lightPostData['content'] = content[:20000]
        # print("亮帖", count, "的内容", content[:10])
        num_goldInfor = mySelector.xpath('div[@class="floor_box"]/div[@class="reply-sponsor-users"]')
        # print(num_gold)
        num_gold = num_goldInfor.xpath('text()').extract()
        num_gold = '0' if len(num_gold)==0 else num_gold[1]
        num_gold = re.findall('[0-9]+', num_gold)
        num_gold = int(num_gold[0])

        num_gold_pro = num_goldInfor.xpath('a/text()').extract()
        num_gold_pro = '0' if len(num_gold_pro)==0 else num_gold_pro[0]
        num_gold_pro = re.findall('[0-9]+', num_gold_pro)
        num_gold_pro = int(num_gold_pro[0])
        lightPostData['num_gold_pro'] = num_gold_pro
        lightPostData['num_gold'] = num_gold

        # print("亮帖", count, "的赞赏情况是:金币数", num_gold ,"人数",  num_gold_pro)
        cited_floor, cited_uid, cited_content = 0, 'kong', 'kong'
        lightPostData['cited_floor'] = cited_floor
        lightPostData['cited_uid'] = cited_uid
        lightPostData['cited_content'] = cited_content[:20000]
        lightPostData['ifSticky'] = 1
        res.append(orgnizePost(lightPostData))
    return res

def getFollPostInfor(response, post_id):
    selectors = response.xpath('//*[@id="t_main"]/form/div[@class and @id]')[1:]#第一个是主贴
    res = []
    count = 0
    for selector in selectors:
        # 发布时间
        count += 1
        follData = {'type': "follData"}
        floor_num = selector.xpath('div/div[@class="floor_box "]/div/div/a[@class="floornum"]/@id').extract()
        floor_num = int(floor_num[0])
        follData['floor_num'] = floor_num
        # print("回帖的楼层", floor_num)
        post_time = selector.xpath('div[@class="floor-show  "]/div[@class="floor_box "]/div[@class="author"]/div[@class="left"]/span[@class="stime"]/text()').extract()
        # print(post_time)
        post_time = '2019-01-01 00:00' if len(post_time)==0 else post_time[0]

        stat_date = post_time.split(" ")[0]  # 发帖日期，用于hive表分区
        timeArray = time.strptime(post_time, "%Y-%m-%d %H:%M")
        post_time = int(time.mktime(timeArray))
        # print("回帖的时间是", stat_date, post_time)
        follData['post_time'] = post_time
        follData['stat_date'] = stat_date

        # 点亮次数
        num_light = selector.xpath('div[@class="floor-show  "]/div[@class="floor_box "]/div[@class="author"]/div[@class="left"]/span/span[@class="ilike_icon_list"]/span[@class="stime"]/text()').extract()
        # print(num_light)
        num_light = '0' if len(num_light)==0 else num_light[0]
        num_light = int(num_light)
        follData['num_light'] = num_light

        # print("回帖的点亮次数是", num_light)
        uid = selector.xpath('div/div/div')[0].xpath('@uid').extract()
        uid = 'kong' if len(uid) else uid[0]
        follData['uid'] = uid

        # 内容
        content = selector.xpath('div[@class="floor-show  "]/div[@class="floor_box "]/table/tbody/tr/td/text()').extract()
        content = ''.join(content)
        follData['content'] = content[:20000]

        # print("回帖的内容是", content[:10])
        #引用帖子的情况
        citedInfo = selector.xpath('div[@class="floor-show  "]/div[@class="floor_box "]/table/tbody/tr/td/blockquote/p')
        # print(citedInfo)
        if len(citedInfo)==0:
            cited_uid = "null"
            cited_floor = -1
            cited_content = "null"
        else:
            citedInfo = citedInfo[0]
            # print(citedInfo)
            cited_uid = citedInfo.xpath('b/a/@href').extract()
            # print(cited_uid)
            cited_uid = 'kong' if len(cited_uid)==0 else cited_uid[0]
            # print(cited_uid)
            cited_uid = cited_uid.split('/')[-1]
            cited_floor = citedInfo.xpath('b/text()').extract()
            # print(cited_floor)

            # print(citedInfo.xpath('b/text()'))
            cited_floor = '-1' if len(cited_floor)==0 else cited_floor[0]
            cited_floor = re.findall('[0-9]+', cited_floor)
            cited_floor = int(cited_floor[0]) if len(cited_floor)>0 else 0
            # print(cited_floor)
            content = citedInfo.xpath('text()').extract()
            cited_content = 'kong' if len(content)==0 else ''.join(content[1:])[:20000]

        follData['cited_uid'] = cited_uid
        follData['cited_floor'] = cited_floor
        follData['cited_content'] = cited_content[:20000]
        follData['ifSticky'] = 0
        # print("回帖引用的情况是",cited_uid,  cited_floor, cited_content[:10])
        # 获取赞赏情况
        num_gold, num_gold_pro = 0, 0
        goldInfo = selector.xpath('div[@class="floor-show  "]/div[@class="floor_box "]/div[@class="reply-sponsor-users"]')
        if len(goldInfo)>0:
            num_gold_pro = goldInfo.xpath('a/text()').extract()
            num_gold_pro = '0' if len(num_gold_pro)==0 else num_gold_pro[0]
            num_gold_pro = re.findall('[0-9]+', num_gold_pro)
            num_gold_pro = int(num_gold_pro[0])
            num_gold = goldInfo.xpath('text()').extract()[1]
            num_gold = re.findall('[0-9]+', num_gold)
            num_gold = 0 if len(num_gold)==0 else int(num_gold[0])
        # print("回帖的赞赏情况是", num_gold_pro, num_gold)
        follData['num_gold_pro'] = num_gold_pro
        follData['num_gold'] = num_gold
        follData['post_id'] = post_id
        follData['ifSticky'] = 0
        res.append(orgnizePost(follData))
    return res

def getContentInAPost_nstPage(response,post_id):  # 提取帖子的一页中的内容，包括帖子标题，每个post的发布者名字、发布时间、内容。
    selectors = response.xpath('//*[@id="t_main"]/form/div[@class and @id]')[0:]#第一个是主贴
    res = []
    count = 0
    # print("这这一页的回帖数是", len(selectors))
    for selector in selectors:
        # 发布时间
        count += 1
        follData = {'type': "follData"}
        floor_num = selector.xpath('div/div[@class="floor_box "]/div/div/a[@class="floornum"]/@id').extract()
        floor_num = int(floor_num[0])
        follData['floor_num'] = floor_num
        # print("回帖的楼层", floor_num)
        post_time = selector.xpath(
            'div[@class="floor-show  "]/div[@class="floor_box "]/div[@class="author"]/div[@class="left"]/span[@class="stime"]/text()').extract()
        # print(post_time)
        post_time = '2019-01-01 00:00' if len(post_time) == 0 else post_time[0]

        stat_date = post_time.split(" ")[0]  # 发帖日期，用于hive表分区
        timeArray = time.strptime(post_time, "%Y-%m-%d %H:%M")
        post_time = int(time.mktime(timeArray))
        # print("回帖的时间是", stat_date, post_time)
        follData['post_time'] = post_time
        follData['stat_date'] = stat_date

        # 点亮次数
        num_light = selector.xpath(
            'div[@class="floor-show  "]/div[@class="floor_box "]/div[@class="author"]/div[@class="left"]/span/span[@class="ilike_icon_list"]/span[@class="stime"]/text()').extract()
        # print(num_light)
        num_light = '0' if len(num_light) == 0 else num_light[0]
        num_light = int(num_light)
        follData['num_light'] = num_light

        # print("回帖的点亮次数是", num_light)
        uid = selector.xpath('div/div/div')[0].xpath('@uid').extract()
        uid = 'kong' if len(uid) else uid[0]
        follData['uid'] = uid

        # 内容
        content = selector.xpath(
            'div[@class="floor-show  "]/div[@class="floor_box "]/table/tbody/tr/td/text()').extract()
        content = ''.join(content)
        follData['content'] = content[:20000]

        # print("回帖的内容是", content[:10])
        # 引用帖子的情况
        citedInfo = selector.xpath('div[@class="floor-show  "]/div[@class="floor_box "]/table/tbody/tr/td/blockquote/p')
        # print(citedInfo)
        if len(citedInfo) == 0:
            cited_uid = "null"
            cited_floor = -1
            cited_content = "null"
        else:
            citedInfo = citedInfo[0]
            cited_uid = citedInfo.xpath('b/a/@href').extract()
            cited_uid = 'kong' if len(cited_uid) == 0 else cited_uid[0]
            cited_uid = cited_uid.split('/')[-1]
            cited_floor = citedInfo.xpath('b/text()').extract()
            cited_floor = '-1' if len(cited_floor) == 0 else cited_floor[0]
            cited_floor = re.findall('[0-9]+', cited_floor)
            cited_floor = int(cited_floor[0]) if len(cited_floor)>0 else 0
            content = citedInfo.xpath('text()').extract()
            cited_content = 'kong' if len(content) == 0 else ''.join(content[1:])[:20000]

        follData['cited_uid'] = cited_uid
        follData['cited_floor'] = cited_floor
        follData['cited_content'] = cited_content[:20000]

        # print("回帖引用的情况是", cited_uid, cited_floor, cited_content[:10])
        # 获取赞赏情况
        num_gold, num_gold_pro = 0, 0
        goldInfo = selector.xpath(
            'div[@class="floor-show  "]/div[@class="floor_box "]/div[@class="reply-sponsor-users"]')
        if len(goldInfo) > 0:
            num_gold_pro = goldInfo.xpath('a/text()').extract()
            num_gold_pro = '0' if len(num_gold_pro) == 0 else num_gold_pro[0]
            num_gold_pro = re.findall('[0-9]+', num_gold_pro)
            num_gold_pro = int(num_gold_pro[0])
            num_gold = goldInfo.xpath('text()').extract()[1]
            num_gold = re.findall('[0-9]+', num_gold)
            num_gold = 0 if len(num_gold) == 0 else int(num_gold[0])
        # print("回帖的赞赏情况是", num_gold_pro, num_gold)
        follData['num_gold_pro'] = num_gold_pro
        follData['num_gold'] = num_gold
        follData['post_id'] = post_id
        res.append(orgnizePost(follData))
    return res

def orgnizePost(dataMap):
    item = HupuItem()
    for key in dataMap: item[key] = dataMap[key]
    return item