import scrapy
from hupu.items import HupuItem
from scrapy.http import Request
from bs4 import BeautifulSoup
import re, time

class TencentpositionSpider(scrapy.Spider):
    """
    功能：爬取腾讯社招信息
    """
    # 爬虫名
    name = "hupu"
    # 爬虫作用范围
    allowed_domains = ["hupu.com"]

    # 起始url
    # start_urls = [url + str(offset)]

    def start_requests(self):
        t1 = time.time()
        for i in range(10000000, 10000000 + 10000):
            url = "https://bbs.hupu.com/" + str(i) + '.html'
            print("第", i, time.time()-t1)
            yield Request(url, self.parse)
    #解析第一页
    def parse(self, response):
        # 初始化模型对象
        html = response.xpath("/html/body").extract()[0]
        #/html/body/pre/span[1353]
        advPostData, lightPostData, follPostData, N, post_id =\
               self.getContentInAPost_1stPage(html, response.url)
        # print(advPostData)
        for data in [advPostData, lightPostData, follPostData]:
            item = HupuItem()
            item['type'] = data['type']
            item['post_id'] = data['data'][0]
            # print(item)
            yield item
        for i in range(2, N + 1):
            aUrl_i = 'http://bbs.hupu.com/' + post_id + '-' + str(i) + '.html'
            yield  Request(aUrl_i, self.sparsePageN, meta={'tid': post_id})
        #
        # yield item

    #解析第2页开始的跟帖
    def sparsePageN(self, response):
        html = response.xpath("/html/body").extract()[0]
        list_foll_posts = getContentInAPost_nstPage(html, response.meta['tid'])
        for data in list_foll_posts:
            item = HupuItem()
            item['type'] = data['type']
            item['post_id'] = data['data'][0]
            # print(item)
            yield item

    def getContentInAPost_1stPage(self, html, aUrl):  # 提取帖子的一页中的内容，包括帖子标题，每个post的发布者名字、发布时间、内容。
        soup = BeautifulSoup(html)  # , "html5lib")
        lis = re.findall("pageCount:[0-9]+", str(html))  # <a herf=\"[0-9]+[-]+[0-9]+[.]html\"
        # 帖子页数
        if len(lis) == 0:
            N = 1
        else:
            N = int(lis[0].replace('pageCount:', ""))
        # 回复，亮了的跟帖，浏览
        try:  # 有的帖子被删
            bbs_head = soup.find(name="div", attrs={"class": "bbs_head"}).find(name="div",
                                                                               attrs={"class": "bbs-hd-h1"}).find(
                name="span", attrs={"class": "browse"})
        except:
            return -1
        # print "aa"
        bbs_head = str(bbs_head)
        temp = re.findall("[0-9]+回复", bbs_head)
        if temp != []:
            num_reply = int(temp[0].replace("回复", ""))
        else:
            num_reply = 0
        temp = re.findall("[0-9]+亮", bbs_head)
        # print "bb"
        if temp != []:
            num_light_reposts = int(temp[0].replace("亮", ""))
        else:
            num_light_reposts = 0
        temp = re.findall("[0-9]+浏览", bbs_head)
        # print "cc"
        if temp != []:
            num_brows = int(temp[0].replace("浏览", ""))
        else:
            num_brows = 0
        # 用户名和uid
        ori_post = soup.find_all(name="div", attrs={"id": "tpc", "class": "floor"})[0].find(name="div")
        uid_uname = ori_post.find(name="div").find(name="div", attrs={"class": "j_u"})  #
        uid = uid_uname.attrs["uid"]  # .decode("gbk","replace")#.encode("utf-8", 'ignore')
        uname = uid_uname.attrs["uname"]  # .encode("utf-8")
        # 主贴内容
        ori_context = ori_post.find(name="div", attrs={"class": "floor_box"}).find(name="div",
                                                                                   attrs={"class": "quote-content"})
        ori_context = str(ori_context).replace("\n", "").replace("<div class=\"quote-content\">", "").replace(
            "</br></div>", "")
        # 推荐数
        try:
            num_rec = ori_post.find(name="div", attrs={"class": "floor_box"}).find(name="span",
                                                                                   attrs={"id": "Recers"}).find(
                name="a")
            num_rec = str(num_rec)
            num_rec = int(re.findall("[0-9]+人", num_rec)[0].replace("人", ""))
        except:
            num_rec = 0
        # 赞赏情况gold-users
        num_gold_infor = str(ori_post.find(name="div", attrs={"class": "gold-users"}))
        num_gold = re.findall("JRs，赞赏了 [0-9]+ 虎扑币", num_gold_infor)
        if num_gold == []:
            num_gold = 0  # 金币总数
            num_gold_pro = 0  # 赞赏人数
        else:
            num_gold_pro = int(str(re.findall("[0-9]+个", num_gold_infor)[0]).replace("个", ""))
            num_gold = int(re.findall("[0-9]+", num_gold[0])[0])
        # 发帖时间
        post_time = ori_post.find(name="div", attrs={"class": "floor_box"}).find(name="span",
                                                                                 attrs={"class": "stime"})
        post_time = str(post_time).replace("<span class=\"stime\">", "").replace("</span>", "")
        stat_date = post_time.split(" ")[0]  # 发帖日期，用于hive表分区
        timeArray = time.strptime(post_time, "%Y-%m-%d %H:%M")
        post_time = int(time.mktime(timeArray))

        # 板块
        theBlock = str(ori_post.find(name="div", attrs={"class": "floor_box"}).
                       find(name="div", attrs={"class": "subhead"}))
        theBlock = theBlock.split('</a>')
        mainBlock = theBlock[0].split('>')[-1]
        detailBlock = theBlock[1].split('>')[-1]
        # print(mainBlock,detailBlock )
        # 题目
        title = str(
            ori_post.find(name="div", attrs={"class": "floor_box"}).find(name="div",
                                                                         attrs={"class": "subhead"}).find(
                name="span")).replace("<span>", "").replace("</span>", "").replace("\r", '').replace("\n", '')
        post_id = aUrl.replace("https://bbs.hupu.com/", "").replace(".html", "")

        # 主贴信息
        res_ori = [post_id, title, mainBlock, detailBlock, post_time, num_gold, num_gold_pro, num_rec, ori_context,
                   uid, uname, num_reply,
                   num_light_reposts, num_brows, stat_date]
        # print(sql_str)
        advPostData = {'type': 'advPost', 'data': res_ori}
        ###########################
        # 亮帖信息
        list_light_posts = soup.find_all(name="div", attrs={"id": "readfloor"})
        # print(list_light_posts)
        list_light_posts = getLightedPostInfor(list_light_posts)  # 获得所有亮帖信息
        if list_light_posts != -1:
            list_light_posts = list(
                map(lambda x: [post_id, 1, 0] + x,
                    list_light_posts))  # 把主贴id加上，一边以后链表查询;第二位是亮帖标志位;第三位是楼层数，亮帖的楼层数为0.
            sql_str = ""
            sql_str += "insert into hupu_bxj_foll_posts_1(post_id,if_lighted,floor_num,post_time,num_light,num_gold,num_gold_pro,uid,cited_floor,cited_uid,cited_content,content,stat_date) "
            sql_str += " values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"  # %tuple(list_light_posts)
            list_light_posts = list(map(lambda x: tuple(x), list_light_posts))
        lightPostData = list_light_posts
        lightPostData = {'type': 'lightPost', 'data': lightPostData}
        #######################
        # 跟帖信息
        follPostData = []
        post_foll = soup.find_all(name="div", attrs={"id": re.compile("[0-9]+"), "class": "floor"})
        # print(list_light_posts)
        list_foll_posts_I = getFollPostInfor(post_foll[len(list_light_posts):])  # 获得第一页所有跟帖信息
        if list_foll_posts_I != -1:
            list_foll_posts = list(map(lambda x: [post_id, 0] + x, list_foll_posts_I))  # 把主贴id加上，一边以后链表查询;第二位是亮帖标志位
            sql_str = ""
            sql_str += "insert into hupu_bxj_foll_posts_1(post_id,if_lighted,floor_num,post_time,num_light,num_gold,num_gold_pro,uid,cited_floor,cited_uid,cited_content,content,stat_date) "
            sql_str += " values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"  # %tuple(list_light_posts)

            list_foll_posts = list(map(lambda x: tuple(x), list_foll_posts))

            follPostData = list_foll_posts
            follPostData = {'type': 'follPost', 'data': follPostData}

        #####################
        return advPostData, lightPostData, follPostData, N, post_id

def getLightedPostInfor(lighted_posts):
    if lighted_posts == []:
        return []
    else:
        # print lighted_posts
        list_posts = []
        list_all = lighted_posts[0].find_all(name="div", attrs={"class": "floor"})
        for apost in list_all:
            list_posts.append(getPostInfor_light(apost))
    return list_posts

# 获取亮帖信息
def getPostInfor_light(apost):
    # 发布时间
    post_time = str(apost.find(name="span", attrs={"class": "stime"})).replace("<span class=\"stime\">", "").replace(
        "</span>", "")
    stat_date = post_time.split(" ")[0]  # 发帖日期，用于hive表分区
    timeArray = time.strptime(post_time, "%Y-%m-%d %H:%M")
    post_time = int(time.mktime(timeArray))
    # 点亮次数
    num_light = int(
        str(apost.find_all(name="span", attrs={"class": "stime"})[1]).replace("<span class=\"stime\">", "").replace(
            "</span>", ""))
    # uid
    uid = \
    apost.find(name="div", attrs={"class": "author"}).find(name="span", attrs={"uid": re.compile("[0-9]+")}).attrs[
        "uid"]
    # 内容
    content = apost.find(name="table").find(name="td")  # .find(name="blockquote")#
    cited = content.find(name="blockquote")
    cited_uid = "null"
    cited_floor = 0
    content = str(content)
    cited_content = "null"
    if cited != None:
        try:
            cited_uid_floor = re.findall("引用[0-9]+楼.+发表的", str(cited.find(name="b")))[0]
            cited_content = str(cited).replace(cited_uid_floor, "")
            [cited_floor, cited_uid] = cited_uid_floor.split("楼")
            cited_floor = int(cited_floor.replace("引用", ""))
            # print  re.findall("https:[/][/]my.hupu.com[/][0-9]+", cited_uid)
            cited_uid = re.findall("https:[/][/]my.hupu.com[/][a-zA-Z0-9]+", cited_uid)[0].split("/")[-1]
        except:
            pass
        content = content.replace(str(cited), "")
    num_gold_and_pro = str(apost.find(name="", attrs={"class":"reply-sponsor-users"}))
    num_gold = re.findall("[0-9]+个",num_gold_and_pro)
    num_gold_pro = re.findall("[0-9]+虎扑币",num_gold_and_pro)
    if num_gold==[] and num_gold_pro==[]:
        num_gold=0
        num_gold_pro=0
    else:
        num_gold = int(num_gold[0].replace("个",""))
        num_gold_pro = int(num_gold_pro[0].replace("虎扑币",""))
    res = [post_time, num_light,num_gold,num_gold_pro, uid, cited_floor, cited_uid, cited_content, content, stat_date]
    return res

def getFollPostInfor(foll_posts):
    if foll_posts == []:
        return -1
    else:
        # print lighted_posts
        list_posts = []
        for apost in foll_posts:
            list_posts.append(getPostInfor_foll(apost))
    return list_posts


def getPostInfor_foll(apost):
    # 发布时间
    #print(apost)
    post_time = str(apost.find(name="span", attrs={"class": "stime"})).replace("<span class=\"stime\">", "").replace(
        "</span>", "")
    stat_date = post_time.split(" ")[0]  # 发帖日期，用于hive表分区
    timeArray = time.strptime(post_time, "%Y-%m-%d %H:%M")
    post_time = int(time.mktime(timeArray))
    # 点亮次数
    num_light = int(
        str(apost.find_all(name="span", attrs={"class": "stime"})[1]).replace("<span class=\"stime\">", "").replace(
            "</span>", ""))
    # uid
    uid = \
    apost.find(name="div", attrs={"class": "author"}).find(name="span", attrs={"uid": re.compile("[0-9]+")}).attrs[
        "uid"]
    # 内容
    content = apost.find(name="table").find(name="td")  # .find(name="blockquote")#

    # print "qwe",content
    cited = content.find(name="blockquote")
    cited_uid = "null"
    cited_floor = 0
    content = str(content)#.decode("utf-8", "ignore").encode("utf-8")
    cited_content = "null"
    if cited != None:
        try:
            cited_uid_floor = re.findall("引用[0-9]+楼.+发表的", str(cited.find(name="b")))[0]
            cited_content = str(cited).replace(cited_uid_floor, "")
            [cited_floor, cited_uid] = cited_uid_floor.split("楼")
            cited_floor = int(cited_floor.replace("引用", ""))
            # print  re.findall("https:[/][/]my.hupu.com[/][0-9]+", cited_uid)
            cited_uid = re.findall("https:[/][/]my.hupu.com[/][a-zA-Z0-9]+", cited_uid)[0].split("/")[-1]
        except:
            pass
        content = content.replace(str(cited), "")
    try:
        if str(apost.find(name="a", attrs={"class": "floornum"}).attrs["id"]) == "":
            floor_num = 0
        else:
            floor_num = int(str(apost.find(name="a", attrs={"class": "floornum"}).attrs["id"]))
    except:
        if str(apost.find(name="a", attrs={"class": "reply"}).attrs["lid"]) == "":
            floor_num = 0
        else:
            floor_num = int(str(apost.find(name="a", attrs={"class": "reply"}).attrs["lid"]))

    #获取赞赏情况
    num_gold_and_pro = str(apost.find(name="", attrs={"class":"reply-sponsor-users"}))
    num_gold = re.findall("[0-9]+个",num_gold_and_pro)
    num_gold_pro = re.findall("[0-9]+虎扑币",num_gold_and_pro)
    if num_gold==[] and num_gold_pro==[]:
        num_gold=0
        num_gold_pro=0
    else:
        num_gold = int(num_gold[0].replace("个",""))
        num_gold_pro = int(num_gold_pro[0].replace("虎扑币",""))
   # print "asasdasdas",num_gold,num_gold_pro
    return [floor_num, post_time, num_light,num_gold,num_gold_pro, uid, cited_floor, cited_uid, cited_content, content, stat_date]

def getContentInAPost_nstPage(html,post_id):  # 提取帖子的一页中的内容，包括帖子标题，每个post的发布者名字、发布时间、内容。
    soup = BeautifulSoup(html)#, "html5lib")
    post_foll = soup.find_all(name="div", attrs={"id": re.compile("[0-9]+"), "class": "floor"})
    res = getFollPostInfor(post_foll)  # 获得第n页所有跟帖信息
    list_foll_posts = []
    if res != -1:
        list_foll_posts = list(map(lambda x: [post_id, 0] + x, res))  # 把主贴id加上，一边以后链表查询;第二位是亮帖标志位
        sql_str = ""
        sql_str += "insert into hupu_bxj_foll_posts_1(post_id,if_lighted,floor_num,post_time,num_light,num_gold,num_gold_pro,uid,cited_floor,cited_uid,cited_content,content,stat_date) "
        sql_str += " values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"  # %tuple(list_light_posts)
        list_foll_posts = list(map(lambda x: tuple(x), list_foll_posts))
    list_foll_posts = {'type': 'follPost', 'data':list_foll_posts}
    return list_foll_posts