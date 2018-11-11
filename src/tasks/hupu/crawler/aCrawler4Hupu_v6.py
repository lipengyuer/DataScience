# _*_coding:utf-8_*_
import urllib
#import urllib2
import os
import re
import time
#import cookielib
from  http import cookiejar
from  urllib import request
from bs4 import BeautifulSoup
import pymysql
import runTime


conn = pymysql.connect(host='192.168.1.199', user="root", passwd='1q2w3e4r', db='test', port=3306, use_unicode=True,charset="utf8")
cur = conn.cursor()
cur.execute("SET NAMES 'utf8mb4'")
conn.commit()

# sql查询，不返回数据
def sql_query_no_data(sql_str, data):
    # print data
    cur.execute(sql_str, data)
    conn.commit()

def sql_insert_many(sql_str, data_list):
    # print data_list
    # print len(data_list[0])
    cur.executemany(sql_str, data_list)
    conn.commit()

def myLogin():
    # 设置保存cookie的文件，同级目录下的cookie.txt
    filename = r'C:\Users\Administrator\PycharmProjects\test\hupu\crawler\cookies.txt'    # cookie位置，这里使用的是火狐浏览器的cookie
    # 声明一个MozillaCookieJar对象实例来保存cookie，之后写入文件
    cookie = cookiejar.MozillaCookieJar()
    cookie.load(filename, ignore_discard=True, ignore_expires=True)
    # 利用urllib2库的HTTPCookieProcessor对象来创建cookie处理器
    handler = request.HTTPCookieProcessor(cookie)
    # 通过handler来构建opener
    opener = request.build_opener(handler)
    return opener

def writeitintoafile(line, path):  # 将字符串追加到文件里
    f = open(path, 'a+')
    line = str(line)
    line = line.replace('\n', '')
    line = line.replace('\r', '')
    f.write(line + '\n')
    f.close()

def writeListIntoAFile(mylist, path):
    for line in mylist:
        writeitintoafile(line, path)

def getAPage(opener, myurl):  #
    response = opener.open(myurl, timeout=30)
    content = response.read()  # .decode('gbk').encode('utf-8')
    # content = content.decode(chardet.detect(content)["encoding"]).encode("utf-8")
    return content

def getPostURLSInAPage(opener, url_page):  # 获取一页里所有帖子的url
    #print ("zxc")
    response = opener.open(url_page, timeout=30)
    content = response.read()
    # html = str(content)
    #print type(content)
    soup = BeautifulSoup(content)#, "html5lib")
    lis = soup.find_all(name="li")
    list_urls = []
    for li in lis:
        theurl = li.find(name="a")
        if theurl != None:
            theurl = theurl.attrs["href"]
            if "html" in theurl and "https" not in theurl:
                list_urls.append("https://bbs.hupu.com/" + theurl)
    # print len(list_urls)
    return list_urls


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
    # uid = re.findall("[0-9]+",str(apost.find(name="a",attrs={"class":"u"}).attrs["href"]))[0]
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

def getFollPostInfor(foll_posts):
    if foll_posts == []:
        return -1
    else:
        # print lighted_posts
        list_posts = []
        for apost in foll_posts:
            list_posts.append(getPostInfor_foll(apost))
    return list_posts

def getContentInAPost_1stPage(html, aUrl):  # 提取帖子的一页中的内容，包括帖子标题，每个post的发布者名字、发布时间、内容。
    soup = BeautifulSoup(html)#, "html5lib")
    lis = re.findall("pageCount:[0-9]+",str(html))#<a herf=\"[0-9]+[-]+[0-9]+[.]html\"
    # 帖子页数
    if len(lis) == 0:
        N = 1
    else:
        N=int(lis[0].replace('pageCount:',""))
    # 回复，亮了的跟帖，浏览
    try:  # 有的帖子被删
        bbs_head = soup.find(name="div", attrs={"class": "bbs_head"}).find(name="div",
                                                                           attrs={"class": "bbs-hd-h1"}).find(
            name="span", attrs={"class": "browse"})
    except:
        return -1
    #print "aa"
    bbs_head = str(bbs_head)
    temp = re.findall("[0-9]+回复", bbs_head)
    if temp != []:
        num_reply = int(temp[0].replace("回复", ""))
    else:
        num_reply = 0
    temp = re.findall("[0-9]+亮", bbs_head)
    #print "bb"
    if temp != []:
        num_light_reposts = int(temp[0].replace("亮", ""))
    else:
        num_light_reposts = 0
    temp = re.findall("[0-9]+浏览", bbs_head)
    #print "cc"
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
    ori_context = str(ori_context).replace("\n", "").replace("<div class=\"quote-content\">", "").replace("</br></div>",                                                                                                  "")
    # 推荐数
    try:
        num_rec = ori_post.find(name="div", attrs={"class": "floor_box"}).find(name="span",
                                                                               attrs={"id": "Recers"}).find(name="a")
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
    post_time = ori_post.find(name="div", attrs={"class": "floor_box"}).find(name="span", attrs={"class": "stime"})
    post_time = str(post_time).replace("<span class=\"stime\">", "").replace("</span>", "")
    stat_date = post_time.split(" ")[0]  # 发帖日期，用于hive表分区
    timeArray = time.strptime(post_time, "%Y-%m-%d %H:%M")
    post_time = int(time.mktime(timeArray))

    #板块
    theBlock = str(ori_post.find(name="div", attrs={"class": "floor_box"}).
          find(name="div", attrs={"class": "subhead"}))
    theBlock = theBlock.split('</a>')
    mainBlock = theBlock[0].split('>')[-1]
    detailBlock = theBlock[1].split('>')[-1]
    #print(mainBlock,detailBlock )
    # 题目
    tittle = str(
        ori_post.find(name="div", attrs={"class": "floor_box"}).find(name="div", attrs={"class": "subhead"}).find(
            name="span")).replace("<span>", "").replace("</span>", "")
    post_id = aUrl.replace("https://bbs.hupu.com/", "").replace(".html", "")

    # 主贴信息
    res_ori = [post_id, tittle,mainBlock,detailBlock,  post_time, num_gold, num_gold_pro, num_rec, ori_context, uid, uname, num_reply,
               num_light_reposts, num_brows, stat_date]
    sql_str = ""
    sql_str += "insert into hupu_bxj_advocate_posts_1(post_id,title,main_block, detail_block,post_time,num_gold,num_gold_pro,num_rec,ori_context,uid,uname,num_reply,num_light_reposts,num_brows,stat_date) "
    sql_str += " values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    #print(sql_str)
    sql_query_no_data(sql_str, tuple(res_ori))
    # 亮帖信息
    list_light_posts = soup.find_all(name="div", attrs={"id": "readfloor"})
   # print(list_light_posts)
    list_light_posts = getLightedPostInfor(list_light_posts)  # 获得所有亮帖信息
    if list_light_posts != -1:
        list_light_posts = list(
            map(lambda x: [post_id, 1, 0] + x, list_light_posts))  # 把主贴id加上，一边以后链表查询;第二位是亮帖标志位;第三位是楼层数，亮帖的楼层数为0.
        sql_str = ""
        sql_str += "insert into hupu_bxj_foll_posts_1(post_id,if_lighted,floor_num,post_time,num_light,num_gold,num_gold_pro,uid,cited_floor,cited_uid,cited_content,content,stat_date) "
        sql_str += " values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"  # %tuple(list_light_posts)
        list_light_posts = list(map(lambda x: tuple(x), list_light_posts))
        sql_insert_many(sql_str, list_light_posts)

    # 跟帖信息
    post_foll = soup.find_all(name="div", attrs={"id": re.compile("[0-9]+"), "class": "floor"})
   # print(list_light_posts)
    list_foll_posts_I = getFollPostInfor(post_foll[len(list_light_posts):])  # 获得第一页所有跟帖信息
    if list_foll_posts_I != -1:
        list_foll_posts = list(map(lambda x: [post_id, 0] + x, list_foll_posts_I))  # 把主贴id加上，一边以后链表查询;第二位是亮帖标志位
        sql_str = ""
        sql_str += "insert into hupu_bxj_foll_posts_1(post_id,if_lighted,floor_num,post_time,num_light,num_gold,num_gold_pro,uid,cited_floor,cited_uid,cited_content,content,stat_date) "
        sql_str += " values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"  # %tuple(list_light_posts)

        list_foll_posts = list(map(lambda x: tuple(x), list_foll_posts))

        sql_insert_many(sql_str, list_foll_posts)
    else:
        print ("asd", list_foll_posts_I)
    return N

def getContentInAPost_nstPage(html,post_id):  # 提取帖子的一页中的内容，包括帖子标题，每个post的发布者名字、发布时间、内容。
    soup = BeautifulSoup(html)#, "html5lib")
    post_foll = soup.find_all(name="div", attrs={"id": re.compile("[0-9]+"), "class": "floor"})
    res = getFollPostInfor(post_foll)  # 获得第n页所有跟帖信息
    if res != -1:
        list_foll_posts = list(map(lambda x: [post_id, 0] + x, res))  # 把主贴id加上，一边以后链表查询;第二位是亮帖标志位
        sql_str = ""
        sql_str += "insert into hupu_bxj_foll_posts_1(post_id,if_lighted,floor_num,post_time,num_light,num_gold,num_gold_pro,uid,cited_floor,cited_uid,cited_content,content,stat_date) "
        sql_str += " values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"  # %tuple(list_light_posts)
        list_foll_posts = list(map(lambda x: tuple(x), list_foll_posts))
        print("正在向数据库写数据")
        sql_insert_many(sql_str, list_foll_posts)
    else:
        print ("asd", res)

def getInformationOfAPost(opener, aUrl):
    try:
        url_1st_page = aUrl + '.html'
        print('获取首页')
        contentOfTheFirstPage = getAPage(opener, url_1st_page)
        numOfPages = getContentInAPost_1stPage(contentOfTheFirstPage, url_1st_page)
        print('帖子的页数是', numOfPages)
        tid = re.findall('[0-9]+', aUrl)[0].replace('\n', '').replace('\r', '')
        tid = str(tid)
        if numOfPages != 1:
            for i in range(2, numOfPages + 1):
                aUrl_i = 'http://bbs.hupu.com/' + tid + '-' + str(i) + '.html'
                print(aUrl_i)
                contentOfThePage = getAPage(opener, aUrl_i)
                getContentInAPost_nstPage(contentOfThePage, tid)
    except:
        print("Failed!!!")

def worker():
    opener_I = myLogin()
    base_url = 'https://bbs.hupu.com/'  # 第一页
    print('队列长度是', (runTime.POST_ID_QUEUE).qsize())
    while (runTime.POST_ID_QUEUE).qsize() > 0:
        post_id = (runTime.POST_ID_QUEUE).get()
        aurl = base_url + str(post_id)
        # aurl = 'https://bbs.hupu.com/22175815'
        print(aurl)
        getInformationOfAPost(opener_I, aurl)
import threading

from multiprocessing import Queue,Process
#from multiprocessing.pool import Thread
import time
if __name__ == '__main__':
    uidList = range(2846000, 1500 * 10000)
    # uidList = [22849313]
    for i in uidList:
        if i%100000==0:
            print('正在初始化uid队列', i)
        runTime.POST_ID_QUEUE.put(i)

    for i in range(50):
        print('正在启动进程', i)
        p = threading.Thread(target=worker)
        p.start()
#C:\Users\Administrator\PycharmProjects\test\venv1\Scripts\python.exe C:\Users\Administrator\PycharmProjects\test\hupu\aCrawler4Hupu_soup_4_test.py

# mysql建表语句
'''
SET GLOBAL innodb_file_per_table=1;
#SET GLOBAL innodb_file_format=Barracuda;
create table hupu_bxj_foll_posts(
post_id char varying(100),
if_lighted int,
floor_num int,
post_time int,
num_light int,
num_gold int,
num_gold_pro int,
uid text,
cited_floor int,
cited_uid text,
cited_content LongText,
content LongText,
stat_date text
)
 ENGINE=InnoDB
 ROW_FORMAT=COMPRESSED
 KEY_BLOCK_SIZE=8;


create table hupu_bxj_advocate_posts(
post_id char varying(100),
title text,
main_block text,
detail_block text,
post_time int,
num_gold int,
num_gold_pro int,
num_rec int,
ori_context LongText,
uid text,
uname text,
num_reply int,
num_light_reposts int,
num_brows int,
stat_date text)
 ENGINE=InnoDB
 ROW_FORMAT=COMPRESSED
 KEY_BLOCK_SIZE=8;

alter table hupu_bxj_foll_posts convert to character set utf8mb4 collate utf8mb4_unicode_ci;
alter table hupu_bxj_advocate_posts convert to character set utf8mb4 collate utf8mb4_unicode_ci;
alter table hupu_bxj_advocate_posts add index myindex(post_id);
alter table hupu_bxj_foll_posts add index myindex(post_id);
'''