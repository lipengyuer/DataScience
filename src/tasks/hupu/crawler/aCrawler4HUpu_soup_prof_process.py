# _*_coding:utf-8_*_
import urllib
import os
import re
import time
from bs4 import BeautifulSoup
import pymysql
import multiprocessing
from multiprocessing import Pool
import time
from entity import personalInfo
import runTime


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

def getAPage(myurl):  #
    count = 0
    while True:
        try:
            if count == 3:
                break
            # print(myurl)
            response = runTime.opener.open(myurl, timeout=10)
            content = response.read()  # .decode('gbk').encode('utf-8')
            # print(content)
            return content
        except:
            print("网络有问题",myurl, time.time())
            # time.sleep(1)
            count += 1
    return 'nothing'

def getBasicInfo(Person, soup):
    mainContent = soup.find(name='div', attrs={'class': 'hp-wrap'}).find(name='div', attrs={'class': 'personal'})
    if mainContent is None:
        return None
    #用户名
    userName = mainContent.find(name='h3', attrs={'class':'mpersonal'})
    if userName!=None:
        userName = userName.find(
            name='div', attrs={'class': 'left'})
        if userName != None:
            Person.userName = userName.string
        else:
            return None
    else:
        return None
    #print('用户名是',userName,  mainContent.find(name='h3', attrs={'class':'mpersonal'}))

    #来访的用户数
    tempRes = mainContent.find(name='h3', attrs={'class':'mpersonal'}).find(
        name='span', attrs={'class': 'f666'})
    if tempRes!=None:
        Person.userNumCame = int(tempRes.string.replace('有','').replace('人次访问',''))

    #性别
    tempRes = mainContent.find(name='div', attrs={'class': 'personalinfo'}).find(
        name='span', attrs={'itemprop': 'gender'})
    if tempRes!=None:
        Person.gender = 'm' if tempRes.string=='男' else 'f'

    #所在地
    tempRes = mainContent.find(name='div', attrs={'class': 'personalinfo'}).find(
        name='span', attrs={'itemprop': 'address'})
    if tempRes!=None:
        Person.location = tempRes.string

    #其他信息
    mainContent = soup.find(name='div', attrs={'class': 'hp-wrap'}).find(
        name='div', attrs={'class': 'personalinfo'})
    s = str(mainContent).split('<br/>')

    tempRes = {}
    # print(s)
    for line in s[:-1]:
        # print('asd', line, 'asd')
        key = re.findall('>.+?<', line)[0].replace('>', '').replace('<', '')

        if "主队" in line:
            value = re.findall('href.+?</a>', line)[0].split('>')[1].replace('</a','')

        elif '所属社团' in line:
            value = re.findall('>.*?<',line)[-1].replace('>', '').replace('<', '')
        else:
            value = re.findall('span>.+\s', line)[0]. \
                replace('span>', '').replace('<', '').replace('小时', ''). \
                replace(' ', '').strip()
        #print(key,value)
        tempRes[key] = value

    Person.onlineTime = tempRes.get('在线时间：')
    Person.onlineTime = int(Person.onlineTime) if type(Person.onlineTime)==str else None
    Person.registerTime = tempRes.get('加入时间：')
    Person.HPLevel = tempRes.get('社区等级：')
    try:
        Person.HPLevel = int(Person.HPLevel) if type(Person.HPLevel) == str else None
    except:
        Person.HPLevel = None
    Person.theOrg = tempRes.get('所属社团：')
    Person.communityRPScore = tempRes.get('社区声望：')
    try:
        Person.communityRPScore = int(Person.communityRPScore) if type(Person.communityRPScore) == str else None
    except:
        Person.communityRPScore = None
    Person.dutiesHP = tempRes.get('社区职务：')
    Person.lastLoginTime = tempRes.get('上次登录：')
    Person.crawlTime = int(time.time())

    Person.homeTeams = {}
    for key in tempRes:
        if '主队' in key:
            Person.homeTeams[key] = tempRes[key]

def getDetailInfo(Person):
    detailURL = 'https://my.hupu.com/' + Person.uid + '/profile'
    detailURL = 'https://my.hupu.com/7656961171090/profile'
    soup = BeautifulSoup(getAPage(detailURL), "html5lib")
    contant = soup.find(name='div', attrs={'class': 'hp-wrap'})
    # print('详细信息页内容', contant)
    #在线时间
    table = contant.find(
        name='div', attrs={'class': 'content'})
    # print("详细信息表格",table )
    tempRes = table.find(
        name='td', attrs={'class': 'a1'})
    Person.onlineTime = tempRes


def getDigitUID(homePageURL):
    try:
        content = getAPage(homePageURL)
        soup = BeautifulSoup(content, "html5lib")
        uid = soup.find(name='img', attrs={'id': 'j_head'}).attrs['uid']
        return uid
    except:
        return None

def extractFollows(content):
    soup = BeautifulSoup(content, "html5lib")
    #print(soup)
    res = soup.find_all(name='div', attrs={'class': 'contact_item'})
    resList = []
    if len(res)>0:
        for line in res:
            followID = line.find(name='a', attrs={'rel': 'contact'}).attrs['href']
            followID = followID.split('/')[-1]
            flag = re.findall('[a-z]', followID)
            if flag!=[]:
                #从用户的主页去找数字id
                # print("这个id是字母的", followID, '需要到主页去找数字的。')
                followID = getDigitUID('https://my.hupu.com/' + followID)
            if followID is not None:
                resList.append(followID)
    return resList

def extractFans(content):
    soup = BeautifulSoup(content, "html5lib")
    #print(soup)
    res = soup.find_all(name='div', attrs={'class': 'contact_item'})
    resList = []
    if len(res)>0:
        for line in res:
            followID = line.find(name='a', attrs={'class': 'u'}).attrs['href']
            followID = followID.split('/')[-1]
            flag = re.findall('[a-z]', followID)
            # print("这个粉丝的ID是", followID)
            if flag!=[]:
                #从用户的主页去找数字id
                # print("这个id是字母的", followID, '需要到主页去找数字的。')
                followID = getDigitUID('https://my.hupu.com/' + followID)
            resList.append(followID)
    return resList

def getFollows(Person):
    url = 'https://my.hupu.com/' + Person.uid + '/following?&page='
    Person.follow = []
    pageNum = 1
    while True:
        #print(url+str(pageNum))
        content = getAPage(url+str(pageNum))
        followsInPage = extractFollows(content)
        # print(pageNum,)
        #print(followsInPage)
        if len(followsInPage) == 0:
            #print(pageNum)
            break
        Person.follow += followsInPage
        pageNum += 1

def getFans(Person):
    url = 'https://my.hupu.com/' + Person.uid + '/follower?page='
    Person.fans = []
    pageNum = 1
    while True:
        #print(url+str(pageNum))
        content = getAPage(url+str(pageNum))
        fansInPage = extractFans(content)
        #print(fansInPage)
        if len(fansInPage) == 0:
            break
        Person.fans += fansInPage
        pageNum += 1

def getInformationOfAUser(uid, mongoCollection):
        aurl = 'https://my.hupu.com/' + str(uid)
        print("正在获取主页信息")
        contentOfTheFirstPage = getAPage(aurl)
        print("完成获取主页信息")
        if contentOfTheFirstPage=='nothing':
            print("这是一个空网页")
            return {}
        mysoup = BeautifulSoup(contentOfTheFirstPage, "html5lib")
        Person = personalInfo()
        Person.uid = uid
        # print('正在获取主页基本信息')
        for _ in range(5):
            # try:
                getBasicInfo(Person, mysoup)#获取基本信息
                if getBasicInfo is None:
                    return
                print('正在获取关注信息',)
                getFollows(Person)
                print('正在获取粉丝信息')
                getFans(Person)
                print('正在组织数据')
                result = Person.trans2Json()
                print("正在向数据库写数据", uid)
                mongoCollection.insert(result)
                break
            # except:
            #     print('获取用户', uid, '的数据失败， 这是第', _, '次尝试。')


from dbconnection.getMySQL import getConnection
def getuids():
    t1 = time.time()
    mysqlcon = getConnection()
    mysqlcon.setConnecter(host='192.168.1.199', port=3306, user="root",
                          passwd='1q2w3e4r', db='test',
                          use_unicode=True, charset="utf8")

    sqlstr = ""
    sqlstr += 'select uid '
    sqlstr += 'from hupu_bxj_advocate_posts_1 '
    # sqlstr += ' limit 20000'
    sqlstr += ' where post_time>=1104508800 and post_time<=1262275200'
    data1 = mysqlcon.queryWithReturn(sqlstr)
    sqlstr = ""
    sqlstr += 'select uid '
    sqlstr += ' from hupu_bxj_foll_posts_1'
    sqlstr += ' where post_time>=1104508800 and post_time<=1262275200'
    # sqlstr += ' limit 20000'
    data2 = mysqlcon.queryWithReturn(sqlstr)
    data = data1 + data2
    data = map(lambda x: x[0], data)
    data = list(data)
    return set(data)

def worker(q):
    from tools import myLogin, getConnectionMongo
    mongoCollection = getConnectionMongo()
    print("uid队列现在的长度是", q.qsize())
    while (q).qsize()>0:
        # print((runTime.UID_QUEUE).qsize())
        uid = (q).get()
        print("正在处理的用户是",uid )
        getInformationOfAUser(uid, mongoCollection)
    return

def  initBloomFilter():
    for line in runTime.mongoCollection.find({}, {'_id':1}):
        #print("布隆过滤器", line)
        runTime.BLOOMFILTER.add(line['_id'])

def getUIDsFromMySQL():
    uids = list(getuids())
    lines = map(lambda x: x + '\n', uids)
    lines = list(lines)
    with open('uids_in_this_batch.txt', "w") as f:
        f.writelines(lines)
    return uids
def getUIDsFromFile():
    with open(r'C:\Users\Administrator\PycharmProjects\test\venv1\Scripts\uids_in_this_batch.txt', "r") as f:
        lines = f.readlines()
    lines = map(lambda x: x.replace('\n', ''), lines)
    uids = list(lines)
    return uids
import threading
from multiprocessing import Queue,Process
if __name__ == '__main__':
    # uids = getUIDsFromMySQL()
    uids = getUIDsFromFile()
    # uids = ['139715637919000']
    print("正在初始化布隆过滤器。")
    initBloomFilter()
    print("正在积累uid队列")
    for uid in uids:
        if uid not in runTime.BLOOMFILTER:
            # print(uid, runTime.UID_QUEUE_PERSON.qsize())
            runTime.UID_QUEUE_PERSON.put(uid)
    print('uid队列的初始长度是', runTime.UID_QUEUE_PERSON.qsize())
    print('开始获取数据')
    for i in range(0, 20):
        p = Process(target=worker, args=(runTime.UID_QUEUE_PERSON, ))
        # p = threading.Thread(target=worker)
        p.start()

