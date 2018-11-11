#endocing:utf-8
from bs4 import BeautifulSoup
import re,requests,json
import urllib, bs4
import pyhanlp

from pyhanlp import HanLP
#获取文本的分词结果和词性标注结果
def wordSeg(text):
    text = text.replace('\n', '').replace('\r', '').replace(' ', '')
    wordPostag = HanLP.segment(text)
    words, postags = [], []
    for line in wordPostag:
        line = str(line)
        res = line.split('/')
        if len(res)!=2:
            continue
        word, postag = line.split('/')
        words.append(word)
        postags.append(postag)
    return words, postags


def get_product_url(url):
    # global pid
    # global links
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36')
    req.add_header("GET", url)
    content = urllib.request.urlopen(req).read()
    soup = bs4.BeautifulSoup(content, "lxml")
    product_id = soup.select('.gl-item')
    # links = []
    pid = []
    for i in range(len(product_id)):
        lin = "https://item.jd.com/" + str(product_id[i].get('data-sku')) + ".html"
        print(lin)
        # 获取链接
        # links.append(lin)
        # 获取id
        pid.append(product_id[i].get('data-sku'))
    return pid

def getCommentList(pid):
    s = requests.session()
    url = 'https://club.jd.com/comment/productPageComments.action'
    data = {'callback':'fetchJSON_comment98vv61',  'productId': pid,
               'score':0,  'sortType':5,  'pageSize':50,  'isShadowSku':0,
               'page':1}
    negCommetList = []
    posCommentList = []
    while True:
        try:
            t = s.get(url, params=data).text
            t = re.search(r'(?<=fetchJSON_comment98vv61\().*(?=\);)',t).group(0)
        except Exception as e:
            break
        j = json.loads(t)
        commentSummary = j.get('comments', [])
        for comment in commentSummary:
            c_content = comment['content']
            if '未填写评价内容' in c_content:
                continue
            c_score = comment['score']
            if c_score>4 and len(posCommentList) < len(negCommetList):
                words, _ = wordSeg(c_content)
                c_content = ' '.join(words)
                posCommentList.append(pid + 'kabukabu' +
                                  c_content + '\n')
            if c_score<2:
                words, _ = wordSeg(c_content)
                c_content = ' '.join(words)
                # print(comment.keys())
                # print('{}  {}  {}   {}\n{}\n'.format(c_name,c_time,c_client,c_score,c_content))
                # print(pid,data['page'],  c_content)
                negCommetList.append(pid + 'kabukabu' +
                                  c_content + '\n')
        data['page'] += 1
    return negCommetList, posCommentList

def writeLines(lines, fileName):
    with open(fileName, 'a+') as f:
        f.writelines(lines)

def processProducrName(name):
    res = str(name.encode('utf8'))
    res = res.replace('x', '').replace('b\'', '').replace('\'', '').upper().replace('\\', '%')
    return res

import time
if __name__ == '__main__':
    productNameList = ['手机', '篮球', '电脑', '水杯', '显示器', '打印机', '面包'
        , '桌子', '干果', '牛肉', '手柄', '钢琴', '音响', '键盘'
        , '手表', '洗衣液', '鼠标', '纸巾', '被子', '凉席', '冰箱'
        , '拖鞋', '内衣', '上衣', '花', '奶粉', '饼干', '老干妈']
    s = u'奶粉 婴幼奶粉 孕妈奶粉 营养辅食 益生菌 初乳 米粉 菜粉 果泥 果汁 DHA 宝宝零食 钙铁锌 维生素 清火 开胃 面条 粥 尿裤 湿巾 婴儿尿裤 拉拉裤 婴儿湿巾 成人尿裤 喂养用品 奶瓶奶嘴 吸奶器 暖奶消毒 儿童餐具 水壶 水杯 牙胶安抚 围兜 防溅衣 辅食料理机 食物存储 洗护用品 宝宝护肤 洗发沐浴 奶瓶清洗 驱蚊防晒 理发器 洗澡用具 婴儿口腔清洁 洗衣液 皂 日常护理 座便器 童车童床婴儿推车 餐椅摇椅 婴儿床 学步车 三轮车 自行车 电动车 扭扭车 滑板车 婴儿床垫 寝居服饰婴儿外出服 婴儿内衣 婴儿礼盒 婴儿鞋帽袜 安全防护 家居床品 睡袋 抱被 爬行垫 妈妈专区 妈咪包 背婴带 产后塑身 文胸 内裤 防辐射服 孕妈装 孕期营养 孕妇护肤 待产护理 月子装 防溢乳垫 童装童鞋套装 上衣 裤子 裙子 内衣 家居服 羽绒服 棉服 亲子装 儿童配饰 礼服 演出服 运动鞋 皮鞋 帆布鞋 靴子 凉鞋 功能鞋 户外'
    s = u'运动服 安全座椅 增高垫 平板电视 空调 冰箱 洗衣机 家庭影院 DVD/电视盒子 迷你音响 冷柜 冰吧 家电配件 功放 回音壁 Soundbar Hi-Fi 电视盒子 酒柜 厨卫大电燃气灶 油烟机 热水器 消毒柜 洗碗机 厨房小电料理机 榨汁机 电饭煲 电压力锅 豆浆机 咖啡机 微波炉 电烤箱 电磁炉 面包机 煮蛋器 酸奶机 电炖锅 电水壶/热水瓶 电饼铛 多用途锅 电烧烤炉 果蔬解毒机 其它厨房电器 养生壶 煎药壶 电热饭盒 生活电器取暖电器 净化器 加湿器 扫地机器人 吸尘器 挂烫机 熨斗 插座 电话机 清洁机 除湿机 干衣机 收录音机 电风扇 冷风扇 其它生活电器 生活电器配件 净水器 饮水机 个护健康剃须刀 脱毛器 口腔护理 电吹风 美容器 理发器 卷发器 按摩椅 按摩器 足浴盆 血压计 电子秤 厨房秤 血糖仪 体温计 健康电器 计步器 脂肪检测仪 五金家装电动工具 手动工具 仪器仪表 浴霸 排气扇 灯具 LED灯 洁身器 水槽 龙头 淋浴花洒 厨卫五金 家具五金 门铃 电气开关 插座 电工电料 监控安防 电线 线缆'
    productNameList = s.split(' ')
    negCommentFile = r'C:\Users\Administrator\Desktop\negComments.txt'
    posCommentFile = r'C:\Users\Administrator\Desktop\posComments.txt'
    count = 0
    countC = 0
    for name in productNameList:
        for page in range(1, 100, 2):
            url = u'https://search.jd.com/Search?keyword=' + processProducrName(name) + '&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&wq=' + processProducrName(name) + '&page=' + str(page) + '&s=58&click=0'
            print(url)
            pidList = get_product_url(url)
            for pid in pidList:
                negCommentList, posCommentList = getCommentList(pid)
                try:
                    count += 1
                    countC += len(negCommentList)
                    writeLines(negCommentList, negCommentFile)
                    writeLines(posCommentList, posCommentFile)
                    print(name, "这是第", count, "个产品。", countC)
                except:
                    pass
            time.sleep(1)