'''
Created on Jun 22, 2017

@author: foxbat
'''
import re

def filtCrap(content, crapLength):
    if len(content) > crapLength:
        content = re.sub("[^\u4e00-\u9fa5]{" + str(crapLength) + ",}", "", content)
    return content


def filtUrl(content):
    content = re.sub(
        "(((http|ftp|https):\/\/)|(www.))[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?", "", content)
    return content


def filtHtmlTags(content):
    # 去除文章内容中的html标签
    content = re.sub("\n", "7865256862525", content)
    content = re.sub("<script.*?>.*?</script>", "", content)
    content = re.sub("</?[^<]+>", "", content)
    content = re.sub("\\s*|\t|\r", "", content)
    content = re.sub("&[a-z]{1,6};", "", content)
    content = re.sub("&#[0-9]*;", "", content)
    content = re.sub("(7865256862525)+", "\n", content)
    return content

# 向英文半角转换
notStandardEnglish = {'Ｇ': 'G', 'ｉ': 'i', 'Ｑ': 'Q', 'Ｍ': 'M', 'ｍ': 'm', 'Ｖ': 'V', 'Ｂ': 'B',
                      'ｇ': 'g', 'Ｈ': 'H', 'ｌ': 'l', 'ｖ': 'v', 'Ｔ': 'T', 'ａ': 'a', 'ｂ': 'b',
                      'ｄ': 'd', 'ｒ': 'r', 'Ｃ': 'C', 'Ｉ': 'I', 'ｋ': 'k', 'ｅ': 'e', 'Ｐ': 'P',
                      'Ｗ': 'W', 'Ｒ': 'R', 'Ｅ': 'E', 'Ｊ': 'J', 'Ｋ': 'K', 'ｘ': 'x', 'Ｏ': 'O',
                      'Ｚ': 'Z', 'ｑ': 'q', 'ｐ': 'p', 'Ｕ': 'U', 'ｚ': 'z', 'ｗ': 'w', 'ｓ': 's',
                      'ｆ': 'f', 'ｙ': 'y', 'ｃ': 'c', 'Ａ': 'A', 'Ｙ': 'Y', 'ｊ': 'j', 'ｏ': 'o',
                      'Ｓ': 'S', 'Ｘ': 'X', 'ｕ': 'u', 'Ｄ': 'D', 'Ｌ': 'L', 'ｈ': 'h', 'ｎ': 'n',
                      'ｔ': 't', 'Ｆ': 'F', 'Ｎ': 'N', 'ⅰ': 'i', 'А': 'A', 'ⅴ': 'V',
                      'М': 'M', 'Т': 'T'}

standardEnglish = {'Q', 't', 'B', 'A', 'T', 'V', 'f', 'b', 'd', 'c', 's', 'I', 'D', 'Y', 'R', 'U',
                   'v', 'y', 'z', 'L', 'l', 'p', 'e', 'S', 'K', 'C', 'F', 'X', 'E', 'O', 'G', 'J',
                   'w', 'j', 'g', 'Z', 'n', 'o', 'i', 'W', 'm', 'a', 'r', 'h', 'q', 'H', 'M', 'k',
                   'u', 'N', 'P', 'x'}


notStandardNumber = {'０': '0', '８': '8', '３': '3', '７': '7', '６': '6', '９': '9', '４': '4',
                     '１': '1', '２': '2', '５': '5'}
standardNumber = {'3', '5', '0', '2', '1', '8', '7', '9', '6', '4'}

notStandardInvisibleChar = {'　': ' '}
standardInvisibleChar = {'\n'}#, ' '


# 向中文全角转换
notStandardPunctuation = {'"': '”', '＂': '”', '″': '”', ':': '：', '︰': '：', '!': '！', '{': '｛', ';': '；', '}': '｝', '?': '？',
                          '【': '（', '】': '）', '(': '（', ',': '，', ')': '）', "'": '’', '′': '’', '＇': '’', 'ˋ': '’', '.': '．', '﹔': '；', '∶': '：', '﹐': '，', '﹑': '、', '－': '-', '–': '-', '―': '—', '-': '－', '─': '—', '＠': '@', '〝': '“', '〞': '”', '﹕': '：', '﹒': '·', '﹗': '！', '﹖': '？', '﹪': '％', '%': '％'}

standardPunctuation = {'“', '”', '〈', '〉', '《',  '》', '（', '）', '‘', '’',
                       '、', '…', '—', '；', '。', '：', '？', '，', '！', '·', '．', '～', '＜', '＞', '％','/','[',']','@'}

notStandardMultiPunctuation = {
    '......': '……', '――': '——', '—―': '——', '──': '——'}
standardMultiPunctuation = {'……', '——'}

#   代表字母和数字的特殊符号 α #


def filtBadChar(content):
    for key in notStandardMultiPunctuation:
        content = content.replace(key, notStandardMultiPunctuation[key])

    charList = list(content)
    for i in range(len(charList)):
        if charList[i] >= '\u4e00' and charList[i] <= '\u9fa5':
            continue
        elif charList[i] in notStandardEnglish:
            charList[i] = notStandardEnglish[charList[i]]
        elif charList[i] in standardEnglish:
            continue
        elif charList[i] in notStandardNumber:
            charList[i] = notStandardNumber[charList[i]]
        elif charList[i] in standardNumber:
            continue
        elif charList[i] in notStandardPunctuation:
            charList[i] = notStandardPunctuation[charList[i]]
        elif charList[i] in standardPunctuation:
            continue
        elif charList[i] in notStandardInvisibleChar:
            charList[i] = notStandardInvisibleChar[charList[i]]
        elif charList[i] in standardInvisibleChar:
            continue
        else:
            charList[i] = ''

    content = ''.join(charList)
    return content


def filtDeclaraction(content):
    if content.find('参考消息网') > -1:
        content = content.replace(
            '凡注明“来源：参考消息网”的所有作品，未经本网授权，不得转载、摘编或以其他方式使用。', '')
        content = content.replace(
            '本文系转载，不代表参考消息网的观点。参考消息网对其文字、图片与其他内容的真实性、及时性、完整性和准确性以及其权利属性均不作任何保证和承诺，请读者和相关方自行核实。', '')
    return content


def stripEHead(content):
    index4 = content.rfind('报道')
    index4 = 0 if index4 == -1 else index4 + 2
    index5 = content.rfind('本报')
    index5 = 0 if index5 == -1 else index5 + 2
    index7 = content.rfind('通讯员')
    index7 = 0 if index7 == -1 else index7 + 3
    index9 = content.rfind('日电')
    index9 = 0 if index9 == -1 else index9 + 2
    index10 = content.rfind('讯 ')
    index10 = 0 if index10 == -1 else index10 + 2
    index11 = content.rfind('报讯')
    index11 = 0 if index11 == -1 else index11 + 2
    pointer = max(index4, index5, index7, index9, index10, index11)
    return content[pointer:]


def stripETail(content):
    length = len(content)
    index4 = content.find('编辑')
    index4 = length if index4 == -1 else index4
    index7 = content.find('责任编辑')
    index7 = length if index7 == -1 else index7
    index8 = content.find('责编')
    index8 = length if index8 == -1 else index8
    pointer = min(index4, index7, index8)
    return content[:pointer]


def getCleanContent(content):
    paragraphs = content.replace("\r", "").split("\n")

    pointer = 0
    while pointer < len(paragraphs):
        paragraphs[pointer] = paragraphs[pointer].strip()
        if len(paragraphs[pointer]) == 0:
            del paragraphs[pointer]
        else:
            pointer += 1
    '''if len(paragraphs) > 0:
        paragraphs[0] = stripEHead(paragraphs[0])
        paragraphs[-1] = stripETail(paragraphs[-1])'''
    content = '\n'.join(paragraphs)
    return content
