import os, sys
path = os.getcwd() #获取当前目录的路径
sys.path.append(path)

from scrapy.cmdline import execute

if __name__ == '__main__':
    execute('scrapy crawl hupu --nolog'.split())
