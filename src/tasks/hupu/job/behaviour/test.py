#用关联规则分析用户混迹的板块
import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
path_src = path.split('src')[0]
sys.path.append(path_src + "src")#项目的根目录
from pyhanlp import HanLP
from pyhanlp import *
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import time
from pymongo import MongoClient
#from analysis.algorithm import splitSentence, nlp
import splitSentence, nlp
from pyspark import SparkContext, SparkConf
import runTime
import re