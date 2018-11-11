from tools import myLogin, getConnectionMongo
from multiprocessing import Queue,Process
from bloom_filter import BloomFilter
import  multiprocessing

opener = myLogin()
mongoCollection = getConnectionMongo()
UID_QUEUE_PERSON = Queue()


POST_ID_QUEUE = Queue()
BLOOMFILTER = BloomFilter(max_elements=500000000,error_rate=0.0001)
BLOOMFILTER_POST = BloomFilter(max_elements=500000000,error_rate=0.0001)
taskQueue = Queue()
