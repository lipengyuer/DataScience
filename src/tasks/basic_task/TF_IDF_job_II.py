'''
Created on 2019年5月11日

@author: Administrator
'''
#针对大规模语料的tf-idf统计。
#语料较大的时候，无法全部加载到内存中，需要逐行读取。
#统计环节比较消耗算力，组好并行化。
#多个进程对多个文件块的统计结果需要合并为一份数据。这个任务只能是一个进程来完成了

import multiprocessing
import os

data_batch_size = 1000

def iter_files(result, rootDir):
    #遍历根目录
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            #print(file_name)
            result.append(file_name)
        for dirname in dirs:
            #递归调用自身,只改变目录名称
            iter_files(result, dirname)
    return result

def get_ngrams(line, N=2):
    if len(line)<=N: return []
    res = []
    for i in range(len(line)-N):
        res.append(line[i: i + N])
    return res
    
def file_reader(file_list, task_queue, process_id, data_batch_size = data_batch_size):

    data_batch = []
    count = 1
#     print(file_list)
    for file in file_list:
        
        if 'SUCCESS' in file: continue
        print(file)
        with open(file, 'r', encoding='utf8') as f:
            line = f.readline()
            while True:
                count += 1
                if len(line)<10: continue
                #print(process_id, count, len(line))
                if len(data_batch)==data_batch_size:
                    task_queue.put([data_batch, data_batch_size])
                    #print('task ', task_queue.qsize(), count)
                    data_batch = []
                if count%100000==0:
                    print('process_id', process_id, 'task count ', task_queue.qsize(), count)
                    #return 
                data_batch.append(line)
                line = f.readline()
                if len(line)<10: break
    print('finished reading.')          
                
def worker_tf_df(task_queue, result_queue):
    count = 0
    while True:
        #print("start stage1 ")
        [task_data_batch, batch_size] = task_queue.get()
        tf_map = {}
        df_map = {}
        #print("stage 1")
#         t1 = time.time()
        for line in task_data_batch:
            count += 1
            #if count%1000==0:
            ngrams = get_ngrams(line)
            ngram_set = set(ngrams)
            for word in ngrams:
                tf_map[word] = tf_map.get(word, 0) + 1
            for word in ngram_set:
                df_map[word] = df_map.get(word, 0) + 1
        #print("stage 1")
        #print(len(tf_map), len(df_map))
        result_queue.put([tf_map, df_map, batch_size])
#         t2 = time.time()
#         print('stage 1 ', t2-t1)

import time
import pickle

def worker_total_tf_df(task_queue):
    print("start stage 2")
    last_update_time = time.time()
    count = 0
    doc_num = 0
    tf_map = {}
    df_map = {}
    while True:
        try:
            [tf_map_batch, df_map_batch, batch_size] = task_queue.get_nowait()
            t1 = time.time()
            count += 1
            doc_num += batch_size
            #print('tf_map_batch', len(tf_map_batch), last_update_time)
            for word in tf_map_batch:
                tf_map[word] = tf_map.get(word, 0) + tf_map_batch[word]
            for word in df_map_batch:
                df_map[word] = df_map.get(word, 0) + df_map_batch[word]
            last_update_time = time.time()
        except: 
            if time.time()-last_update_time>20:
                break
            time.sleep(0.01)
        
        t2 = time.time()
#         print('stage 2 ', t2-t1)
    idf_map = {}
    for word in df_map:
        idf_map[word] = doc_num/df_map[word]
    pickle.dump({"tf": tf_map, 'idf': idf_map}, open('res.pkl', 'wb'))
    print("idf_map size", len(idf_map))
    return
   
    
if __name__ == '__main__':
    soure_data_dir = r'D:\backup\mydata\tieba_posts'
    task_queue = multiprocessing.Queue(200)
    result_queue = multiprocessing.Queue(200)
    #file_reader(soure_data_dir, task_queue)
    
    files = []
    iter_files(files, soure_data_dir)
    file_batch_size = int(len(files)/2)
    print('files', len(files), file_batch_size)
    id_list = list(range(0, len(files), file_batch_size))
    print('id_list', id_list)
    for i in range(len(id_list)-1):
        start, end = id_list[i], id_list[i+1]
        print('process',i, start, end )#files[start: end]
        p1 = multiprocessing.Process(target=file_reader, args=(files[start: end], task_queue, i))
        p1.start()
        print('asdasd')
    for i in range(6):
        p = multiprocessing.Process(target=worker_tf_df, args=(task_queue, result_queue))
        p.start()
         
    worker_total_tf_df(result_queue)
    import pickle
    data = pickle.load(open(r'C:\Users\Administrator\Desktop\res.pkl', 'rb'))
    print(len(data['idf']))
    good_words = list(data['idf'].items())
    print(good_words[:10])
    good_words = list(sorted(good_words, key=lambda x: x[1]))
    print(good_words[:10])
    good_words = list(map(lambda x: x[0] + ' ' + str(x[1]) + '\n', good_words))
    print(good_words[:10])
    with open('words_by_idf.txt', 'w', encoding='utf8') as f:
        f.writelines(good_words)
    
    
    