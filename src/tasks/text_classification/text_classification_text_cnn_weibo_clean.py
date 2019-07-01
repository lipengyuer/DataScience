'''
Created on 2019年6月7日

@author: Administrator
用cnn实现一个文本分类器
'''
import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
import tensorflow as tf
import utils
import json, pickle
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import preProcess

stop_chars = {"的"}
class TextCNN():
    
    def __init__(self, with_pretrained_char_embedding=False, static_embedding=True, pretrained_model=None):
        self.mode = 'train'
        self.load_parameters()
        self.load_char_id_data()
        
        self.sess = tf.Session()
        if with_pretrained_char_embedding==False:
            self.char_embedding = tf.Variable(tf.random_normal([self.parameters['char_set_size'], \
                                                            self.parameters['embedding_size']], 
                                                               mean=0, stddev=0.01), name='embedding_init')
        else:
            self.load_pretrained_char_embedding(static_embedding)

        self.load_graph()
        if pretrained_model!=None:
            tf.reset_default_graph()
            self.saver.restore(self.sess, pretrained_model)
#             saver = tf.train.import_meta_graph(pretrained_model + '.meta')
#             graph = tf.get_default_graph()
#             print(graph._nodes_by_name.keys())
#             print(graph.get_tensor_by_name('Variable_10'))
    
    #
    def load_pretrained_char_embedding(self, static_embedding):
        #char_vector_from_gensim = Word2Vec.load("./model/word2vec.model")
        char_vector_list = list(open('./a_vec_sample/vec.txt', 'r', encoding='utf8'))[1:]
        char_vector_list = list(map(lambda x: x.replace('\n', '').split(' '), char_vector_list))
        char_vector_list = list(map(lambda x: [x[0], list(map(lambda y: float(y), x[1:]))],\
                                     char_vector_list))
        char_vector_from_gensim = {}
        for line in char_vector_list:
            char_vector_from_gensim[line[0]] = line[1]
            
        char_embedding = []
        for char_id in range(len(self.char_id_map)):
            char = self.id_char_map[char_id]
            if char in char_vector_from_gensim:
                char_vec = char_vector_from_gensim[char]
            else:
                char_vec = np.zeros(self.parameters['embedding_size'])
            char_embedding.append(char_vec)
        char_embedding = np.array(char_embedding)
        if static_embedding:
            self.char_embedding = tf.constant(char_embedding,
                                            dtype=tf.float32, shape=None, name='Const', verify_shape=False)
        else:
            self.char_embedding = tf.get_variable(name='char_embedding', shape=char_embedding.shape, \
                                        initializer=tf.constant_initializer(char_embedding))            
        
        
    def load_graph(self):
        node_num  = 50
        self.X = tf.placeholder(dtype=tf.int32, shape=[None, self.parameters['max_text_length']], name='id_list')
#         if self.mode=='train':
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.parameters['class_num']],name='output')
        X_char_vec = tf.nn.embedding_lookup(self.char_embedding, self.X)
        X_char_vec4cnn = tf.expand_dims(X_char_vec, -1)
        
        kernel_size_list = [1,2,3,4,5,6,7]
        flatten_out_list = []
        for kernel_size in kernel_size_list:
            self.filter_1_w = tf.Variable(tf.random_normal([\
                                kernel_size, self.parameters['embedding_size'], 1, node_num], mean=0, stddev=0.05),name='filter_1_w')
            self.filter_1_b = tf.Variable(tf.random_normal([node_num], mean=0, stddev=0.01), name='filter_1_b')
            conv_1 = tf.nn.conv2d(X_char_vec4cnn, self.filter_1_w, strides=[1, 1, 1, 1], padding='VALID')
            hidden_out_1 = tf.nn.relu6(tf.nn.bias_add(conv_1, self.filter_1_b))
            pooled_out_1 = tf.nn.max_pool(hidden_out_1, \
                                          ksize=[1, self.parameters['max_text_length'] - kernel_size + 1, 1, 1],\
                                          strides = [1, 1, 1, 1], padding='VALID')
            print("pooled_out_1 size is ", pooled_out_1)
            flatten_out = tf.reshape(pooled_out_1, [-1, node_num])
            flatten_out_list.append(flatten_out)
        #拼接
        flatten_out = tf.concat(flatten_out_list, axis=1)
        #拉直
        #dropout
        flatten_out_droped = tf.nn.dropout(flatten_out, 0.6)
        #print("flatten_out_droped 的形状是" , flatten_out_droped)
        #softmax
        
        self.softmax_w = tf.Variable(tf.random_normal([node_num*len(flatten_out_list),\
                            self.parameters['class_num']], mean=0, stddev=0.1),\
                                      name='softmax_w')
        self.softmax_b = tf.Variable(tf.random_normal([self.parameters['class_num']], \
                                    mean=0, stddev=0.1),\
                                      name="softmax_bias")
        self.prob_dist_1 = tf.nn.softmax(tf.matmul(flatten_out_droped, self.softmax_w) + self.softmax_b)
#         print(self.softmax_b)
        
        self.prob_dist  = tf.layers.dense(self.prob_dist_1, self.parameters['class_num'])
        
        self.predictions = tf.argmax(self.prob_dist, 1, name="predictions")
        
        self.prob_dist_ = tf.reshape(self.prob_dist, [-1,1])
        self.losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y,logits=self.prob_dist))
        tf.summary.scalar('loss', self.losses) 

        correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy) 
        self.train = tf.train.AdagradOptimizer(0.01).minimize(self.losses)
#         self.train = tf.train.AdadeltaOptimizer(0.1).minimize(self.losses)
#         self.train = tf.train.GradientDescentOptimizer(0.01).minimize(self.losses)

        self.sess.run(tf.global_variables_initializer())
        self.merged=tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("log",self.sess.graph)
#         self.merged = tf.summary.merge_all() #将图形、训练过程等数据合并在一起
#         self.writer = tf.summary.FileWriter('logs',self.sess.graph) #将训练日志写入到logs文件夹下
        self.saver = tf.train.Saver()
    def set_mode(self, mode):
        self.mode = mode
    def load_parameters(self):
        self.parameters = json.load(open('parameters.json', 'r', encoding='utf8'))
    
    def load_char_id_data(self):
        self.char_id_map = pickle.load(open(self.parameters['char_id_map_file'], 'rb'))
        self.id_char_map = pickle.load(open(self.parameters['id_char_map_file'], 'rb'))
        class_label_id_map = pickle.load(open(self.parameters['class_label_id_map_file'], 'rb'))
        one_hot = OneHotEncoder()
        class_label_list = list(map(lambda x: [x], class_label_id_map.keys()))
        one_hot.fit(class_label_list)
        class_label_one_hot = one_hot.transform(class_label_list).toarray()
        self.class_label_one_hot = {}
        for i in range(len(class_label_list)):
            class_label = class_label_list[i][0]
            one_hot_label = class_label_one_hot[i]
            self.class_label_one_hot[class_label] = one_hot_label
    #直接读取训练语料，并转换为id,然后转换为字向量。默认是静态字向量；可以选择微调
    def fit(self, if_static_embeding=True):
        file_list = utils.find_all_files(self.parameters['train_corpus_dir'], [])
        test_input, test_output = self.load_test_data()
        count = 0
        batch_size = 50
        
        

        for epoch in range(10000):
            x_batch = []
            y_batch = []
            lines = []
            for file_name in file_list:
                lines += utils.read_lines_small_file(file_name)
            random.shuffle(lines)
            for text_file in lines:
                [text,file_name] = text_file
                text = preProcess.filtUrl(text)
                text = text.replace(" ", '').replace('\n', '')[:self.parameters['max_text_length']]
                id_list = self.trans_char2id(text)
             
                class_label = file_name.split('/')[-2]
                class_label_one_hot = self.class_label_one_hot[class_label]
                count += 1
                #print(class_label, text)
                x_batch.append(id_list)
    #             print(x_batch)
                y_batch.append(class_label_one_hot)
                #print(len(y_batch), y_batch)
            #打乱顺序
#                 print(x_batch)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            #print(y_batch)
            
#                 print(x_batch.shape)
            #训练
            for i in range(0, y_batch.shape[0], batch_size):
                a_x_batch = x_batch[i:i+batch_size,:]
                a_y_batch = y_batch[i:i+batch_size,:]
                #print(a_y_batch)
                Y, prob_dist, _, loss_1, accuracy = self.sess.run([self.Y, self.prob_dist, self.train, self.losses, self.accuracy],\
                                              feed_dict={self.X: a_x_batch, self.Y: a_y_batch})
            #打印损失值
#             print('epoch ', epoch," loss is ", loss, '。 accuracy is ', accuracy)
            loss, accuracy = self.sess.run([self.losses, self.accuracy],\
                          feed_dict={self.X: test_input, self.Y: test_output})
            merg = self.sess.run(self.merged,\
                          feed_dict={self.X: test_input, self.Y: test_output})
            
            self.writer.add_summary(merg, epoch)
            print(epoch, loss_1, "在测试集中的loss为", loss, 'accuracy为', accuracy)
            self.saver.save(self.sess, self.parameters['check_points_dir'] + '/model')
            tf.reset_default_graph()
        self.writer.close()
    
    def load_test_data(self):
        x_batch = []
        y_batch = []
        class_num_map = {}
        test_file_list = utils.find_all_files(self.parameters['test_corpus_dir'], [])
        lines = []
        for file_name in test_file_list:
            lines += utils.read_lines_small_file(file_name)
            
        random.shuffle(lines)
        for text_file in lines:
            [text,file_name] = text_file
            text = preProcess.filtUrl(text)
            text = text.replace(" ", '').replace('\n', '')[:self.parameters['max_text_length']]
            if len(text)==0: continue
            id_list = self.trans_char2id(text)
#             print(text)
#             print(id_list)
            class_label = file_name.split('/')[-2]
            class_num_map[class_label] = class_num_map.get(class_label, 0) + 1
            #if class_num_map[class_label]>20: continue
            class_label_one_hot = self.class_label_one_hot[class_label]
            x_batch.append(id_list)
            y_batch.append(class_label_one_hot)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        return x_batch, y_batch
        
    def trans_char2id(self, text):
        id_list = [self.char_id_map.get(char, 0) if char not in stop_chars else 2 for char in text]
        length_dif = self.parameters['max_text_length'] - len(id_list)
        if length_dif>0: id_list += [1]*length_dif
        id_list.reverse()
        return id_list
                
    
    def predict(self, text_list):
        x_batch = []
        for text in text_list:
            text = text.replace(" ", '').replace('\n', '')[:self.parameters['max_text_length']]
            id_list = self.trans_char2id(text)
            x_batch.append(id_list)
        Y = self.sess.run([self.predictions], feed_dict={self.X: x_batch})
        return Y
    
    def eveluate_this_model(self, test_corpus_dir):
        pass
    
if __name__ == '__main__':
    model = TextCNN(with_pretrained_char_embedding=False, static_embedding=False)
#     model = TextCNN(with_pretrained_char_embedding=True, static_embedding=True, pretrained_model='./check_points_dir/model')
    model.fit()
#     model = TextCNN(with_pretrained_char_embedding=True, static_embedding=True, pretrained_model='./check_points_dir/model')
#     res = model.predict(['我跟你说过多少遍了'])
#     print("预测结果是", res)
#     model = TextCNN()
#     print(model.__dict__.keys())
    
    