'''
Created on 2019年6月7日

@author: Administrator
用cnn实现一个文本分类器
'''

import tensorflow as tf
import utils
import json, pickle
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec


class TextCNN():
    
    def __init__(self, with_pretrained_char_embedding=False, static_embedding=True):
        self.load_parameters()
        self.load_char_id_data()

        if with_pretrained_char_embedding==False:
            self.char_embedding = tf.Variable(tf.random_normal([self.parameters['char_set_size'], \
                                                            self.parameters['embedding_size']]))
        else:
            self.load_pretrained_char_embedding(static_embedding)

        self.load_graph()
    
    def load_pretrained_char_embedding(self, static_embedding):
        char_vector_from_gensim = Word2Vec.load("./model/word2vec.model")
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
        self.X = tf.placeholder(dtype=tf.int32, shape=[None, self.parameters['max_text_length']], name='id_list')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.parameters['class_num']])
        X_char_vec = tf.nn.embedding_lookup(self.char_embedding, self.X)
        X_char_vec4cnn = tf.expand_dims(X_char_vec, -1)
        #1号卷积核
        self.filter_1_w = tf.Variable(tf.random_normal([2, self.parameters['embedding_size'], 1, 50]))
        self.filter_1_b = tf.Variable(tf.random_normal([50]))
        conv_1 = tf.nn.conv2d(X_char_vec4cnn, self.filter_1_w, strides=[1, 1, 1, 1], padding='VALID')
        hidden_out_1 = tf.nn.relu(tf.nn.bias_add(conv_1, self.filter_1_b))
        pooled_out_1 = tf.nn.max_pool(hidden_out_1, \
                                      ksize=[1, self.parameters['max_text_length'] - 2 + 1, 1, 1],\
                                      strides = [1, 1, 1, 1], padding='VALID')
        print("pooled_out_1 size is ", pooled_out_1)
        flatten_out_1 = tf.reshape(pooled_out_1, [-1, 50])
        
        
        #2号卷积核
        self.filter_2_w = tf.Variable(tf.random_normal([3, self.parameters['embedding_size'], 1, 50]))
        self.filter_2_b = tf.Variable(tf.random_normal([50]))
        conv_2 = tf.nn.conv2d(X_char_vec4cnn, self.filter_2_w, strides=[1, 1, 1, 1], padding='VALID')
        hidden_out_2 = tf.nn.relu(tf.nn.bias_add(conv_2, self.filter_2_b))
        pooled_out_2 = tf.nn.max_pool(hidden_out_2, \
                                      ksize=[1, self.parameters['max_text_length'] - 3, 1, 1],\
                                      strides = [1, 1, 1, 1], padding='VALID')
        print("pooled_out_2 size is ", pooled_out_2)
        flatten_out_2 = tf.reshape(pooled_out_2, [-1, 100])
        
        
        #3号卷积核
        self.filter_3_w = tf.Variable(tf.random_normal([4, self.parameters['embedding_size'], 1, 50]))
        self.filter_3_b = tf.Variable(tf.random_normal([50]))
        conv_3 = tf.nn.conv2d(X_char_vec4cnn, self.filter_3_w, strides=[1, 1, 1, 1], padding='VALID')
        hidden_out_3 = tf.nn.relu(tf.nn.bias_add(conv_3, self.filter_3_b))
        pooled_out_3 = tf.nn.max_pool(hidden_out_3, \
                                      ksize=[1, self.parameters['max_text_length'] - 4, 1, 1],\
                                      strides = [1, 1, 1, 1], padding='VALID')
        print("pooled_out_2 size is ", pooled_out_3)
        flatten_out_3 = tf.reshape(pooled_out_3, [-1, 100])
        
        #拼接
        flatten_out = tf.concat([flatten_out_1, flatten_out_2, flatten_out_3], axis=1)
        #拉直
        #dropout
        flatten_out_droped = tf.nn.dropout(flatten_out, 0.6)
        print("flatten_out_droped 的形状是" , flatten_out_droped)
        #softmax
        
        self.softmax_w = tf.Variable(tf.random_normal([250, self.parameters['class_num']]))
        self.softmax_b = tf.Variable(tf.random_normal([self.parameters['class_num']]))
        
        prob_dist = tf.matmul(flatten_out_droped, self.softmax_w) + self.softmax_b
        
        self.predictions = tf.argmax(prob_dist, 1, name="predictions")
        
        self.losses = tf.reduce_mean(\
                        tf.nn.softmax_cross_entropy_with_logits(logits=prob_dist, labels=self.Y))
        
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        self.train = tf.train.AdagradOptimizer(0.05).minimize(self.losses)
#         self.train = tf.train.GradientDescentOptimizer(0.05).minimize(self.losses)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

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
        x_batch = []
        y_batch = []
        for epoch in range(1000):
            random.shuffle(file_list)
            for file_name in file_list:
                print(file_name)
                lines = utils.read_lines_small_file(file_name)
                text = self.get_title_content(lines)
                if len(text)==0: continue
                id_list = self.trans_char2id(text)
                class_label = file_name.split('/')[-2]
                print(self.class_label_one_hot)
                class_label_one_hot = self.class_label_one_hot[class_label]
                count += 1

                x_batch.append(id_list)
    #             print(x_batch)
                y_batch.append(class_label_one_hot)
                if len(x_batch)==500:
                    #打乱顺序
                    random_index = list(range(500))
                    random.shuffle(random_index)
    #                 print(x_batch)
                    x_batch = np.array(x_batch)[random_index]
                    y_batch = np.array(y_batch)[random_index]
                    
    #                 print(x_batch.shape)
                    #训练
                    _, loss, accuracy = self.sess.run([self.train, self.losses, self.accuracy],\
                                                      feed_dict={self.X: x_batch, self.Y: y_batch})
                    #打印损失值
                    
                    x_batch = []
                    y_batch = []
                    
                if count%5000==0:
                    print('epoch ', epoch," loss is ", loss, '。 accuracy is ', accuracy)
                    loss, accuracy = self.sess.run([self.losses, self.accuracy],\
                                  feed_dict={self.X: test_input, self.Y: test_output})
                    print("在测试集中的loss为", loss, 'accuracy为', accuracy)
                    self.saver.save(self.sess, self.parameters['check_points_dir'] + '/model')
    
    def load_test_data(self):
        x_batch = []
        y_batch = []
        class_num_map = {}
        test_file_list = utils.find_all_files(self.parameters['test_corpus_dir'], [])
        for file_name in test_file_list:
            lines = utils.read_lines_small_file(file_name)
            text = self.get_title_content(lines)
            if len(text)==0: continue
            id_list = self.trans_char2id(text)
            class_label = file_name.split('/')[-2]
            class_num_map[class_label] = class_num_map.get(class_label, 0) + 1
            if class_num_map[class_label]>20: continue
            print(self.class_label_one_hot)
            class_label_one_hot = self.class_label_one_hot[class_label]
            x_batch.append(id_list)
            y_batch.append(class_label_one_hot)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        return x_batch, y_batch
        
        
    def trans_char2id(self, text):
        id_list = [self.char_id_map.get(char, 0) for char in text]
        length_dif = self.parameters['max_text_length'] - len(id_list)
        if length_dif>0: id_list += [1]*length_dif
        return id_list
            
    def get_title_content(self, lines):
        title_index, content_start_index = 0, 0
        for i in range(len(lines)):
            line = lines[i]
            if '标  题' in line:
                title_index = i
            if '正  文' in line:
                content_start_index = i
                break
#         print(lines)
        if title_index==0 or content_start_index==0:
            text = ''
        else:
            text = lines[title_index].split('】')[1] + '。' 
        for i in range(content_start_index + 1, len(lines)):
            text += lines[i]
            if len(text)>self.parameters['max_text_length']:
                text = text[:self.parameters['max_text_length']]
                break
        text = text.replace('\n', '').replace(' ', '')
        return text
                
    
    def predict(self, text_list):
        pass
    
    def eveluate_this_model(self, test_corpus_dir):
        pass
        
        
if __name__ == '__main__':
    model = TextCNN(with_pretrained_char_embedding=True, static_embedding=False)
    model.fit()
    
    