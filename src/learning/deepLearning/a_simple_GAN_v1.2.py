'''
Created on 2019年10月11日

@author: lipy
'''
#一个简单的对抗生成网络，用来生成和判断手写体数字
# from tensorflow.keras import datasets
from keras import datasets
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
    
class AGAN():
    
    def __init__(self):
        self.init_graph()
        
    def init_graph(self):
        self.X = tf.placeholder(tf.float32, shape=[None, 784])
        self.fake_images = tf.placeholder(tf.float32, shape=[None, 100])
        self.dropout_rate = tf.placeholder(tf.float32, shape=[])
        self.if_train = tf.placeholder(tf.bool, shape=[])
        
        
        #生成器
        self.gen_images = self.gen_layer()
        #判别器
        self.prob_fake = self.dis_layer(self.gen_images)
        self.prob_real = self.dis_layer(self.X) 
        self.loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.prob_real), logits=self.prob_real*0.95))
        self.loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.prob_fake), logits=self.prob_fake))
        
        self.loss_d = tf.add(self.loss_d_real, self.loss_d_fake) 
        self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.prob_fake), logits=self.prob_fake*0.95))#最大化self.prob_fake
        
        #损失函数
        train_vars = tf.trainable_variables()
        g_vars = [var for var in train_vars if var.name.startswith("gen")]
        d_vars = [var for var in train_vars if var.name.startswith("dis")]
        self.train_d = tf.train.AdamOptimizer(0.001, beta1 = 0.2).minimize(self.loss_d, var_list=d_vars)
        self.train_g = tf.train.AdamOptimizer(0.001, beta1 = 0.2).minimize(self.loss_g, var_list=g_vars)
    
    def gen_layer(self):
        with tf.variable_scope('gen', reuse=tf.AUTO_REUSE) as scope:
                gen_layer_1 = tf.layers.dense(inputs=self.fake_images, units=200, activation=tf.nn.leaky_relu)
#                 gen_layer_2 = tf.nn.dropout(gen_layer_1, rate=self.dropout_rate)
                if self.if_train==True:
                    gen_layer_1 = tf.layers.batch_normalization(gen_layer_1, training=True)
                gen_layer_2 = tf.layers.dense(inputs=gen_layer_1, units=1000, activation=tf.nn.leaky_relu)
#                 gen_layer_2 = tf.nn.dropout(gen_layer_2, rate=self.dropout_rate)
                if self.if_train==True:
                    gen_layer_2 = tf.layers.batch_normalization(gen_layer_2, training=True)
                gen_images = tf.layers.dense(inputs=gen_layer_2, units=784, activation=tf.nn.tanh)
        return gen_images
    
    def dis_layer(self, inputs):
        with tf.variable_scope('dis', reuse=tf.AUTO_REUSE) as scope:
            dis_layer_1 = tf.layers.dense(inputs=inputs, units=200, activation=tf.nn.leaky_relu,name='Discri1')#             dis_layer_1 = tf.nn.dropout(dis_layer_1, rate=self.dropout_rate)
            if self.if_train==True:
                dis_layer_1 = tf.layers.batch_normalization(dis_layer_1, training=self.if_train)
            dis_layer_2 = tf.layers.dense(inputs=dis_layer_1, units=500, activation=tf.nn.leaky_relu,name='Discri2')
            if self.if_train==True:
                dis_layer_2 = tf.layers.batch_normalization(dis_layer_2, training=self.if_train)
#             dis_layer_2 = tf.layers.dense(inputs=dis_layer_2, units=100, activation=tf.nn.leaky_relu,name='Discri3')
#             if self.if_train==True:
#                 dis_layer_2 = tf.layers.batch_normalization(dis_layer_2, training=self.if_train)
#             dis_layer_2 = tf.nn.dropout(dis_layer_2, rate=self.dropout_rate)
            features = tf.layers.dense(inputs=dis_layer_2, units=1,name='Discri4')#, activation=tf.nn.sigmoid
        return  features

    def init_tf_graph(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # need ~700MB GPU memory
        self.sess = tf.Session(config=config)
#         self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())
    def fit(self, train_X, train_Y):
        self.init_tf_graph()
        data = Data()
        train_X_new, train_Y_new = train_X, train_Y
        for epoch in range(10000):
            random_index = list(range(len(train_X_new)))
            random.shuffle(random_index)
            train_X, train_Y = np.array(train_X_new)[random_index], np.array(train_Y_new)[random_index]
            train_X, train_Y = data.preprocess(train_X, train_Y)
            X_batches, _ = data.get_data_batches(train_X, train_Y)
            random_images_list = []
            for i in range(len(X_batches)):
                random_images_list.append(np.random.uniform(-1, 1, len( X_batches[i])*100 ).reshape(len( X_batches[i]), 100))
            for i in range(len(X_batches)):
                total_step = epoch*len(X_batches) + i
                random_images = random_images_list[i]
                if epoch < 1:
                    new_data_batch = X_batches[i]# + np.random.normal(0, 0.1, (len(X_batches[i]), 784))
                else:
                    new_data_batch = X_batches[i]
                _, loss_d_value = self.sess.run((self.train_d, self.loss_d), \
                            feed_dict={self.X: new_data_batch, self.fake_images:random_images, self.dropout_rate: 0.5, self.if_train: True})
                _, loss_g_value = self.sess.run((self.train_g, self.loss_g), \
                                                feed_dict={self.fake_images:random_images, self.dropout_rate: 0.5, self.if_train: True})
                if (total_step + 1)%1000==0:
                    loss_g_value, loss_d_real, loss_d_fake = self.sess.run((self.loss_g, self.loss_d_real, self.loss_d_fake), \
                        feed_dict={self.X: new_data_batch, self.fake_images:random_images, self.dropout_rate: 0, self.if_train: False})
                    print('epoch', epoch, 'step', i, 'total_step', total_step, 'loss_d_value', loss_d_value, \
                          'loss_g_value', loss_g_value, 'loss_d_real', loss_d_real, 'loss_d_fake', loss_d_fake)
#                     print(prob_fake)
#                 if (total_step + 1)%10000==0:
            test_data = np.random.uniform(-1, 1, 5*100).reshape(5, 100)
            gen_images = self.sess.run((self.gen_images), feed_dict={self.fake_images: test_data, self.dropout_rate: 0, self.if_train: False})
            gen_images = gen_images.reshape(-1,28,28)
            data.draw_images(gen_images, epoch=total_step)
            print("完成一次打印")
        

class Data():
    
    def __init__(self):
        dir_path = './gen_images'
        if os.path.exists(dir_path):
            os.system("rm -rf ./gen_images")
        try:
            os.mkdir('./gen_images')
        except:
            pass
    #加载手写体数字数据 
    def load_data(self):
        (X_ori, Y_ori), (_, _)  = self.load_mnist_data()
        X_ori = X_ori.reshape(-1, 784)
        return X_ori, Y_ori
    
    def preprocess(self, X_ori, Y_ori):
#         X = X_ori.reshape(-1,784)/255.0 #- 1
        X = 2*X_ori.reshape(-1,784)/255.0 - 1
        Y = np.zeros((len(Y_ori), 10))
        for i in range(len(Y_ori)): Y[i, Y_ori[i]] = 1
        return X, Y
    
    def draw_images(self, digits_list, epoch=0, dim=(28,28), figsize=(10,2)):
        digits_list = digits_list[:5]
        plt.clf()
        f = plt.figure(figsize=figsize)

        for i in range(digits_list.shape[0]):
            image = digits_list[i].reshape(dim)
#             image = image*255.0
#             print(i, image)
            image = (image+ 1)*255/2
            plt.subplot(1, len(digits_list), i+1)
#             plt.figure()
            plt.imshow(image, interpolation='nearest', cmap='Greys')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('./gen_images/Generated_images %d.png' %epoch)
        plt.clf()
        plt.close(f)
        
    def get_data_batches(self, X, Y, bacth_size=50):
        X_batches, Y_batches = [], []
        for i in range(0, len(X), bacth_size):
            X_batches.append(X[i: i + bacth_size, :])
            Y_batches.append(Y[i: i + bacth_size])
        return X_batches, Y_batches
        
    def load_mnist_data(self, path='mnist.npz'):
        f = np.load(path)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
        return (x_train, y_train), (x_test, y_test)
            

if __name__ == '__main__':
    data = Data()
    X, Y = data.load_data()
#     data.draw_images(X[0:10,:])
    model = AGAN()
    model.fit(X, Y)
    

    