from utils import *
import tensorflow as tf


class Model:

    def __init__(self, sess, name, in_size, ksize, lr):
        self.sess = sess
        self.name = name
        self.in_size = in_size
        self.ksize = ksize
        self.lr = lr
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name, values=[self.in_size, self.ksize, self.lr]):
            
            # input place holders
            self.X = tf.placeholder(tf.float32, shape = [None, self.in_size, self.in_size, 1])
            # img 64x64x1 (gray scale)
            self.Y = tf.placeholder(tf.float32, shape = [None, self.in_size, self.in_size])
            
            self.keep = tf.placeholder(tf.float32)
          
            in_x = tf.image.resize_image_with_crop_or_pad(self.X,
                                                          self.in_size+self.ksize-1, self.in_size+self.ksize-1)
            in_x = tf.extract_image_patches(in_x,
                                            ksizes=[1, self.in_size, self.in_size, 1],
                                            strides=[1,1,1,1],
                                            rates = [1,1,1,1], padding="VALID")
            
            w1 = init_w('w1', [1, self.ksize, self.ksize, in_x.shape[3]])
            L1 = tf.nn.elu(tf.reduce_sum(tf.multiply(in_x, w1), axis = (1,2)))
            L1 = tf.reshape(L1, [-1, self.in_size, self.in_size, 1])
            
            in_L2 = tf.image.resize_image_with_crop_or_pad(L1,
                                                          self.in_size+self.ksize-1, self.in_size+self.ksize-1)
            in_L2 = tf.extract_image_patches(in_L2,
                                            ksizes=[1, self.in_size, self.in_size, 1],
                                            strides=[1,1,1,1],
                                            rates = [1,1,1,1], padding="VALID")
            w2 = init_w('w2', [1, self.ksize, self.ksize, in_L2.shape[3]])
            L2 = tf.nn.elu(tf.reduce_sum(tf.multiply(in_L2, w2), axis = (1,2)))
            L2 = tf.reshape(L2, [-1, self.in_size, self.in_size, 1])
            
            in_L3 = tf.image.resize_image_with_crop_or_pad(L2,
                                                          self.in_size+self.ksize-1, self.in_size+self.ksize-1)
            in_L3 = tf.extract_image_patches(in_L3,
                                            ksizes=[1, self.in_size, self.in_size, 1],
                                            strides=[1,1,1,1],
                                            rates = [1,1,1,1], padding="VALID")
            w3 = init_w('w3', [1, self.ksize, self.ksize, in_L2.shape[3]])
            L3 = tf.reduce_sum(tf.multiply(in_L3, w3), axis = (1,2))
            
            
            
            
            self.logits = tf.reshape(L3, [-1, self.in_size, self.in_size])
            

        # define cost/loss & optimizer
        #beta = 0.01
        #self.regularizers = tf.nn.l2_loss(w1)# + tf.nn.l2_loss(w2)
        self.cost = tf.reduce_mean(tf.square(self.logits-self.Y))# + beta*self.regularizers)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr).minimize(self.cost)
    
    def _cost(self, x_data, y_data, keep = 0.7):
        return self.sess.run(self.cost, feed_dict={
            self.X: x_data, self.Y: y_data, self.keep: keep})
    
    def train(self, x_data, y_data, keep = 0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep: keep})
    
    def predict(self, x_test, keep = 1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep: keep})