import time, cv2, os, sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
from utils import *
#from matplotlib import pyplot as plt
tf.set_random_seed(777)
print('Package Loaded')

tmp = np.load('/data1/jerry/project/deblurring/data/diff_sig/data_128_medium_3.npz')
timg = tmp['data']
img = np.zeros_like(timg)
img[:,:64,:64] = timg[:,:64,:64]
lab = np.reshape(tmp['label'], [-1, 128,128])

tmp = np.load('/data1/jerry/project/deblurring/data/diff_sig/data_128_medium_5.npz')
timg = tmp['data']
img[:,:64,64:] = timg[:,:64,64:]

tmp = np.load('/data1/jerry/project/deblurring/data/diff_sig/data_128_medium_7.npz')
timg = tmp['data']
img[:,64:,:64] = timg[:,64:,:64]

tmp = np.load('/data1/jerry/project/deblurring/data/diff_sig/data_128_medium_9.npz')
timg = tmp['data']
img[:,64:,64:] = timg[:,64:,64:]

tr_idx = []
te_idx = []
cnt = []
for i, d in enumerate(img):
    tmp = (d!=0).sum()
    if tmp > 16382 : 
        cnt.append(tmp)
        tr_idx.append(i)
    else : te_idx.append(i)
tr_idx = np.array(tr_idx)
te_idx = np.array(te_idx)

tr_img = np.expand_dims(img[tr_idx], 3)
tr_lab = lab[tr_idx]

te_img = np.expand_dims(img[te_idx], 3)
te_lab = lab[te_idx]

print('Train Image Size : %s'%str(tr_img.shape))
print('Train Label Size : %s'%str(tr_lab.shape))

print('Test Image Size : %s'%str(te_img.shape))
print('Test Label Size : %s'%str(te_lab.shape))

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
            # L1 ImgIn shape = (?, node)
            w1 = init_w('w1', [1, self.ksize, self.ksize, in_x.shape[3]])
            L1 = tf.reduce_sum(tf.multiply(in_x, w1), axis = (1,2))
            L1 = tf.reshape(L1, [-1, self.in_size, self.in_size, 1])
            
            in_L2 = tf.image.resize_image_with_crop_or_pad(L1,
                                                          self.in_size+self.ksize-1, self.in_size+self.ksize-1)
            in_L2 = tf.extract_image_patches(in_L2,
                                            ksizes=[1, self.in_size, self.in_size, 1],
                                            strides=[1,1,1,1],
                                            rates = [1,1,1,1], padding="VALID")
            w2 = init_w('w2', [1, self.ksize, self.ksize, in_L2.shape[3]])
            L2 = tf.reduce_sum(tf.multiply(in_L2, w2), axis = (1,2))
            self.logits = tf.reshape(L2, [-1, self.in_size, self.in_size])
            

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
    
sess = tf.Session()
deblur = Model(sess, "deblur", 128, 3, 1e-3)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess, './checkpoint/locally')
print("Initializatin Done")

val_tr_cost = []
val_te_cost = []

print("Start Train")
start = time.time()
epochs = 10000
batch = 200
for epoch in range(epochs):
    avg_train_cost = 0
    avg_test_cost = 0
    total_batch = int(len(tr_idx)/batch)

    for i in range(total_batch):
        
        tr_batch_seq = np.random.choice(len(tr_img), batch, replace=False)
        batch_xs = tr_img[tr_batch_seq]
        batch_ys = tr_lab[tr_batch_seq]
        c, _ = deblur.train(batch_xs, batch_ys)
        avg_train_cost += c/total_batch
    
        te_batch_seq = np.random.choice(len(te_img), batch, replace=False)
        test_c = deblur._cost(te_img[te_batch_seq], te_lab[te_batch_seq])

        avg_test_cost += test_c / total_batch
        val_tr_cost.append(avg_train_cost)
        val_te_cost.append(avg_test_cost)

    
    
    if (epoch+1)%100 == 0:
        tr_batch_seq = np.random.choice(len(tr_idx), 1, replace=False)
        test_tr = deblur.predict(tr_img[[tr_batch_seq[0]]])
        te_batch_seq = np.random.choice(len(te_img), 1, replace=False)
        test_te = deblur.predict(te_img[[te_batch_seq]])
        print("Epoch  : %d/%d"%(epoch+1, epochs))
        print("Train Cost : %.9f"%(avg_train_cost))
        print("Test Cost  : %.9f"%(avg_test_cost))
        print("----------------------------\n\n")
        '''
        plt.figure(figsize=(20, 10))

        plt.subplot(2,6,1)
        plt.title("Input")
        plt.imshow(tr_img[tr_batch_seq[0],:,:,0], cmap=plt.cm.bone)
        plt.subplot(2,6,2)
        plt.title("Inference")
        plt.imshow(test_tr[0], cmap=plt.cm.bone)
        plt.subplot(2,6,3)
        plt.title("Label")
        plt.imshow(tr_lab[tr_batch_seq[0]], cmap=plt.cm.bone)
        plt.subplot(2,6,4)
        plt.title("Infer - Input")
        plt.imshow(test_tr[0] - tr_img[tr_batch_seq[0],:,:,0], cmap=plt.cm.bone)
        plt.subplot(2,6,5)
        plt.title("Label - Input")
        plt.imshow(tr_lab[tr_batch_seq[0]] - tr_img[tr_batch_seq[0],:,:,0], cmap=plt.cm.bone)
        plt.subplot(2,6,6)
        plt.title("Label - Infer")
        plt.imshow(tr_lab[tr_batch_seq[0]] - test_tr[0], cmap=plt.cm.bone)
        
        plt.subplot(2,6,7)
        plt.title("Input")
        plt.imshow(te_img[te_batch_seq[0],:,:,0], cmap=plt.cm.bone)
        plt.subplot(2,6,8)
        plt.title("Inference")
        plt.imshow(test_te[0], cmap=plt.cm.bone)
        plt.subplot(2,6,9)
        plt.title("Label")
        plt.imshow(te_lab[te_batch_seq[0]], cmap=plt.cm.bone)
        plt.subplot(2,6,10)
        plt.title("Infer - Input")
        plt.imshow(test_te[0] - te_img[te_batch_seq[0],:,:,0], cmap=plt.cm.bone)
        plt.subplot(2,6,11)
        plt.title("Label - Input")
        plt.imshow(te_lab[te_batch_seq[0]] - te_img[te_batch_seq[0],:,:,0], cmap=plt.cm.bone)
        plt.subplot(2,6,12)
        plt.title("Label - Infer")
        plt.imshow(te_lab[te_batch_seq[0]] - test_te[0], cmap=plt.cm.bone)
        
        plt.show()
        '''
print("Learning Finished!")
print("Elapsed time : ", time.time()-start)

saver.save(sess, './checkpoint/locally01')
print("Model saving Done.")