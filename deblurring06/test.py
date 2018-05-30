import time, cv2, os, sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
from utils import *
from matplotlib import pyplot as plt
tf.set_random_seed(777)
from tensorflow import keras as ke

tmp = np.load('/data1/jerry/project/deblurring/data/same_sig/new_data_128_easy_3_shuffle.npz')

img = np.reshape(tmp['data'], [-1, 128, 128, 1])
lab = np.reshape(tmp['label'], [-1, 128, 128, 1])

print('Input Size : %s'%str(img.shape))
print('Label Size : %s'%str(lab.shape))

tr_idx = []
te_idx = []
cnt = []
for i, d in enumerate(img):
    tmp = (d!=0).sum()
    if tmp > 16382 : 
        cnt.append(tmp)
        tr_idx.append(i)
    else :
        te_idx.append(i)
tr_idx = np.array(tr_idx)
te_idx = np.array(te_idx)
#print(idx)    
print('Amount of train data : %s'%str(len(tr_idx)))
print('Amount of test data : %s'%str(len(te_idx)))

tr_img = img[tr_idx]
tr_lab = lab[tr_idx]

te_img = img[te_idx]
te_lab = lab[te_idx]

print('Train Image Size : %s'%str(tr_img.shape))

print('Test Image Size : %s'%str(te_img.shape))

model = ke.Sequential()
model.add(ke.layers.Reshape((130,130,1), input_shape=(128,128,1)))
model.add(ke.layers.LocallyConnected2D(1, (3, 3), activation = 'relu', input_shape=(128,128,1)))
model.add(ke.layers.Reshape((130,130,1), input_shape=(128,128,1)))
model.add(ke.layers.LocallyConnected2D(1, (3, 3), input_shape=(128,128,1)))

model.compile(optimizer='adam',
             loss = 'mean_squared_error',
             metrics=[ke.metrics.mean_squared_error])

model.fit(tr_img, tr_lab, batch_size=50, epochs=1000, verbose=1)

score = model.evaluate(te_img, te_lab, verbose=0)

print(model.metrics_names)
print(score)