from utils import *
import cv2, os, glob, time, sys
import numpy as np

origin_data = np.load('/data1/project/tumor/data/input/data_2d.npz')['img']
print("Loaded Original Data")
tot_start = time.time()

size = int(sys.argv[1])
ksize = int(sys.argv[2])
new_data = np.zeros([origin_data.shape[0], size, size])
loop_start = time.time()
for idx, img in enumerate(origin_data):
    new_data[idx] = gaussian_conv_medium(resizeNscale(img, size), ksize)
    if (idx+1)%350 == 0:
        print("%d/%d"%(idx+1, len(origin_data)))
        print("Time : ", time.time()-loop_start)
        loop_start = time.time()
print("%d/%d"%(idx+1, len(origin_data)))
print("Time : ", time.time()-tot_start)
print("Making done")    

new_label = np.zeros([origin_data.shape[0], size, size])
for idx, img in enumerate(origin_data):
    new_label[idx] = resizeNscale(img, size)
    
'''
for i in range(4):
    shu_seq = np.random.choice(len(new_data), len(new_data), replace=False)

    new_data = new_data[shu_seq]
    new_label = new_label[shu_seq]
'''   
np.savez('/data1/jerry/project/deblurring/data/diff_sig/data_%d_medium_%d'%(size, ksize), data = new_data, label = new_label)