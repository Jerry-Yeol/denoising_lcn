
import cv2
import numpy as np
import tensorflow as tf

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def gaussian_function(x, y, sigma):
    
    frac = 1/(2*np.pi*sigma**2)
    expo = -(x**2 + y**2)/(2*sigma**2)
    
    return frac*np.exp(expo)

def gaussian_filter(x, sigma):
    '''
    x is odd
    '''
    mid_point = np.uint8(x/2)
    out = np.zeros([x, x])
    for i in range(x):
        for j in range(x):
            out[i, j] = gaussian_function(j-mid_point, i-mid_point, sigma)
            
    return out/out.sum()

def gaussian_conv_easy(img, ksize):
  
    h, w = img.shape
    
    p = int(ksize/2)

    tmp_img = np.zeros([h+2*p, w+2*p])
    tmp_img[p:p+h, p:p+w] = img

    out = np.zeros_like(img, dtype=np.float32)
    tmp_h, tmp_w = tmp_img.shape
    
    for h_i in range(tmp_h-ksize+1):
        sigma = np.log(ksize)
        step = sigma/(h/1.5-1)
        #sigma = 1.5
        #step = 1.5*(sigma-0.01)/h
        cnt =1
        for w_i in range(tmp_w-ksize+1):
            
            filt = gaussian_filter(ksize, sigma**2)
           
            out[h_i, w_i] = (tmp_img[h_i:h_i+ksize, w_i:w_i+ksize]*filt).sum()
            
            if cnt < (tmp_w-ksize+1)/2 :
                sigma -= step
            elif cnt > (tmp_w-ksize+1)/2 :
                sigma += step
                
            cnt += 1
        
            
    return out

def gaussian_conv_medium(img, ksize):
    
    h, w = img.shape
    
    p = int(ksize/2)

    tmp_img = np.zeros([h+2*p, w+2*p])
    tmp_img[p:p+h, p:p+w] = img

    out = np.zeros_like(img, dtype=np.float32)
    tmp_h, tmp_w = tmp_img.shape
    
    for h_i in range(tmp_h-ksize+1):
        
        for w_i in range(tmp_w-ksize+1):
            
            filt = gaussian_filter(ksize, 3*(np.random.ranf(1)))
           
            out[h_i, w_i] = (tmp_img[h_i:h_i+ksize, w_i:w_i+ksize]*filt).sum()
                 
    return out

def gaussian_conv_hard(img):
    
    h, w = img.shape
    
    fil_shape = np.array(range(h))
    fil_shape[fil_shape%2==0] += 1
    fil_shape = abs(fil_shape-h/2)+2
    
    
    p = int(fil_shape[0]/2)

    tmp_img = np.zeros([h+2*p, w+2*p])
    tmp_img[p:p+h, p:p+w] = img

    out = np.zeros_like(img, dtype=np.float32)

    
    for h_i in range(h):
        tmp = fil_shape[0]
        for (w_i, f) in zip(range(w), fil_shape):
            f = int(f)
            filt = gaussian_filter(f, f/np.sqrt(tmp))
            f_h, f_w = filt.shape
            mid_point = int(f_h/2)
            #print(mid_point)
            h_min = int(abs(tmp-f)/2)+h_i
            w_min = int(abs(tmp-f)/2)+w_i
            #print("[%d:%d, %d:%d], %d"%(h_min, h_min+f, w_min, w_min+f, f))
            
            #print((tmp_img[h_min:h_min + f, w_min:w_min+f]*filt).sum())
            out[h_i, w_i] = (tmp_img[h_min:h_min + f, w_min:w_min+f]*filt).sum()
        
    return out

def resizeNscale(img, size):
    
    out = cv2.resize(img, (size, size)) # interpolation : bilinear
    
    _max = out.max()
    _min = out.min()
    
    out = (out - _min)/(_max - _min)
    
    return out
    
    
def extend(x):

    '''
    Extend tensor X
    b, h, w, c => b, 2h, 2w, c

    '''

    b,h, w, c = x.get_shape().as_list()



    out = tf.transpose(x, [0,3,1,2])       # (b, h, w, c) - > (b, c, h, w)
    out = tf.reshape(out, [-1,1])          # (b, c, h, w) - > (b*c*h*w, 1)
    out = tf.matmul(out, tf.ones([1,2]))      # (b*c*h*w, 1) - > (b*c*h*w, 2)


    out = tf.reshape(out, [-1, c, h, 2*w])  # (b*c*h*w, 2) - > (b, c, h, 2*w)
    out = tf.transpose(out, [0,1,3,2])     # (b, c, h, 2*w) - > (b, c, 2*w, h)
    out = tf.reshape(out, [-1,1])          # (b, c, 2*w, h) - > (b*c*2w*h, 1)
    out = tf.matmul(out, tf.ones([1,2]))      # (b*c*2w*h, 1) - > (b*c*2w*h, 2)


    out = tf.reshape(out, [-1, c, 2*w, 2*h]) # (b*c*2w*h, 2) - > (b, c, 2w, 2h)
    out = tf.transpose(out, [0, 3, 2, 1])

    return out



def extract_loc(x, y):

    '''
    x : input data at pooling layer
    y : extending pooling data
    x'shape == y'shape

    '''

    out = tf.equal(x, y)                  # tf.equal([[1,1],[3,4]], [[4,4],[4,4]]) = [[False, False],[False, True]]
    out = tf.cast(out, dtype=tf.float32)  # tf.cast([[False, False],[False, True]], dtype = tf.float32) = [[0.,0.],[0.,1.]]


    return out

def unpool2d(x, y):
    _x = extend(x)
    out = extract_loc(_x, y)
    return out

def init_w(name, shape):
    '''
    shape : [filter_h, filter_w, input_channel, ouput_channel]
    '''
    w = tf.get_variable(name, shape=shape,
                          initializer=tf.contrib.layers.xavier_initializer())
    return w

def init_b(name, shape):
    '''
    shape : [filter_h, filter_w, input_channel, ouput_channel]
    '''
    b = tf.get_variable(name, shape=shape,
                          initializer=tf.contrib.layers.xavier_initializer())
    return b

def max_pool_2d(_input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME"):
    pool = tf.nn.max_pool(_input,  ksize=ksize, strides=strides, padding="SAME")
    return pool


def conv2d(_input, weight, strides = [1,1,1,1], padding = "SAME"):
    conv = tf.nn.conv2d(_input, weight, strides=strides, padding = padding)
    return conv



def deconv2d(_input, weight, _output_shape = None, strides = [1, 1, 1, 1], padding = "SAME"):
    _input_shape = _input.shape.as_list()
    weight_shape = weight.shape.as_list()

    if _output_shape == None:
        _output_shape = [tf.shape(_input)[0], _input_shape[1], _input_shape[2], weight_shape[3]]
    add_zero = tf.zeros([1, _output_shape[1], _output_shape[2], _output_shape[3],])
    deconv = tf.nn.conv2d_transpose(_input, weight, output_shape=_output_shape, strides=strides, padding=padding)
    return (deconv + add_zero)


def batch_norm(_input, center=True, scale=True, decay=0.8, is_training=True):
    norm = tf.contrib.layers.batch_norm(_input, center=center, scale = scale,
                                        decay = decay, is_training=is_training)
    return norm

