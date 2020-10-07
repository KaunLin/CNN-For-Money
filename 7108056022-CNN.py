
# coding: utf-8

# In[40]:


import numpy as np
from numpy import expand_dims
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
from PIL import Image
from pylab import rcParams
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# # Data Augmentation

# ## 初始化ImageDataGenerator物件，並定義其參數

# In[42]:


'''
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1)
train_generator = train_datagen.flow_from_directory(
        "C:/Users/User/Desktop/money/train/",
        target_size=(128, 128),  #可以在此指定要rescale的尺寸
        batch_size=30,
        class_mode='categorical')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
'''
cwd = 'C:/Users/User/Desktop/money/train/'
classes = {'100','500','1000'}
for index,name in enumerate(classes):
    class_path = cwd+name+'/';
    for img_name in os.listdir(class_path):
        img_path = class_path+img_name
        img = cv.imread(img_path)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(samples, batch_size=1)
        for i in range(9):
            plt.subplot(330 + 1 + i)
            batch = it.next()
            image = batch[0].astype('uint8')
            cv.imwrite(class_path+img_name+"shif"+str(i)+".jpg",image)
            plt.imshow(image)
plt.show()


# # make_own_data
# ## 製作tfrecords檔案

# In[43]:


cwd = 'C:/Users/User/Desktop/money/train/'
classes = {'100','500','1000'}
writer = tf.io.TFRecordWriter('100_500_1000_train.tfrecords')


# In[44]:


for index,name in enumerate(classes):
    class_path = cwd+name+'/';
    for img_name in os.listdir(class_path):
        img_path = class_path+img_name
        img = Image.open(img_path)
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
writer.close()


# # ReadMyOwnData
# ## 讀取tfrecords檔案

# In[45]:


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label


# In[46]:


#訓練次數
batch_size = 50


# # 使用卷積神經網路訓練
# ## initial weights

# In[47]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)


# ## initial bias

# In[48]:


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# ## convolution layer

# In[49]:


def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


# ## max_pool layer

# In[50]:


def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')


# In[51]:


x = tf.placeholder(tf.float32, [batch_size,128,128,3])
y_ = tf.placeholder(tf.float32, [batch_size,1])


# ## first convolution and max_pool layer

# In[52]:


W_conv1 = weight_variable([5,5,3,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_4x4(h_conv1)


# ## second convolution and max_pool layer

# In[53]:


W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_4x4(h_conv2)


# ## 變成全連線層，用一個MLP處理

# In[54]:


reshape = tf.reshape(h_pool2,[batch_size, -1])
dim = reshape.get_shape()[1].value
W_fc1 = weight_variable([dim, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)


# ## dropout

# In[55]:


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[56]:


W_fc2 = weight_variable([1024,2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# ## 損失函式及優化演算法

# In[57]:


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[58]:


correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# # 訓練

# In[59]:


image, label = read_and_decode("100_500_1000_train.tfrecords")
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
coord=tf.train.Coordinator()
threads= tf.train.start_queue_runners(coord=coord)


# In[60]:


example = np.zeros((batch_size,128,128,3))
l = np.zeros((batch_size,1))
try:
    for i in range(20000):        
        for epoch in range(batch_size):
            example[epoch], l[epoch] = sess.run([image,label])#在會話中取出image和label           
        train_step.run(feed_dict={x: example, y_: l, keep_prob: 0.5})        
    print(accuracy.eval(feed_dict={x: example, y_: l, keep_prob: 0.5})) #eval函式類似於重新run一遍，驗證，同時修正

except tf.errors.OutOfRangeError:
        print('done!')
finally:
    coord.request_stop()
coord.join(threads)

