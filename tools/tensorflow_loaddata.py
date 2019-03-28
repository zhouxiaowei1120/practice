# This is used for loading data from disk for tensorflow
# Reference from https://blog.csdn.net/u012759136/article/details/52232266
# By Dave Zhou
# 11/21/2018

import tensorflow as tf
from PIL import Image
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

class dataloader:
    def __init__(self, datadir, attackdata):
        self.img_filter = ['jpg','png']
        self.data_dir = datadir
        self.attackdata = attackdata
        self.tfRecoderName = self.data_dir+"data.tfrecords"
        self.input_dim = [28,28,1]

    def _tfWriter(self):
        writer = tf.python_io.TFRecordWriter(self.tfRecoderName)
        for root, dirs, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename[-3:] in self.img_filter and filename.split('_')[-1] != 'ori.png':
                    img_path = os.path.join(root,filename)
                    img = Image.open(img_path)
                    #print(np.array(img).shape)
                    img_raw = img.tobytes()              #将图片转化为原生bytes
                    
                    filename_b = bytes(root+filename[:-4],'utf-8')
                    if self.attackdata: # True for that adversarial samples are from fgsm or jsma
                        label = int(img_path.split('/')[-2][0])
                    else:
                        label = int(img_path.split('/')[-3][0])

                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                        'filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename_b]))
                    }))
                    writer.write(example.SerializeToString())  #序列化为字符串
        writer.close()


    def read_and_decode(self):
        self._tfWriter()
        filename = self.tfRecoderName
        #根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer([filename])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                        features={
                                            'label': tf.FixedLenFeature([], tf.int64),
                                            'img_raw' : tf.FixedLenFeature([], tf.string),
                                            'filename' : tf.FixedLenFeature([],tf.string)
                                        })

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, self.input_dim)
        img = tf.cast(img, tf.float32) * (1. / 255)
        #print(img.shape)
        label = tf.cast(features['label'], tf.int32)
        filename = features['filename']

        return img, label, filename

if __name__ == "__main__":
    para = {'dataset':'CAT2000','path':'./CAT2000/trainSet/','n_labels':20}
    datasetloader = dataloader(para)
    img,label,img_mp,classname,filename = datasetloader.read_and_decode()
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=10, capacity=20,
                                                    min_after_dequeue=10,num_threads=4)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(2):
            val, l,val_mp= sess.run([img_batch, label_batch,img_mp_batch])
            #我们也可以根据需要对val， l进行处理
            #l = to_categorical(l, 12) 
            print(val.shape, l,val_mp.shape)
            plt.axis('off')
            plt.subplot(221)
            plt.hist(val.flatten())
            # plt.show()
            plt.subplot(222)
            plt.hist(val_mp.flatten())
            # plt.show()

            plt.subplot(223)
            plt.imshow(val[0,:])
            plt.subplot(224)
            plt.imshow(np.squeeze(val_mp[0,:]),cmap='Greys_r')
            plt.show()
        coord.request_stop()
        coord.join(threads)