# -*- coding: utf-8 -*
import tensorflow as tf
import csv
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.pyplot import savefig

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


IMG_PATH = '../../resource/faces/valid_training.csv'
IMG_PATH = './resource/valid_training.csv'
TEST_PATH = './resource/test.csv'

IMAGE_SAVE_PATH = './resource/images/'

epoch = 300
batch_size = 64
layer_id = 0


def load_images(image_num=None, path=IMG_PATH):
    num = 0
    all_images = []
    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if type(image_num) is int:
                num += 1
                if num > image_num: break
            image = map(float,row['Image'].strip().split(' '))
            image = np.array(image).reshape([96, 96, 1]) / 255
            row['Image'] = image
            all_images.append(row)
    return all_images


def load_header(path=IMG_PATH):
    with open(path, 'r') as csvfile:
        header = csvfile.readline().strip().split(',')
        return header


def get_batch(data, header):
    '''
    :param data:
    :param header: csv的头，最后一个是image， 前面全是数据（label）
    :return:
    '''
    batch = random.sample(data, batch_size)
    image_batch = [x['Image'] for x in batch]   #96*96
    label_batch = []
    for element in batch:
        try:
            # 做归一化
            traning_data = [float(element[colunm]) / 96.0 for colunm in header[:-1]]
        except:
            print element[colunm]
            exit()
        label_batch.append(traning_data)
    return image_batch, label_batch


def init_Weights(shape):
    #  tf.random_normal() is not recommended.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def init_Biases(shape):
    # constant is better.
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_layer(x, W, b, activation_func=tf.nn.relu):
    global layer_id
    layer_id += 1
    with tf.variable_scope("layer%d"%layer_id):
        return activation_func(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep_prob=0.5):
    return tf.nn.dropout(x,keep_prob=keep_prob)


def fc_layer(x,W,b,activation_func=tf.nn.relu):
    global layer_id
    layer_id += 1
    with tf.variable_scope("layer%d" % layer_id):
        return activation_func(tf.nn.xw_plus_b(x,W,b))
        # return activation_func(tf.matmul(x,W) + b)

image = tf.placeholder(shape=[None, 96, 96, 1], dtype=tf.float32)
label = tf.placeholder(shape=[None, 30], dtype=tf.float32)

W_conv1 = init_Weights([5,5,1,32])
b_conv1 = init_Biases([32])
conv1 = conv_layer(image, W_conv1, b_conv1)
pooling1 = max_pool_2x2(conv1)  # to 48 * 48

W_conv2 = init_Weights([5,5,32,64])
b_conv2 = init_Biases([64])
conv2 = conv_layer(pooling1, W_conv2, b_conv2)
pooling2 = max_pool_2x2(conv2)  # 24 * 24


W_fc1 = init_Weights([24*24*64, 1024])
b_fc1 = init_Biases([1024])
pooling1_flat = tf.reshape(pooling2,[-1, 24*24*64])
fc1 = fc_layer(pooling1_flat,W_fc1,b_fc1)
fc1 = dropout(fc1)

W_fc2 = init_Weights([1024, 30])
b_fc2 = init_Biases([30])
# prediction = fc_layer(fc1,W_fc2,b_fc2,tf.nn.softmax)
prediction = tf.nn.sigmoid(tf.nn.xw_plus_b(fc1,W_fc2,b_fc2))

# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=prediction))
loss = tf.reduce_mean(tf.square(label-prediction))
# a high learning rate could not learn anything.
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)


all_images = load_images()
header = load_header()

training_valid_rate = 0.9
training_img = all_images[:int(training_valid_rate*len(all_images))]
test_img = all_images[int(training_valid_rate*len(all_images)):]

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        batch_xs, batch_ys = get_batch(training_img, header)
        # 以下这个形式是错的，因为每次next batch 会随机产生不同的数据
        # batch_xs = mnist.train.next_batch(batch_size)[0]
        # batch_ys = mnist.train.next_batch(batch_size)[1]
        sess.run(train, feed_dict={image:batch_xs,
                                   label:batch_ys})
        if i % 50 == 0:
            batch_xs, batch_ys = get_batch(test_img, header)
            local_loss, local_pred = sess.run([loss, prediction], feed_dict={image:batch_xs, label:batch_ys})
            # print local_pred
            print local_loss
            print '-' * 100

        if i == epoch - 1:
            # batch_xs, batch_ys = get_batch(test_img, header)
            batch_xs = load_images(image_num=200, path=TEST_PATH)
            batch_xs = [x['Image'] for x in batch_xs]
            local_pred = sess.run(prediction, feed_dict={image: batch_xs})

            for index, (x, y) in enumerate(zip(batch_xs, local_pred)):
                X = []
                Y = []
                for e, element in enumerate(y):
                    element *= 96
                    if e % 2 == 0:
                        X.append(element)
                    else:
                        Y.append(element)

                x = x.reshape([96, 96])
                plt.imshow(x)
                plt.scatter(X, Y, c='red')
                savefig(IMAGE_SAVE_PATH + '%d.jpg' % index)
                plt.clf()
