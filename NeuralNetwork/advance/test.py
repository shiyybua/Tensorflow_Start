# -*- coding: utf-8 -*
import utils
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

FLAGS = utils.FLAGS
BATCH_SIZE = 64

images, labels = [], []
image_holder = tf.placeholder(tf.float32, [ BATCH_SIZE, 32, 32, 3 ] )
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    image, label = utils.build_input(
        FLAGS.dataset, FLAGS.train_data_path, BATCH_SIZE, FLAGS.mode)

    value = sess.run(image)
    print value[0]


# PAHT = '../../resource/cifar10/cifar-10-batches-py/data_batch_2'
# from skimage.external.tifffile import imshow
# import cPickle as pickle
# import numpy as np

#
#
# def load_CIFAR_batch(filename):
#   """ load single batch of cifar """
#   with open(filename, 'rb') as f:
#     datadict = pickle.load(f)
#     X = datadict['data']
#     Y = datadict['labels']
#     print X.shape
#     X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
#     # X = X.reshape(10000, 32, 32,3 ).astype("float")
#     Y = np.array(Y)
#     print X.shape, Y.shape
#     return X,Y
# X, Y = load_CIFAR_batch(PAHT)
# plt.imshow(X[20])
# plt.show()

# data_dir = '../../resource/cifar10/cifar-10-batches-bin'
# import cifar10, cifar10_input
#
#
# image_holder = tf.placeholder( tf.float32, [ 128, 24, 24, 3 ] )
# images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=128)
#
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
#
# tf.train.start_queue_runners()
#
#
# for _ in range(100):
#   X= sess.run(images_train)
#   image = sess.run(images_train, feed_dict={images_train: X})
#   print image.shape
#   print 'finished'
#   print image[0]
#   plt.imshow(image[100])
#   plt.show()
#
#   break
# sess.close()
