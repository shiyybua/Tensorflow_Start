import utils
import numpy as np
import tensorflow as tf

FLAGS = utils.FLAGS
BATCH_SIZE = 64

images, labels = [], []


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    image, label = utils.build_input(
        FLAGS.dataset, FLAGS.train_data_path, BATCH_SIZE, FLAGS.mode)
    print image
    print sess.run(image)
