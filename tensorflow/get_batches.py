# -*- coding: utf-8 -*
import tensorflow as tf

path = "../resource/faces/football.csv"

def load_header():
    with open(path, 'r') as csvfile:
        header = csvfile.readline().strip().split(',')
        return header

header = load_header()
column_num = len(header)

#######################################################################
filename_queue = tf.train.string_input_producer([path])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
# 这里的默认值决定了对应column的类型，同一个colmn类型要相同
record_defaults = [[0.0] for _ in range(column_num)]


cols = tf.decode_csv(value, record_defaults=record_defaults)

#features = tf.stack(cols)

images, label_batch = tf.train.shuffle_batch(
        [cols[:-1], cols[-1]],
        batch_size=64,
        num_threads=8,
        capacity=1000 + 3 * 64,
        min_after_dequeue=1000)

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  for i in range(2):
    # Retrieve a single instance:
    example = sess.run(images)
    print example[0]

  coord.request_stop()
  coord.join(threads)