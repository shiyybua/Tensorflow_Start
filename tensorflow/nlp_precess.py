# -*- coding: utf-8 -*
import tensorflow as tf
from tensorflow.python.ops import lookup_ops


def table_lookup():
      src_vocab_file = 'resource/vocab.from'
      UNK_ID = 99999

      src_vocab_table = lookup_ops.index_table_from_file(
            src_vocab_file, default_value=UNK_ID)

      features = tf.constant(["我", "你", "他", "XXIi"])
      with tf.Session() as sess:
            tf.tables_initializer().run()
            print sess.run(src_vocab_table.lookup(features))

def Data_set():
      def batching_func(x):

            return x.padded_batch(
                  3,
                  # The first three entries are the source and target line rows;
                  # these have unknown-length vectors.  The last two entries are
                  # the source and target row sizes; these are scalars.
                  padded_shapes=([None],  # src
                                 [None]),  # tgt_len
                  # Pad the source and target sequences with eos tokens.
                  # (Though notice we don't generally need to do this since
                  # later on we will be masking out calculations past the true sequence.
                  padding_values=(tf.cast('1',tf.string),  # src
                                  tf.cast('2',tf.string)  # tgt_input
                                  ))
      src_dataset = tf.contrib.data.TextLineDataset('resource/vocab.from')
      src_dataset_copy = tf.contrib.data.TextLineDataset('resource/vocab.from')

      # 截断
      src_dataset = src_dataset.map(lambda x: tf.fill([tf.cast(tf.size(x), tf.int32)], x))
      src_dataset_copy = src_dataset_copy.map(lambda x: tf.fill([tf.cast(tf.size(x), tf.int32)], x))


      src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, src_dataset_copy))

      # buffer_size 就是表示取前几个。。。
      src_tgt_dataset = src_tgt_dataset.shuffle(buffer_size=100000)
      # No padding...
      batched_dataset = src_tgt_dataset.batch(8)
      # # padding...
      # batched_dataset = src_tgt_dataset.padded_batch(4)
      # batched_dataset = batching_func(src_tgt_dataset)

      iterator = batched_dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

      with tf.Session() as sess:
            result = sess.run(next_element)
            for batch in result:
                for e in batch:
                      print e
                print '*' * 100


            def key_func(src_dataset_v, src_dataset_copy_v):
                  return tf.size(src_dataset_v) % 2 == 0

            def reduce_func(window_size, windowed_data):
                  return batching_func(windowed_data)
            '''
                  group_by_window:
                      This method maps each consecutive element in this dataset to a key
                      using `key_func` and groups the elements by key. It then applies
                      `reduce_func` to at most `window_size` elements matching the same
                      key. All execpt the final window for each key will contain
                      `window_size` elements; the final window may be smaller.   '''
            group = src_tgt_dataset.group_by_window(
                  key_func=key_func, reduce_func=reduce_func, window_size=8)
            iterator = group.make_initializable_iterator()
            sess.run(iterator.initializer)
            next_element = iterator.get_next()
            for e in sess.run(next_element):
                  print '-'*10
                  for batch in e:
                        for x in batch:
                              print x,
                        print


def paddding():
    dataset = tf.contrib.data.Dataset.range(100)
    dataset = dataset.shuffle(buffer_size=12)
    # 这里相当于做截断, 必须要加
    dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
    padding_value = tf.cast(90, tf.int64)
    dataset = dataset.padded_batch(9, padded_shapes=[20], padding_values=padding_value)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        print sess.run(next_element)

def test():
      def lambd(x):
          x = tf.cast(x, str)
          x = x.split()
          return len(x)

      src_dataset = tf.contrib.data.TextLineDataset('resource/vocab.from')
      src_datasetx = src_dataset.map(lambda x: lambd(x))

      iterator = src_datasetx.make_one_shot_iterator()
      next_element = iterator.get_next()

      with tf.Session() as sess:
            print sess.run(next_element)


Data_set()

