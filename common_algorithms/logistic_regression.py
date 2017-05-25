# -*- coding: utf-8 -*
'''
    更多特征工程，见：
    https://www.tensorflow.org/tutorials/wide
'''
# TODO： 结果好像不对。

import tensorflow as tf
import pandas as pd

train_file = './iris_training.csv'
test_file = './iris_test.csv'

COLUMNS = ["first", "second", "setosa", "versicolor", "virginica"]

df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True,header=0)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1,header=0)
LABEL_COLUMN = "virginica"

CONTINUOUS_COLUMNS = ["first", "second", "setosa", "versicolor"]


def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}

  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

first = tf.contrib.layers.real_valued_column("first")
second = tf.contrib.layers.real_valued_column("second")
setosa = tf.contrib.layers.real_valued_column("setosa")
versicolor = tf.contrib.layers.real_valued_column("versicolor")

model_dir = "/tmp"
m = tf.contrib.learn.LinearClassifier(feature_columns=[first, second, setosa, versicolor],
  model_dir=model_dir)

m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
