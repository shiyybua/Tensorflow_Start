from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../resource/MNIST_data', one_hot=True)

batch_xs, batch_ys = mnist.train.next_batch(10)
print type(batch_ys)