# -*- coding: utf-8 -*
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.datasets import cifar10

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('../resource/MNIST_data', one_hot=True)
#
# split_rate = 0.8
# length = int(mnist.train.images.shape[0] * split_rate)
# x_train, x_test = mnist.train.images[:length], mnist.train.images[length:]
# y_train, y_test = mnist.train.labels[:length], mnist.train.labels[length:]

batch_size = 32
num_classes = 10
epochs = 200
# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 这个 除以255很关键！
x_train /= 255
x_test /= 255
#
# #
model.fit(x_train, y_train)
# print model.metrics_names
# print model.evaluate(x_train, y_test)