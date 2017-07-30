# -*- coding: utf-8 -*
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../resource/MNIST_data', one_hot=True)

split_rate = 0.8
length = int(mnist.train.images.shape[0] * split_rate)
x_train, x_test = mnist.train.images[:length], mnist.train.images[length:]
y_train, y_test = mnist.train.labels[:length], mnist.train.labels[length:]

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28,28,1)))
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


x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

if True:
    model.fit(x_train, y_train, epochs=10)
    model.save_weights('./weights')
else:
    model.load_weights('./weights')
    print model.metrics_names
    print model.evaluate(x_test, y_test)