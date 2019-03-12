
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
import keras
import tensorflow as tf

import csv
# load mnist dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print("wewe")

f = open('train.csv', 'r')
reader = csv.reader(f)


train_array = []
train_label = []

ctr = 0
for row in reader:
    if(ctr):
        train_label.append(int(float(row[1])))
        row.pop(0)
        row.pop(0)
        # print(len(row))
        train_array.append([float(i) for i in row])
        # break
    ctr += 1
f.close()

x_train = np.asarray(train_array)
y_train = np.asarray(train_label)

print(x_train.shape)
print(y_train.shape)


print(x_train[0].shape)

# exit()

# get test
f = open('test.csv', 'r')
reader = csv.reader(f)


test_array = []
test_label = []

ctr = 0
for row in reader:
    if(ctr):
        test_label.append(int(float(row[0])))
        row.pop(0)
        test_array.append([float(i) for i in row])
    ctr += 1
f.close()

x_test = np.asarray(test_array)
y_test = np.asarray(test_label)

print(x_test.shape)
print(y_test.shape)









model_mlp = keras.Sequential([
    # keras.layers.Flatten(input_shape=(300, )),
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(300, )),
    # keras.layers.Dropout(0.3),
    keras.layers.Dense(512, activation=tf.nn.relu),
    # keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model_mlp.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy', 'binary_crossentropy'])

model_mlp.summary()


model_mlp.fit(x_train, y_train, epochs=20, batch_size=25)


res = model_mlp.predict(x_test)
print(res.shape)
print('PREDICT: %d Actual %d' % (res[0], y_train[0]))

with open('sub2.csv', "w",  newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(('id', 'target'))
        for (x, y) in zip(y_test, res):
            if y[0] >= 0.5:
                targ = 1
            else:
                targ = 0
            writer.writerow((x,targ))

exit() 