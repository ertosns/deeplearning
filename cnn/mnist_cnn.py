# -*- coding: utf-8 -*-
"""MNIST-CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dsNTbHSt5Sqjh_Rml0VaeRBdqxz8tUpz

#  Import Libraries
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import *
from keras import backend as K

"""# Load MNIST Dataset"""

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("MNIST training set  len ={}".format(len(X_train)))
print("MNIST training set  len ={}".format(len(X_test)))

"""# Scalling Images"""

#--------------------------------------
# Scalling Images by 255
#--------------------------------------
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

"""#  Change Output to Categorical"""

#--------------------------------------
# Change Output to Categorical
#--------------------------------------
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train[:5, :]

"""#  MNIST-MLP

##  Build MLP layers
"""

#--------------------------------------
# Build MLP NN mlp
#--------------------------------------
# define the mlp
mlp = Sequential()
mlp.add(Flatten(input_shape=X_train.shape[1:]))
mlp.add(Dense(512, activation='relu'))
mlp.add(Dropout(0.2))
mlp.add(Dense(512, activation='relu'))
mlp.add(Dropout(0.2))
mlp.add(Dense(10, activation='softmax'))

mlp.compile(loss='categorical_crossentropy', optimizer='adam', 
    metrics=['accuracy'])

# summarize the mlp
mlp.summary()

"""##  Trainning MLP"""

batch_size = 128
epochs = 12
mlp.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

"""## Model Evaluation"""

score = mlp.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""## Model Saving"""

mlp.save('MLP-MNIST.h5')

"""#  MNIST - CNN

##  Prepare Image for CNN
"""

K.image_data_format()

# input image dimensions
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

"""## Build CNN"""

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(num_classes, activation='softmax'))

cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
cnn.summary()

"""## Training CNN"""

cnn.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

"""## Model Evaluation"""

score = cnn.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""##  Model Saving"""

cnn.save('CNN-MNIST.h5')

