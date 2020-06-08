##Code file for fashion mnist

import tensorflow

import numpy as np

from tensorflow.keras.datasets import fashion_mnist

(trainX, trainY), (testX, testY) =fashion_mnist.load_data()

trainX=trainX.reshape(-1,28,28,1)

trainY=trainY.reshape(-1,28,28,1)

# convert from integers to floats

trainX=trainX.astype('float32')
testX=testX.astype('float32')

#normalising the data

trainX=trainX/255
testX=testX/255

# one hot encode target values

from tensorflow.keras.utils import to_categorical

trainY = to_categorical(trainY)
testY = to_categorical(testY)

#creating our cnn model

from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=''adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=2)
  
test_loss,test_acc=model.evaluate(testX,testY)
print('Accuray',test_acc*100)