from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import numpy as np
import pickle

def map_classes(way_classes):
	m = {}
	y = np.zeros((len(way_classes,)))
	uc = np.unique(way_classes)
	for i in range(0, len(uc)):
		m[uc[i]] = i
	for i in range(0, len(way_classes)):
		y[i] = m[way_classes[i]]
	return y, m

def loadCifar(file):
	with open(file,'rb') as foo:
		dict = pickle.load(foo)
	dictData = dict
	X = dictData['data']
	y = np.asarray(dictData['labels'])
	return X, y
#putting on data in a memory in two arrays
#this is data for to train
X1, y1 = loadCifar('data_batch_1')
X2, y2 = loadCifar('data_batch_2')

Xtr = np.concatenate([X1, X2], axis=0)
ytr = np.concatenate([y1, y2])

Xte, yte = loadCifar('test_batch')

ytr, mtr = map_classes(ytr)
yte, mte = map_classes(yte)

widthImage = 32
heighImage = 32
batch_size = 30000
num_classes = 10
epochs = 20

Xtr = Xtr.astype('float32')
Xte = Xte.astype('float32')
Xtr /= 255
Xte /= 255

print(Xtr.shape[0], 'Train')
print(Xte.shape[0], 'Test')

#convert class vectors to binary class matrices
ytr = keras.utils.to_categorical(ytr, num_classes)
yte = keras.utils.to_categorical(yte, num_classes)

model = Sequential();
model.add(Conv(512, activation='relu', input_shape=(widthImage*heighImage*3,)))
model.add(Dropout(0.2))
model.add(Conv(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Conv(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(Xtr, ytr, batch_size=batch_size, epochs=epochs, verbose = 1, validation_data=(Xte, yte))

score = model.evaluate(Xte, yte, verbose=0)

print('Test loss:', score[0])
print('Test accuracy ', score[1])