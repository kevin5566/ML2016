import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
import pickle
import sys
import theano
from keras import backend as k

k.set_image_dim_ordering('th')

BATCH_SIZE=32
EPOCH=40

labelraw=pickle.load(open(sys.argv[1]+'all_label.p','rb'))
x_labeldata=[]
y_labeldata=[]
for i in range(0,3):
	for j in range(0,50):
		x_labeldata.append(np.reshape(labelraw[i][j],(3,32,32)))
		y_labeldata.append(i)
del labelraw

x_labeldata=np.asarray(x_labeldata)
y_labeldata=np.asmatrix(y_labeldata)
y_labeldata=np.transpose(y_labeldata)
y_labeldata=np.asarray(y_labeldata)
y_labeldata=np_utils.to_categorical(y_labeldata,3)
print x_labeldata.shape
print y_labeldata.shape

model = Sequential()
model.add(Convolution2D(32, 3, 3,input_shape=(3,32,32)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

model.fit(x_labeldata,y_labeldata,batch_size=BATCH_SIZE,nb_epoch=EPOCH)

unlabelraw=pickle.load(open(sys.argv[1]+'all_unlabel.p','rb'))
x_unlabeldata=[]
for i in range(0,45000):
	x_unlabeldata.append(np.reshape(unlabelraw[i],(3,32,32)))
del unlabelraw

y_predict=model.predict_proba(x_unlabeldata,batch_size=32) #45000x10
x_selflearn=[]
y_selflearn=[]
for i in range(0,45000):
	if(np.amax(y_predict[i][:])>0.85):
		x_selflearn.append(x_unlabeldata[i])
		y_selflearn.append(model.predict_classes(x_unlabeldata[i],batch_size=32))
y_selflearn=np_utils.to_categorical(y_selflearn,3)
model.fit(x_selflearn,y_selflearn,batch_size=BATCH_SIZE,nb_epoch=EPOCH)

from keras.models import load_model
#model.save('model_1.h5')
model.save(sys.argv[2])

# returns a compiled model
# identical to the previous one
# model = load_model('model_1.h5')















