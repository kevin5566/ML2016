import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
import pickle
import sys
import theano
import h5py
from keras import backend as k

k.set_image_dim_ordering('th')

#labelraw=pickle.load(open('./data/all_label.p','rb'))
labelraw=pickle.load(open(sys.argv[1]+'/all_label.p','rb'))
x_labeldata=[]
y_labeldata=[]
for i in range(0,10):
	for j in range(0,500):
		x_labeldata.append(np.reshape(labelraw[i][j],(3,32,32)))
		y_labeldata.append(i)
del labelraw

x_labeldata=np.asarray(x_labeldata)
y_labeldata=np.asmatrix(y_labeldata)
y_labeldata=np.transpose(y_labeldata)
y_labeldata=np.asarray(y_labeldata)
y_labeldata=np_utils.to_categorical(y_labeldata,10)
print x_labeldata.shape
print y_labeldata.shape

BATCH_SIZE=12
EPOCH=40

model = Sequential()
model.add(Convolution2D( 32,3,3,input_shape=(3,32,32)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D( 32,4,4))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D( 32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(output_dim=512))
model.add(Activation("sigmoid"))
model.add(Dense(output_dim=512))
model.add(Activation("sigmoid"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_labeldata,y_labeldata,batch_size=BATCH_SIZE,nb_epoch=EPOCH)

print 'hi'
'''
y_predict=model.predict_proba(x_unlabeldata,batch_size=32) #45000x10
x_selflearn=[]
y_selflearn=[]
for i in range(0,45000):
	if(np.amax(y_predict[i][:])>0.85):
		x_selflearn.append(x_unlabeldata[i])
		y_selflearn.append(model.predict_classes(x_unlabeldata[i:i+1],batch_size=32))
y_selflearn=np_utils.to_categorical(y_selflearn,10)
model.fit(x_selflearn,y_selflearn,batch_size=BATCH_SIZE,nb_epoch=EPOCH)
'''
from keras.models import load_model
#model.save('model_2.h5')
model.save(sys.argv[2])

unlabelraw=pickle.load(open(sys.argv[1]+'/all_unlabel.p','rb'))
x_unlabeldata=[]
for i in range(0,45000):
	x_unlabeldata.append(np.reshape(unlabelraw[i],(3,32,32)))
del unlabelraw

EPOCH=20

y_predict=model.predict_proba(x_unlabeldata,batch_size=32) #45000x10
x_selflearn=[]
y_selflearn=[]
for i in range(0,45000):
	if(np.amax(y_predict[i][:])>0.85):
		x_selflearn.append(x_unlabeldata[i])
		y_selflearn.append(model.predict_classes(x_unlabeldata[i:i+1],batch_size=32))
y_selflearn=np_utils.to_categorical(y_selflearn,10)
model.fit(x_selflearn,y_selflearn,batch_size=BATCH_SIZE,nb_epoch=EPOCH)

model.save(sys.argv[2])

'''
BATCH_SIZE=20
EPOCH=10

model = Sequential()

model.add(Convolution2D( 16,3,3,input_shape=(3,32,32)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.25))
model.add(Convolution2D( 20,4,4))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D( 20,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(output_dim=500))
model.add(Activation("sigmoid"))
#model.add(Dropout(0.5))
model.add(Dense(output_dim=500))
model.add(Activation("sigmoid"))
#model.add(Dense(output_dim=1000))
#model.add(Activation("sigmoid"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))


model.add(Convolution2D(16, 3, 3,input_shape=(3,32,32)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

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
		y_selflearn.append(model.predict_classes(x_unlabeldata[i:i+1],batch_size=32))
y_selflearn=np_utils.to_categorical(y_selflearn,10)
model.fit(x_selflearn,y_selflearn,batch_size=BATCH_SIZE,nb_epoch=EPOCH)

from keras.models import load_model
#model.save('model_1.h5')
model.save(sys.argv[2])


'''

