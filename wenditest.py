from keras.models import Sequential
model = Sequential()

from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as k
print(k.image_dim_ordering())
k.set_image_dim_ordering('th')

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


#from keras.optimizers import SGD
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9,nesterov=True))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#import data
import pickle
train = pickle.load(open("all_label.p","r"))
line_number=0
X_train = []
Y_train = []
for line in train:
    for item in line:
        y_train = [0,0,0,0,0,0,0,0,0]
        y_train.insert(line_number,1)
        Y_train.append(y_train)
        X_train.append(item)
    line_number += 1

import numpy
X_train=numpy.asarray(X_train)
Y_train=numpy.asarray(Y_train)
X_train=X_train.reshape(5000,3,32,32)


model.fit(X_train,Y_train, nb_epoch=40,batch_size=20)

#loss_and_metrics = model.evaluate(X_train, Y_train, batch_size=20)
#classed = model.predict_classes(X_train, batch_size=20)
#proba = model.predict_proba(X_train, batch_size=20)

#print('Accuracy of Testing Set:', loss_and_metrics[1] )

from keras.models import load_model
model.save('model02.h5')


