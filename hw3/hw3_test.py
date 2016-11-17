from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as k
import pickle
import numpy as np
import sys
k.set_image_dim_ordering('th')
from keras.models import load_model
model=load_model(sys.argv[2])

import csv
test=pickle.load(open(sys.argv[1]+'/test.p','r'))
ID = test['ID']
X_test = test['data']
X_test = numpy.asarray(X_test)
X_test = X_test.reshape(10000,3,32,32)

Y_test=model.predict_classes(X_test,batch_size=12)
Y_test.shape=(10000,1)

file=open(sys.argv[3],'w')
csv.writer(file).writerow(['ID','class'])
for i in range(0,10000):
    csv.writer(file).writerow([ID[i],Y_test[i][0]])
file.close()

