# coding=utf-8
import numpy as np
import aux

f1=open('./spam_train.csv','r')    #train.csv
#f1=open('./smalltrain.csv','r')
f2=open('./spam_test.csv','r')    #test.csv
fw=open('output.csv','w')
fw.write('id,label\n')      #1~600

#read train data
(X,Y)=aux.readdata(f1)      #(57,4001)
f1.close()

#read test data
X2=aux.readdata2(f2)        #(57,600)
f2.close()


import pandas as pd
from sklearn.linear_model import LogisticRegression
X=X.transpose()
Y=Y.transpose()
Y=np.ravel(Y)
model=LogisticRegression()
model = model.fit(X, Y)
X2=X2.transpose()
predicted = model.predict(X2)
for i in range(0,600):
    fw.write(str(i+1)+','+str(int(predicted[i]))+'\n')
