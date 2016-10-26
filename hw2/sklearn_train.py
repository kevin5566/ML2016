  # coding=utf-8
import numpy as np
import aux
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

f1=open(sys.argv[1],'r')    #train.csv
#f1=open('./smalltrain.csv','r')
#f2=open('./spam_test.csv','r')    #test.csv
#fw=open('output.csv','w')
#fw.write('id,label\n')      #1~600

#read train data
(X,Y)=aux.readdata(f1)      #(57,4001)
f1.close()

#read test data
#X2=aux.readdata2(f2)        #(57,600)
#f2.close()

X=X.transpose()
Y=Y.transpose()
Y=np.ravel(Y)
model=LogisticRegression()
model=model.fit(X, Y)

fw=open(sys.argv[2],'w')
pickle.dump(model,fw)
#predicted = model.predict(X2)
#print model.score(X, Y)