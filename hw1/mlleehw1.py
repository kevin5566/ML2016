# coding=utf-8
import numpy as np
import sys
import math
import mlleehw1_aux

f1=open('./train.csv','r')    #train.csv
f2=open('./test_X.csv','r')    #test.csv
fw=open('linear_regression.csv','w')
fw.write('id,value\n')

### read raw train data ###
f1.readline()
Xf=mlleehw1_aux.readdata(f1)
f1.close()   # Xf.shape   17 x 5760

### read raw test data ###
X2f=mlleehw1_aux.readdata2(f2)
f2.close()    #X2f.shape   17 x 2160

### extract train feature ###
N=3   # N belong to [2,9]
(X,Y)=mlleehw1_aux.trainFeatureExtract(N,Xf)

### normalize data ###
(X,Xmean,Xsd)=mlleehw1_aux.normalizeData(X)

### Start to train ###
(W,b)=mlleehw1_aux.trainGradDes(X,Y,N)
print('Done (>8<)')
### Test ###
Ytest=mlleehw1_aux.testResult(X2f,Xmean,Xsd,W,b,N)

### Output ###
for i in range(0,240):
    fw.write('id_'+str(i)+','+str(Ytest[0,i])+'\n')
