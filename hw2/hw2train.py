# coding=utf-8
import numpy as np
import aux
import math
import sys
f1=open(sys.argv[1],'r')    #train.csv
fw=open(sys.argv[2],'w')

#read train data
(X,Y)=aux.readdata(f1)      #(57,4001)
f1.close()

#normalize data
(X,Xmean,Xsd)=aux.normalizeData(X)

#training
w=aux.trainGrad(X,Y)

#output model
(p,q)=Xmean.shape
for i in range(0,p):
    fw.write(str(Xmean[i,0])+'\n')
(p,q)=Xsd.shape
for i in range(0,p):
    fw.write(str(Xsd[i,0])+'\n')
(p,q)=w.shape
for i in range(0,q):
    fw.write(str(w[0,i])+'\n')
