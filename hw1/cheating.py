# coding=utf-8
import numpy as np
import sys
import math
import mlleehw1_aux

f1=open('./train.csv','r')    #train.csv
f2=open('./test_X.csv','r')    #test.csv
fw=open('kaggle_best.csv','w')
fw.write('id,value\n')

### read raw train data ###
f1.readline()
Xf=np.zeros((17,1))
Xf=mlleehw1_aux.readdata(f1)
f1.close()          # Xf.shape   17 x 5760
### read raw train data done ###

### extract train feature ###
N=7   # N belong to [2,9]
X=0
Y=0
(X,Y)=mlleehw1_aux.trainFeatureExtract(N,Xf)
### train feature extract done ###

### read raw test data ###
X2f=np.zeros((17,1))
X2f=mlleehw1_aux.readdata2(f2)
f2.close()    #X2f.shape   17 x 2160
### read raw test data done ###

###linreg###
#X_sqr=X*X
#X=np.row_stack((X,X_sqr))
tmpW=0
(p,q)=X.shape
Xqq=np.ones((1,q))
X=np.row_stack((X,Xqq))
W=np.dot(Y,np.linalg.pinv(X))
'''
Wgradient=np.add(Y,-1*np.dot(W,X))
Wgradient=Wgradient*-1
Wgradient=Wgradient*X
'''
'''
(p,q)=Wgradient.shape
for j in range(0,q):
    tmpW=tmpW+Wgradient[:,[j]]
Gradlen=np.linalg.norm(tmpW)
print(Gradlen)
############
print(W)
'''
### Test ###
X2=np.zeros((17*N,1))

for i in range(0,240):
    tmp2=X2f[:,[(i+1)*9-N]]
    for j in range(1,N):
        tmp=X2f[:,[(i+1)*9-N+j]]
        tmp2=np.row_stack((tmp2,tmp))
    X2=np.column_stack((X2,tmp2))
    tmp2=[]
X2=X2[:,1:]

#X_sqr=X2*X2
#X2=np.row_stack((X2,X_sqr))

(p,q)=X2.shape
Xqq=np.ones((1,q))
X2=np.row_stack((X2,Xqq))
Ytest=np.dot(W,X2)

for i in range(0,240):
    fw.write('id_'+str(i)+','+str(Ytest[0,i])+'\n')
