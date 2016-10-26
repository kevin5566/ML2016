# coding=utf-8
import numpy as np
import aux
import math
import sys
f1=open(sys.argv[1],'r')    #model
f2=open(sys.argv[2],'r')    #test.csv
fw=open(sys.argv[3],'w')    #output
fw.write('id,label\n')

#read model
(w,Xmean,Xsd)=aux.readmodel(f1)

#read test data
X2=aux.readdata2(f2)        #(57,600)
f2.close()

#testing
Ytest=aux.test(w,X2,Xmean,Xsd)

#output
ans=0
for i in range(0,600):
    if aux.sigmoid(Ytest[0,i])>0.5:
        ans=1
    else:
        ans=0
    fw.write(str(i+1)+','+str(ans)+'\n')