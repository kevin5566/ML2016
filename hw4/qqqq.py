import numpy as np
import sys

f=open(sys.argv[1]+'title_StackOverflow.txt','r')

txt=f.read()
txt=txt.split('\n')
del txt[20000]

from sklearn.feature_extraction.text import CountVectorizer

trans=CountVectorizer(token_pattern=r'\b[a-z]{2,10}\b',stop_words='english')
Xtmp=trans.fit_transform(txt)
del txt
Xtmp=Xtmp.toarray().transpose()
Xtmp=Xtmp.tolist()

wordnumraw=len(Xtmp)
X=[]
for i in range(wordnumraw):
    tmp=sum(Xtmp[i])
    if tmp==1 or tmp>1000:
        continue
    else:
        X.append(Xtmp[i])
del Xtmp

X=np.asarray(X).transpose()

print X.shape

'''
X=np.zeros((20000,1))
Xsum=Xtmp.sum(axis=0,dtype=int)
cnt=0
for i in range(len(Xsum)):
    if Xsum[i]==1 or Xsum[i]>1000:
        cnt=cnt+1
        continue
    else:
        X=np.column_stack((X,Xtmp[:,i]))
'''
# trainnnn trainnnnn
'''
from sklearn.cluster import KMeans
clus=KMeans(n_clusters=20).fit(X)


import cPickle
with open('./modelstop.pkl','wb') as fid:
    cPickle.dump(clus,fid)
'''
# model loaddddddddd

import cPickle
with open('./modelstop.pkl','rb') as fid:
    clus=cPickle.load(fid)
del fid
print 'load model done'

'''
f=open('./check_index.csv','r')
fw=open('./submit.csv','w')
f.readline()
print 'output start'
fw.write('ID,Ans\n')

for line in f:
    line=line.strip().split(',')
    if clus.labels_[int(line[1])] == clus.labels_[int(line[2])] and np.dot(X[int(line[1]),:],X[int(line[2]),:].transpose())>0:
        fw.write(line[0]+','+'1\n')
    else:
        fw.write(line[0]+','+'0\n')
'''

f=open(sys.argv[1]+'check_index.csv','r')
fw=open(sys.argv[2],'w')
f.readline()
print 'output start'
fw.write('ID,Ans\n')

for line in f:
    line=line.strip().split(',')
    if clus.labels_[int(line[1])] == clus.labels_[int(line[2])]:
        if np.dot(X[int(line[1]),:],X[int(line[2]),:].transpose())>0:
            fw.write(line[0]+','+'1\n')
        else:
            fw.write(line[0]+','+'0\n')
    else:
        fw.write(line[0]+','+'0\n')

'''
for line in f:
    line=line.strip().split(',')
    if clus.labels_[int(line[1])] == clus.labels_[int(line[2])]:
        fw.write(line[0]+','+'1\n')
    else:
        fw.write(line[0]+','+'0\n')
'''






