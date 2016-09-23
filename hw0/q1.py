# coding=utf-8
import numpy as np
import sys

col=int(sys.argv[1])
f=open(sys.argv[2],'r')

ans=[]
for line in f:
    tmp=line.strip().split(' ')
    ans.append(float(tmp[col]))
f.close()

ans.sort()
ans.reverse()
strg=""
i=len(ans)
for j in range(0,i):
    strg=strg+str(ans.pop())+","
#print(strg[:-1])
f=open('ans1.txt','w')
f.write(strg[:-1])