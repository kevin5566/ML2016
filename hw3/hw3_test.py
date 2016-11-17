import pickle
import numpy as np
import sys
import csv

test=pickle.load(open(sys.argv[1]+'test.p','r'))
ID = test['ID']

file=open(sys.argv[3],'w')
csv.writer(file).writerow(['ID','class'])
for i in range(0,10000):
    csv.writer(file).writerow([ID[i],'0'])
file.close()


