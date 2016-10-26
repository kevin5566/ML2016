import sys
import pickle
import aux

f1=open(sys.argv[1],'r') #model
f2=open(sys.argv[2],'r') #test.csv
fw=open(sys.argv[3],'w') #output.csv
fw.write('id,label\n')

model=pickle.load(f1)
X2=aux.readdata2(f2)
(p,q)=X2.shape
X2=X2.transpose()
predicted = model.predict(X2)
for i in range(0,q):
    fw.write(str(i+1)+','+str(int(predicted[i]))+'\n')