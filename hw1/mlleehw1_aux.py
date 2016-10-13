import numpy as np

def readdata(f):
    tmp=[]
    X=[]
    Xf=np.zeros((17,1))
    index=0
    
    for line in f:
        index=index+1
        if (index % 18)==11:
            continue
        tmp=line.strip().split(',')
        del tmp[0]
        del tmp[0]
        del tmp[0]
        tmp=[float(i) for i in tmp]
        X.append(tmp)
        if (index % 18)==0:
            X=np.asarray(X)
            Xf=np.column_stack((Xf,X))
            X=[]

    Xf=Xf[:,1:]

    return Xf

def readdata2(f):
    tmp=[]
    X=[]
    Xf=np.zeros((17,1))
    index=0
    
    for line in f:
        index=index+1
        if (index % 18)==11:
            continue
        tmp=line.strip().split(',')
        del tmp[0]
        del tmp[0]
        tmp=[float(i) for i in tmp]
        X.append(tmp)
        if (index % 18)==0:
            X=np.asarray(X)
            Xf=np.column_stack((Xf,X))
            X=[]

    Xf=Xf[:,1:]
    
    return Xf

def trainFeatureExtract(N,Xf):
    X=np.zeros((17*N,1))
    Ytmp=[]
    jumpmonth=0
    for k in range(1,13):
        for i in range((k-1)*480,k*480):
            jumpmonth=jumpmonth+1
            if(jumpmonth%480)==480-N+1:
                jumpmonth=0
                break
            Ytmp.append(Xf[9,N+i])
            tmp2=Xf[:,[i]]
            for j in range(1,N):
                tmp=Xf[:,[i+j]]
                tmp2=np.row_stack((tmp2,tmp))
            X=np.column_stack((X,tmp2))
            tmp2=[]
    X=X[:,1:]               #(17*N, 5760-12*N)
    Y=np.zeros((1,5760-12*N))
    Ytmp=np.asarray(Ytmp)
    Y=np.row_stack((Y,Ytmp))
    Y=Y[1:,:]               #(1, 5760-12*N)
    return (X,Y)

def normalizeData(X):
    Xmean=np.mean(X, axis=1, dtype=np.float64)
    Xmean=np.asmatrix(Xmean)
    Xmean=Xmean.transpose()

    Xsd=np.std(X, axis=1, dtype=np.float64)
    Xsd=np.asmatrix(Xsd)
    Xsd=Xsd.transpose()

    (p,q)=X.shape
    for i in range(0,p):
        for j in range(0,q):
            X[i,j]=X[i,j]-Xmean[i,0]
            X[i,j]=X[i,j]/Xsd[i,0]
    return (X,Xmean,Xsd)

def trainGradDes(X,Y,N):
    b=1
    W=np.ones((1,17*N))
    W=W*0.005
    eta=0.00001
    tmpW=0
    tmpb=0
    i=0
    #lamda=0.00001  #regu
    
    while True:
        i=i+1
        Wgradient=np.add(Y,-1*np.add(b,np.dot(W,X)))
        Wgradient=Wgradient*-1
        bgradient=Wgradient
        Wgradient=Wgradient*X
        
        (p,q)=Wgradient.shape
        for j in range(0,q):
            tmpW=tmpW+Wgradient[:,[j]]
            tmpb=tmpb+bgradient[:,[j]]
        tmpW=tmpW.transpose()
        
        #W=W-eta*tmpW-lamda*W
        W=W-eta*tmpW
        b=b-eta*tmpb[0,0]
        if(i%100==0):
            print('iteration: '+str(i))
            Gradlen=np.linalg.norm(tmpW)
            print(Gradlen)
            if(Gradlen<200):    #report <0.01
                break
        tmpW=0
        tmpb=0
    return(W,b)

def testResult(X2f,Xmean,Xsd,W,b,N):
    X2=np.zeros((17*N,1))

    for i in range(0,240):
        tmp2=X2f[:,[(i+1)*9-N]]
        for j in range(1,N):
            tmp=X2f[:,[(i+1)*9-N+j]]
            tmp2=np.row_stack((tmp2,tmp))
        X2=np.column_stack((X2,tmp2))
        tmp2=[]
    X2=X2[:,1:]

    (p,q)=X2.shape
    for i in range(0,p):
        for j in range(0,q):
            X2[i,j]=X2[i,j]-Xmean[i,0]
            X2[i,j]=X2[i,j]/Xsd[i,0]

    Ytest=np.dot(W,X2)
    Ytest=Ytest+b

    return Ytest
### old version of extract data ###
'''
    N=2
    X=np.zeros((17*N,1))
    Y=np.zeros((q-N,1))
    
    for i in range(0,q-N):
    Y[i,0]=Xf[9,N+i]
    tmp2=Xf[:,[i]]
    for j in range(1,N):
    tmp=Xf[:,[i+j]]
    tmp2=np.row_stack((tmp2,tmp))
    X=np.column_stack((X,tmp2))
    tmp2=[]
    X=X[:,1:]               #(17*N, 5760-N)
    Y=np.transpose(Y)
'''
###################################

'''
    Ytest=Ytest-Y2
    ERR=0
    for i in range(0,240):
    ERR=ERR+math.pow(Ytest[0,i],2)
    print(ERR)
'''

'''
    for i in range(0,240):
    fw.write(str(Ytest[0,i])+','+str(Y2[0,i])+'\n')
    '''
'''
    idname=[]
    sum=0
    index=0
    for line in f:
    index=index+1
    tmp=line.strip().split(',')
    if(index%18)==10:
    idname=tmp[0]
    for i in range(2,10):
    sum=sum+float(tmp[i])
    sum=sum/9
    prdic=str(sum)
    fw.write(idname+','+prdic+'\n')
    sum=0
    f.close()
    '''