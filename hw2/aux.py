import numpy as np
import math

def readmodel(f1):
    Xmean = np.zeros((57, 1))
    Xsd = np.zeros((57, 1))
    w = np.zeros((1, 58))
    i = 0
    for line in f1:
        if i < 57:
            Xmean[i, 0] = float(line)
        elif i < 114:
            Xsd[i - 57, 0] = float(line)
        else:
            w[0, i - 114] = float(line)
        i = i + 1

    return (w, Xmean, Xsd)


def sigmoid(z):
    if z > 40:
        return 1.0
    if z < -30:
        return 9e-14
    return 1 / (1 + math.exp(-1 * z))


def trainGrad(X, Y):
    p, q = X.shape
    tmp = np.ones((1, q))
    X = np.row_stack((X, tmp))
    w = np.zeros((1, 58))
    eta = 1
    wGrad = 0
    j = 0
    while True:
        for i in range(0, q):
            xtmp = np.asmatrix(X[:, i])
            wGrad = wGrad + (Y[0, i] - sigmoid(np.dot(w, xtmp)[(0, 0)])) * -1 * xtmp

        wGrad = wGrad / q
        w = w - eta * wGrad.transpose()
        if j % 100 == 0:
            normGrad = np.linalg.norm(wGrad)
            print 'iter: ' + str(j)
            print normGrad
            if normGrad < 0.001:
                break
        wGrad = 0
        j = j + 1

    return w


def test(w, X2, Xmean, Xsd):
    p, q = X2.shape
    for i in range(0, p):
        for j in range(0, q):
            X2[i, j] = X2[i, j] - Xmean[i, 0]
            X2[i, j] = X2[i, j] / Xsd[i, 0]

    tmp = np.ones((1, q))
    X2 = np.row_stack((X2, tmp))
    return np.dot(w, X2)


def readdata(f):
    tmp = []
    Y = []
    X = np.zeros((57, 1))
    for line in f:
        tmp = line.strip().split(',')
        del tmp[0]
        tmp = [ float(i) for i in tmp ]
        Y.append(tmp[-1])
        del tmp[-1]
        tmp = np.asmatrix(tmp)
        X = np.column_stack((X, tmp.transpose()))

    return (X[:, 1:], np.asmatrix(Y))


def readdata2(f):
    tmp = []
    X = np.zeros((57, 1))
    for line in f:
        tmp = line.strip().split(',')
        del tmp[0]
        tmp = [ float(i) for i in tmp ]
        tmp = np.asmatrix(tmp)
        X = np.column_stack((X, tmp.transpose()))

    return X[:, 1:]


def normalizeData(X):
    Xmean = np.mean(X, axis=1, dtype=np.float64)
    Xmean = np.asmatrix(Xmean)
    Xsd = np.std(X, axis=1, dtype=np.float64)
    Xsd = np.asmatrix(Xsd)
    p, q = X.shape
    for i in range(0, p):
        for j in range(0, q):
            X[i, j] = X[i, j] - Xmean[i, 0]
            X[i, j] = X[i, j] / Xsd[i, 0]

    return (X, Xmean, Xsd)

