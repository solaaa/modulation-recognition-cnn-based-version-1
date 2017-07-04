#
# adopt local weighted linear regression(LWLR) to fit a curve.
# simultaneous gradient roughly suggests the SNR.
# with the SNR in high-level(8~20), the factor of LWLR should be low(0.1 or so), which means the fitting curve will be highly similar to the original signal. 
# on the contrary, SNR in low-lever(-4~-16) results in bigger factor(0.25 or so).
# in middle-level (0 or so), 0.18 is a good factor in my experament.
#
# this method is aimed to pre-process the dataset and expect a better results(yet not sure)
# cite: http://lib.csdn.net/article/machinelearning/35027
import numpy as np
import matplotlib.pyplot as plt

import os,random
import os
from multiprocessing import Process


def lwlrPoint(point, xMat, yMat, k):
    # Params:
    #     point: x-axis of current point
    #     xMat: the entire x-axis
    #     yMat: the entire y-axis
    #     k: the factor of LWLR
    #
    col = xMat.shape[0]
    weights = np.zeros([col, col])
    for i in range(col):
        diffMat = point - xMat[i]
        weights[i][i] = np.exp((diffMat**2) / (-2.0 * (k**2)))
    xT = np.dot(xMat, weights)
    xTx = np.dot(xT, xMat.T)
    if float(xTx) == 0:
        print('singular matrix')
    Ty = np.dot(weights, yMat.T)
    theta = (xTx)**(-1) * (np.dot(xMat, Ty))
    yHatPoint = point * theta
    return yHatPoint

def lwlr(xMat, yMat, k):
    # Params:
    #     xMat: the entire x-axis
    #     yMat: the entire y-axis
    #     k: the factor of LWLR
    #
    col = xMat.shape[0]
    yHat = np.zeros([col])
    for i in range(col):
        yHat[i] = lwlrPoint(xMat[i], xMat, yMat, k)
    return yHat

def getFactor(x,y):
    # Params:
    # x: I series
    # y: Q series
    # p: poly-fit parameters to predict the lwlr factor
    #
    s = 0
    for i in range(len(x)-1):
        s = s + np.abs(x[i] - x[i+1]) + np.abs(y[i] - y[i+1])
    p = np.array([-0.234, 1.4549, -3.4053, 3.6493, -1.7388, 0.4138, 0.0727])
    factor = 0
    for i in range(len(p)):
        factor = factor + p[i]*(s**(len(p)-i-1))
    return factor

def process1(X_train):
    X_train_fit = X_train
    # be careful, something happends here
    # here 't' must be [0.0 : 0.1 : 12.7]
    # but it doesn't matter 
    t = np.array(range(128))
    t = t/10
    #
    i = 0
    for sample in X_train:
        factor = getFactor(sample[0], sample[1])
        I_fit = lwlr(t, sample[0], factor)
        Q_fit = lwlr(t, sample[1], factor)
        X_train_fit[i][0] = I_fit
        X_train_fit[i][1] = Q_fit
        pid = os.getpid()
        print('prossID:%d   iterator:%d'%(pid, i))        
        i = i+1
    print('Shape: X_train:%s'%(str(X_train_fit.shape)))
    np.save('train_set_digital_fit.npy', X_train_fit)
    
    
def process2(X_test):
    X_test_fit = X_test
    # be careful, something happends here
    # here 't' must be [0.0 : 0.1 : 12.7]
    # but it doesn't matter 
    t = np.array(range(128))
    t = t/10
    #
    i = 0
    for sample in X_test:
        factor = getFactor(sample[0], sample[1])
        I_fit = lwlr(t, sample[0], factor)
        Q_fit = lwlr(t, sample[1], factor)
        X_test_fit[i][0] = I_fit
        X_test_fit[i][1] = Q_fit
        pid = os.getpid()
        print('prossID:%d   iterator:%d'%(pid, i))        
        i = i+1
    print('Shape: X_test:%s'%(str(X_test_fit.shape)))
    np.save('test_set_digital_fit.npy', X_test_fit)

def main():
    X_train = np.load('train_set_digital.npy')
    Y_train = np.load('train_label_digital.npy')
    X_test = np.load('test_set_digital.npy')
    Y_test = np.load('test_label_digital.npy')
    Z_train = np.load('train_snr_digital.npy')
    Z_test = np.load('test_snr_digital.npy')
    
    X_train_fit = np.load('train_set_digital_fit.npy')
    X_test_fit = np.load('test_set_digital_fit.npy')
    
    print(X_train.shape)
    print(X_train_fit.shape)
    
    print(X_test.shape)
    print(X_test_fit.shape)    
    
    p1 = Process(target = process1, args = (X_train,))
    p2 = Process(target = process2, args = (X_test,))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    
    # be careful, something happends here
    # here 't' must be [0.0 : 0.1 : 12.7]
    # but it doesn't matter 
    #t = np.array(range(128))
    #t = t/10
    #
    #
    # test
    #N = 5
    #sample = X_train[N]
    #sample_fit = X_train_fit[N]
    #factor = getFactor(sample[0], sample[1])
    #print(factor)
    #print(Z_train[N])
    

    
    #plt.figure(1)
    #plt.plot(t, sample[0], 'b')
    #plt.plot(t, sample_fit[0], 'r-')
    #plt.xlabel('time')
    #plt.ylabel('A')
    #plt.show()


if __name__ == '__main__':
    main()