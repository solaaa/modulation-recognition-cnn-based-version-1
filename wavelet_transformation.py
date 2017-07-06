from pywt import wavedec,downcoef, upcoef
import numpy as np
import matplotlib.pyplot as plt
import os,random
import os
from multiprocessing import Process

def getFactor(x,y):
    # Params:
    # x: I series
    # y: Q series
    # p: poly-fit parameters to decide the level
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
    X_train_wl = X_train
    level = 1
    k = 1
    for i in range(len(X_train)):
        sample = X_train[i]
        #level = int(getFactor(sample[0], sample[1])*10 + 0.9)
        #print(level)
        #print(getFactor(sample[0], sample[1]))
        #if level == 0:
            #level = 1        
        cAi = downcoef('a', sample[0], 'db10', mode='zero',  level=level)
        cAq = downcoef('a', sample[1], 'db10', mode='zero',  level=level)
        Ai = upcoef('a', cAi, 'db10', level=level)
        Aq = upcoef('a', cAq, 'db10', level=level)
        X_train_wl[i][0] = cutCurve(Ai, sample[0].shape[0])
        X_train_wl[i][1] = cutCurve(Aq, sample[1].shape[0])
        pid = os.getpid()
        print('prossID:%d   iterator:%d  level:%d'%(pid, k,level))
        k = k + 1
    print('Shape: X_train:%s'%(str(X_train_wl.shape)))
    np.save('train_set_digital_wl.npy', X_train_wl)    
    
def process2(X_test):
    X_test_wl = X_test
    level = 1
    k = 1
    for i in range(len(X_test)):
        sample = X_test[i]
        #level = int(getFactor(sample[0], sample[1])*10 + 0.9)
        #if level == 0:
            #level = 1
        cAi = downcoef('a', sample[0], 'db10', mode='zero',  level=level)
        cAq = downcoef('a', sample[1], 'db10', mode='zero',  level=level)
        Ai = upcoef('a', cAi, 'db10', level=level)
        Aq = upcoef('a', cAq, 'db10', level=level)
        X_test_wl[i][0] = cutCurve(Ai, sample[0].shape[0])
        X_test_wl[i][1] = cutCurve(Aq, sample[1].shape[0])
        pid = os.getpid()
        print('prossID:%d   iterator:%d    level:%d'%(pid, k, level))
        k = k + 1
    print('Shape: X_test:%s'%(str(X_test_wl.shape)))
    np.save('test_set_digital_wl.npy', X_test_wl)    

        



def cutCurve(series, num=128):
    index = int((series.shape[0] - num)/2)
    oriSer = series[index : index + num]
    return oriSer
def main():
    X_train = np.load('train_set_digital.npy')
    X_test = np.load('test_set_digital.npy')
    p1 = Process(target = process1, args = (X_train,))
    p2 = Process(target = process2, args = (X_test,))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()    

if __name__ == '__main__':
    main()
