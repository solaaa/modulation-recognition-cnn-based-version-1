import pywt 
import numpy as np
import matplotlib.pyplot as plt
import os,random
import os
from multiprocessing import Process
from numba import jit

@jit
def denoiseBySWT(signal):
    level = 1
    wave = 'db6'
    mode = 'soft'
    sample = signal
    coef = pywt.swt(sample, wave, level=level)
    sigmaHat1 = np.median(np.abs(coef[0][1])/0.6745)
    #sigmaHat2 = np.median(np.abs(coef[1][1])/0.6745)
    #sigmaHat3 = np.median(np.abs(coef[2][1])/0.6745)
    
    cap1 = sigmaHat1 * np.sqrt(2*np.log(len(coef[0][1]))) 
    #cap2 = sigmaHat2 * np.sqrt(2*np.log(len(coef[1][1]))) 
    #cap3 = sigmaHat3 * np.sqrt(2*np.log(len(coef[2][1]))) 
    
    coef[0] = list(coef[0])
    #coef[1] = list(coef[1])
    #coef[2] = list(coef[2])
    
    
    coef[0][1] = pywt.threshold(coef[0][1], cap1, mode=mode)
    #coef[1][1] = pywt.threshold(coef[1][1], cap2, mode=mode)
    #coef[2][1] = pywt.threshold(coef[2][1], cap3, mode=mode)
    
    coef[0] = tuple(coef[0])
    #coef[1] = tuple(coef[1])
    #coef[2] = tuple(coef[2])
    
    sampleRec = pywt.iswt(coef, wave)
    return sampleRec

def process1(X_train):
    X_train_wl = X_train
    k = 1
    for i in range(len(X_train)):
        sample = X_train[i]

        X_train_wl[i][0] = denoiseBySWT(sample[0])
        X_train_wl[i][1] = denoiseBySWT(sample[1])
        pid = os.getpid()
        print('prossID:%d   iterator:%d'%(pid, k))
        k = k + 1
    print('Shape: X_train:%s'%(str(X_train_wl.shape)))
    np.save('train_set_swt_lv1.npy', X_train_wl)    
    
def process2(X_test):
    X_test_wl = X_test
    k = 1
    for i in range(len(X_test)):
        sample = X_test[i]      

        X_test_wl[i][0] = denoiseBySWT(sample[0])
        X_test_wl[i][1] = denoiseBySWT(sample[1])
        pid = os.getpid()
        print('prossID:%d   iterator:%d'%(pid, k))
        k = k + 1
    print('Shape: X_test:%s'%(str(X_test_wl.shape)))
    np.save('test_set_swt_lv1.npy', X_test_wl)    

        

def main():
    X_train = np.load('train_set.npy')
    #Z = np.load('train_snr.npy')
    #Y = np.load('train_label.npy')
    X_test = np.load('test_set.npy')
    #X_train2 = np.load('train_set_swt_lv1.npy')
    #X_train2 = np.load('train_set_swt_lv2.npy')
    #X_train4 = np.load('train_set_digital_swt_lv3.npy')
    
    
    p1 = Process(target = process1, args = (X_train,))
    p2 = Process(target = process2, args = (X_test,))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()    
    
   
    
    #num = 9
    #print(Z[num])
    #print(Y[num])
    #sample0 = X_train[num]
    #sample1 = X_train2[num]
    #plt.figure(1)
    #plt.plot(sample0[0])
    #plt.plot(sample1[0], 'r')

    #plt.figure(2)
    #plt.plot(sample0[0], sample0[1])
    #plt.plot(sample1[0], sample1[1], 'r')
    #plt.show()
    
if __name__ == '__main__':
    main()
