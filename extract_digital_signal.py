import numpy as np
import os,random
import os
from multiprocessing import Process




def process1(X_train, Y_train, Z_train):
    f = 0
    point = 0
    for i in range(len(Y_train)):
        if point >= len(Y_train):
            break
        if list(Y_train[point]) == [0,1,0,0,0,0,0,0,0,0,0] or list(Y_train[point]) == [0,0,1,0,0,0,0,0,0,0,0] or list(Y_train[point]) == [0,0,0,0,0,0,0,0,0,0,1]:
        #if list(Y_train[point]) == [0,0,0,0,0,0,0,0,0,0,1]:
            Y_train = np.delete(Y_train, point, 0)
            X_train = np.delete(X_train, point, 0)
            Z_train = np.delete(Z_train, point, 0)
            pid = os.getpid()
            print('prossID:%d  delete:%d  iterator:%d  point:%d'%(pid, f, i, point))
            f = f + 1
            point = point - 1
        point = point + 1
    print('Shape: X_train:%s, Y_train:%s, Z_train:%s '%(str(X_train.shape), str(Y_train.shape), str(Z_train.shape)))
    np.save('train_set_digital.npy', X_train)
    np.save('train_label_digital.npy', Y_train)
    np.save('train_snr_digital.npy', Z_train)    

def process2(X_test, Y_test, Z_test):
    f = 0
    point = 0
    for i in range(len(Y_test)):
        if point >= len(Y_test):
            break
        if list(Y_test[point]) == [0,1,0,0,0,0,0,0,0,0,0] or list(Y_test[point]) == [0,0,1,0,0,0,0,0,0,0,0] or list(Y_test[point]) == [0,0,0,0,0,0,0,0,0,0,1]:
        #if list(Y_test[point]) == [0,0,0,0,0,0,0,0,0,0,1]:
            Y_test = np.delete(Y_test, point, 0)
            X_test = np.delete(X_test, point, 0)
            Z_test = np.delete(Z_test, point, 0)
            pid = os.getpid()
            print('prossID:%d  delete:%d  iterator:%d  point:%d'%(pid, f, i, point))
            f = f + 1
            point = point - 1
        point = point + 1
    print('Shape: X_test:%s, Y_test:%s, Z_test:%s '%(str(X_test.shape), str(Y_test.shape), str(Z_test.shape)))
    np.save('test_set_digital.npy', X_test)
    np.save('test_label_digital.npy', Y_test)
    np.save('test_snr_digital.npy', Z_test)    


if __name__ == '__main__':
    X_train = np.load('train_set1.npy')
    Y_train = np.load('train_label1.npy')
    X_test = np.load('test_set1.npy')
    Y_test = np.load('test_label1.npy')
    Z_train = np.load('train_snr1.npy')
    Z_test = np.load('test_snr1.npy')
    classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WB-FM']
    classes_d = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK']
    
    
    p1 = Process(target = process1, args = (X_train, Y_train, Z_train))
    p2 = Process(target = process2, args = (X_test, Y_test, Z_test))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
