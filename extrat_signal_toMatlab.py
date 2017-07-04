#classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WB-FM']

import numpy as np
import os,random

X_train = np.load('train_set.npy')
Y_train = np.load('train_label.npy')
#X_test = np.load('test_set.npy')
#Y_test = np.load('test_label.npy')
Z_train = np.load('train_snr.npy')
#Z_test = np.load('test_snr.npy')

fid = open('data_sample_AM-SSB_SNRmixed.txt', 'w+')

j = 0
for i in range(len(Y_train)-100000):
    if list(Y_train[i]) == [0,0,1,0,0,0,0,0,0,0,0]:
        print(j)
        j = j+1
        string = '16QAM ' + str(Z_train[i])
        fid.writelines(string)
        fid.writelines('\n')  
        for k in range(128):
            string_i = str(X_train[i][0][k]) + ' ' + str(X_train[i][1][k])
            fid.writelines(string_i)
            fid.writelines('\n')        
        
        

