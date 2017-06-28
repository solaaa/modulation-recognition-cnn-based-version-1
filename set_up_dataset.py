#
#  PLZ use python 2.7 ver. to set up this data
#  dataset is from https://radioml.com/ (but now it does not work on 22nd, June, 2017)
#  sola
#
import os,random
import numpy as np
import cPickle
import math
# Load the dataset ...
#  You will need to seperately download or generate this file
Xd = cPickle.load(open("RML2016.10a_dict.dat",'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data
#  into training and test sets of the form we can train/test on 
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(math.floor(n_examples * 0.5))
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

Z_train = (map(lambda x: lbl[x][1], train_idx))
Z_test = (map(lambda x: lbl[x][1], test_idx))

in_shp = list(X_train.shape[1:])
print X_train.shape, in_shp
classes = mods

#
# save whole file
#
np.save('train_set.npy', X_train)
np.save('train_label.npy', Y_train)
np.save('test_set.npy', X_test)
np.save('test_label.npy', Y_test)
np.save('train_snr.npy', Z_train)
np.save('test_snr.npy', Z_test)

#
# extract some samples
#

#label_sample = []
#for i in range(200):
    #t = i+15000
    #label_sample.append(t)
#print label_sample

#label_i = map(lambda x: lbl[x][0], label_sample)
#print label_i
#snr_i = map(lambda x: lbl[x][1], label_sample)

#print snr_i
# fid = open('data_sample_8psk_snr10.txt', 'w+')
# j=0
# for i in label_sample:
    
#     sig_i_I = X[i][0]
#     sig_i_Q = X[i][1]
#     print label_i[j]
#     fid.writelines(label_i[j])
#     fid.writelines(' ')
#     fid.writelines(str(snr_i[j]))
#     fid.writelines('\n')
#     for k in range(128):
#         string_i = str(X[i][0][k]) + ' ' + str(X[i][1][k])
#         fid.writelines(string_i)
#         fid.writelines('\n')
#     print sig_i_I
#     print sig_i_Q
#     j = j+1




