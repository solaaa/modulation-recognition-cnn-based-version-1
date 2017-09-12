'''
#
#    X_train, X_test dataset is 3-D Tensor(ndarray) with the shape of (50000,2,512): 50000 samples, 2-d(I and Q), 512 dots per sample.
#    this set includes 5 modulation .  
#    label set(Y_train, Y_test) consist of (5-D) (0-1) vectors.
#    Class: 
#         ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
#
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors

def changeTypeToBin(Y):
    output = np.zeros([len(Y), 5])
    for i in range(len(Y)):
        j = int(Y[i])
        output[i][j] = 1
    return output

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

classes = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
X_train = np.load('train_set_feature.npy')
X_test = np.load('test_set_feature.npy')  

Y_train = np.load('train_label.npy')
Y_test = np.load('test_label.npy')

Y_train_dex = np.load('train_label_dex.npy')
Y_test_dex = np.load('test_label_dex.npy')
Z_train = np.load('train_snr.npy')
Z_test = np.load('test_snr.npy')


n_neighbors = 15

model = neighbors.KNeighborsClassifier(n_neighbors, algorithm='ball_tree')
model.fit(X_train, Y_train_dex)
ret = model.predict(X_test)
ret = np.array(ret)
print('training over')

count = 0
for i in range(len(Y_test_dex)):
    if Y_test_dex[i] == ret[i]:
        count = count + 1
rate = count/len(ret)
print('rate = %f'%(rate))

print(ret)


#Plot confusion matrix
test_Y_hat = changeTypeToBin(ret)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)  


snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = Z_test
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    test_Y_i_hat = changeTypeToBin(test_Y_i_hat)
    
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("SNR: %d .Overall Accuracy: %f"%(snr ,cor / (cor+ncor)))
    acc[snr] = 1.0*cor/(cor+ncor)            
    