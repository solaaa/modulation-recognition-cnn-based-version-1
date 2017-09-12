'''
#
#    X_train, X_test dataset is 3-D Tensor(ndarray) with the shape of (50000,2,512): 50000 samples, 2-d(I and Q), 512 dots per sample.
#    this set includes 5 modulation .  
#    label set(Y_train, Y_test) consist of (5-D) (0-1) vectors.
#    Class: 
#         ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
#
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import h5py


from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D, AveragePooling2D
import keras
from keras.regularizers import *
from keras.optimizers import adam
import os,random

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



def main():
    classes = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
    X_train = np.load('train_set_feature.npy')
    X_test = np.load('test_set_feature.npy')  
    
    Y_train = np.load('train_label.npy')
    Y_test = np.load('test_label.npy')
    Z_train = np.load('train_snr.npy')
    Z_test = np.load('test_snr.npy')

    #X_train = X_train[:, 20:24]
    #X_test = X_test[:, 20:24]
    
    in_shap = list(X_train.shape[1:])
    dr = 0.5
    DNN_model = Sequential()

    DNN_model.add(Reshape(in_shap, input_shape=in_shap))

    DNN_model.add(Dense(512, activation='relu', init='he_normal'))
    #DNN_model.add(Dropout(0.5))
    
    #DNN_model.add(Dense(256, activation='relu', init='he_normal'))
    #DNN_model.add(Dropout(0.5))    

    #DNN_model.add(Dense(128, activation='relu', init='he_normal'))
    #DNN_model.add(Dropout(0.5))

    DNN_model.add(Dense(len(classes), activation='softmax', init='he_normal'))
    DNN_model.add(Reshape([len(classes)]))

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #DNN_model.load_weights('model_weights_f_swt_lv2.h5')
    DNN_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    history = DNN_model.fit(X_train, Y_train,
              epochs=100,
              batch_size=512,
              verbose=2,
              #validation_data=None)
              validation_data=(X_test, Y_test))
    
    score = DNN_model.evaluate(X_test, Y_test, 
                                   verbose=0,
                               batch_size=512)   

    #Plot confusion matrix
    test_Y_hat = DNN_model.predict(X_test, batch_size=512)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,X_test.shape[0]):
        j = list(Y_test[i,:]).index(1)
        k = int(np.argmax(test_Y_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plot_confusion_matrix(confnorm, labels=classes)    

    print("score: ")
    print(score)
    DNN_model.save_weights('model_weights_f_swt1.h5')

    snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
    acc = {}
    for snr in snrs:

        # extract classes @ SNR
        test_SNRs = Z_test
        test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

        # estimate classes
        test_Y_i_hat = DNN_model.predict(test_X_i)
        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])
        for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            conf[j,k] = conf[j,k] + 1
        for i in range(0,len(classes)):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        plt.figure()
        plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print ("SNR: %d .Overall Accuracy: %f"%(snr ,cor / (cor+ncor)))
        acc[snr] = 1.0*cor/(cor+ncor)            

if __name__ == "__main__":
    main()