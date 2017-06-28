
#
#    
#    dataset from radioml.com (2016//10.a ver.)cannot be encoded under python 3.5
#    here use cPickle(python 2.7), Numpy lib. to read and write it into data-stream(.npy) 
#
#    X_train, X_test dataset is 3-D Tensor(ndarray) with the shape of (110000,2,128): 110000 samples, 2-d(I and Q), 128 dots per sample.
#    this set includes 11 modulation ways(8d 3a).  
#    label set(Y_train, Y_test) consist of (11-D) (0-1) vectors.
#    Class: 
#         ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WB-FM']
#

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD



from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
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
    classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WB-FM']
    X_train = np.load('train_set.npy')
    Y_train = np.load('train_label.npy')
    X_test = np.load('test_set.npy')
    Y_test = np.load('test_label.npy')
    
    X_train_smallsize = X_train[0:10000]
    Y_train_smallsize = Y_train[0:10000]
    X_test_smallsize = X_test[0:10000]
    Y_test_smallsize = Y_test[0:10000]
    
    print(X_train_smallsize.shape)
    in_shap = list(X_train_smallsize.shape[1:])
    
    dr = 0.5
    DNN_model = Sequential()
    # example
    #model = models.Sequential()
    #model.add(Reshape(in_shap+[1], input_shape=in_shap))
    #model.add(ZeroPadding2D((0, 2)))
    #model.add(Convolution2D(256, 1, 3, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
    #model.add(Dropout(dr))
    #model.add(ZeroPadding2D((0, 2)))
    #model.add(Convolution2D(80, 2, 3, border_mode="valid", activation="relu", name="conv2", init='glorot_uniform'))
    #model.add(Dropout(dr))
    #model.add(Flatten())
    #model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
    #model.add(Dropout(dr))
    #model.add(Dense( len(classes), init='he_normal', name="dense2" ))
    #model.add(Activation('softmax'))
    #model.add(Reshape([len(classes)]))
    #model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    #model.summary()
    # Set up some params 
    #nb_epoch = 100     # number of epochs to train on
    #batch_size = 1024  # training batch size
    #history = model.fit(X_train,
        #Y_train,
        #batch_size=batch_size,
        #nb_epoch=nb_epoch,
        #verbose=2,
        #validation_data=(X_test, Y_test))
    
    #score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
    #print (score)    
    
    
    
    # self
    # shape: [N, 2, 128, 1]
    DNN_model.add(Reshape(in_shap+[1], input_shape=in_shap))
    
    DNN_model.add(ZeroPadding2D((1,1)))
    DNN_model.add(Conv2D(64, (1,3), data_format='channels_last',padding='valid', 
                                activation="relu", name="conv1", init='glorot_uniform'))
    #DNN_model.add(MaxPooling2D())
    DNN_model.add(Dropout(0.5))
    
    DNN_model.add(ZeroPadding2D((1,1)))
    DNN_model.add(Conv2D(16, (2,3), data_format='channels_last',padding='valid', 
                         activation="relu", name="conv2", init='glorot_uniform'))
    #DNN_model.add(MaxPooling2D())
    DNN_model.add(Dropout(0.5))
    
    DNN_model.add(Flatten())
    
    DNN_model.add(Dense(128, activation='relu'))
    DNN_model.add(Dropout(0.5))
    
    DNN_model.add(Dense(len(classes), activation='softmax'))
    DNN_model.add(Reshape([len(classes)]))
    
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    DNN_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    history = DNN_model.fit(X_train, Y_train,
              epochs=100,
              batch_size=1024,
              verbose=2,
              validation_data=(X_test, Y_test))
    score = DNN_model.evaluate(X_test, Y_test, 
                               verbose=0,
                               batch_size=1024)   
    
    # Plot confusion matrix
    test_Y_hat = DNN_model.predict(X_test, batch_size=1024)
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
    
if __name__ == "__main__":
    main()