import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
def moment(sample, p, q):
    m = np.mean((sample**(p-q)) * (np.conjugate(sample)**q))
    return m

def featureExtraction(X):
    features = np.zeros([len(X), 27])
    for i in range(len(X)):
        print(i)
        sample = X[i][0] + X[i][1]*1j
        sample = sample/np.power(moment(sample, 2,1), 0.5)
        tmpSample = sample

        #'''high-order moments and cumulants'''
        M20 = moment(sample, 2,0)
        M21 = moment(sample, 2,1)
        M22 = moment(sample, 2,2)
        M40 = moment(sample, 4,0)
        M41 = moment(sample, 4,1)
        M42 = moment(sample, 4,2)
        M43 = moment(sample, 4,3)
        M60 = moment(sample, 6,0)
        M61 = moment(sample, 6,1)
        M62 = moment(sample, 6,2)
        M63 = moment(sample, 6,3)
        M80 = moment(sample, 8,0)
        
        C20 = M20
        C21 = M21
        C40 = M40 - 3*(M20**2)
        C41 = M41 - 3*M20*M21
        C42 = M42 - (np.abs(M20))**2 -2*(M21**2)
        C60 = M60 - 15*M20*M40 + 3*(M20**3)
        C61 = M61 - 5*M21*M40 - 10*M20*M41 + 30*(M20**2)*M21
        C62 = M62 - 6*M20*M42 - 8*M21*M41 - M22*M40 + 6*(M20**2)*M22 + 24*(M21**2)*M20
        C63 = M63 - 9*M21*M42 + 12*(M21**3) - 3*M20*M43 - 3*M22*M41 + 18*M20*M21*M22
        C80 = M80 - 35*M40**2 - 28*M60*M20 + 420*M40 - 630*M20**4
        
        sample_ht = np.imag(sample)
        sample = np.real(sample)

        A_n = np.power(np.power(sample, 2) + np.power(sample_ht, 2), 0.5)
        
        phi_n = np.zeros(len(sample))
        pi = 3.1415926
        for j in range(len(phi_n)):
            if sample[j]>0 and sample_ht[j]>0:
                phi_n[j] = np.arctan(sample_ht[j]/sample[j])
            elif sample[j]==0 and sample_ht[j]>0:
                phi_n[j] = pi
            elif sample[j]<0 and sample_ht[j]>0:
                phi_n[j] = pi - np.arctan(sample_ht[j]/sample[j])
            elif sample[j]<0 and sample_ht[j]<0:
                phi_n[j] = pi + np.arctan(sample_ht[j]/sample[j])
            elif sample[j]==0 and sample_ht[j]<0:
                phi_n[j] = pi * 1.5
            elif sample[j]>0 and sample_ht[j]<0:
                phi_n[j] = 2*pi - np.arctan(sample_ht[j]/sample[j])
        Ck = np.zeros(len(sample))
        for j in range(1, len(Ck)):
            if phi_n[j] - phi_n[j-1] > pi:
                Ck[j] = Ck[j-1] - 2*pi
            elif phi_n[j-1] - phi_n[j] > pi:
                Ck[j] = Ck[j-1] + 2*pi
            else:
                Ck[j] = Ck[j-1]
        phi_n_1 = phi_n + Ck
        phi_NL = phi_n_1

        
        f_n = np.zeros(len(sample))
        for j in range(1, len(f_n)):
            f_n[j] = (phi_NL[j] - phi_NL[j-1])*4 / pi
        
        A_cn = A_n/np.mean(A_n) - 1
        
        gamma = np.max(np.power(np.abs(np.fft.fft(A_cn)), 2)) / len(sample)
    
        tmp = A_n/np.mean(A_n)
        index_upcap = np.array([])
        for j in range(len(tmp)):
            if tmp[j]>1:
                index_upcap = np.append(index_upcap, j)
        PH1 = 0
        PH2 = 0
        PH3 = 0
        F1 = 0
        F2 = 0
        f_n = f_n - np.mean(f_n)
        index_upcap = index_upcap.astype(int)
        for j in index_upcap:
            PH1 = PH1 + np.power(phi_NL[j], 2)
            PH2 = PH2+ np.abs(phi_NL[j])
            PH3 = PH3 + phi_NL[j]
            F1 = F1 + np.power(f_n[j], 2)
            F2 = F2 + np.abs(f_n[j])
        ''' 
        segma_ap: the standard deviation of the absolute value of the non-linear component of the instantaneous phase
        segma_dp: the standard deviation of the direct instantaneous phase
        segma_aa: The fourth feature is the standard deviation of the absolute value of the normalized  instantaneous amplitude of the simulated signal
        segma_af:  the standard deviation of the absolute normalized centered instantaneous frequency for the signal segment
        '''
        segma_ap = np.power(PH1/len(index_upcap) - np.power(PH2/len(index_upcap), 2), 0.5)
        segma_dp = np.power(PH1/len(index_upcap) - np.power(PH3/len(index_upcap), 2), 0.5)
        segma_aa = np.power(np.mean(np.power(A_cn, 2)) - np.power(np.mean(np.abs(A_cn)), 2), 0.5)
        segma_af = np.power(F1/len(index_upcap) - np.power(F2/len(index_upcap), 2), 0.5)
        
        A_v = np.power(A_n/np.var(A_n), 0.5) - 1
        
        segma_v = np.power((np.power(np.sum(np.abs(A_v)), 2))/len(A_v) - np.mean(np.power(A_v, 2)) , 0.5)
        
        features[i][0] = gamma
        features[i][1] = segma_ap
        features[i][2] = segma_dp
        features[i][3] = segma_aa
        features[i][4] = segma_af
        features[i][5] = segma_v

        
        sample = tmpSample
        
        '''belta: power of quadrature / power of in-phrase'''
        belta = np.sum(np.power(np.imag(sample), 2)) / np.sum(np.power(np.real(sample), 2))
        features[i][6] = belta
        
        '''citation: Wang L, Ge L D. Algorithm for Blind Identification of OFDM Signal Based on Higher Order Moment'''
        v20 = M42/np.power(M21, 2)
        features[i][7] = v20

        
        #print('v20')
        #print(v20)             
        
        '''X1: the mean value of the signal magnitude'''
        X1 = np.mean(np.abs(A_n))
        features[i][8] = X1
    
        '''X2: the normalized square root value of sum of amplitude of signal samples'''
        X2 = np.power(np.sum(np.abs(A_n)), 0.5) / len(sample)
        features[i][9] = X2        
        
        features[i][10] = C20
        features[i][11] = C21
        features[i][12] = C40
        features[i][13] = C41
        features[i][14] = C42

        features[i][15] = C63
        features[i][16] = C80
        '''kurtosis'''
        factor = sample - np.mean(sample)
        K = np.abs(np.mean(np.power(factor, 4)) / np.power(np.mean(np.power(factor, 2)), 2))
        features[i][17] = K
    
        '''skewness'''
        S = np.abs(np.mean(np.power(factor, 3)) / np.power(np.mean(np.power(factor, 2)), 1.5))
        features[i][18] = S
    
        '''PR: peak-to-rms ratio'''
        PR = np.max(np.power(np.abs(A_n), 2)) / np.mean(np.power(A_n, 2))
        features[i][19] = PR
    
        '''PA: peak-to-average ratio'''
        PA = np.max(np.abs(A_n)) / np.mean(A_n)
        features[i][20] = PA
        
        k1 = np.abs(C42)
        k2 = C63/np.power(C21, 3)
        features[i][21] = k1
        features[i][22] = k2 
        v30 = M63/np.power(M21, 3)
        features[i][23] = v30
        features[i][24] = C60
        features[i][25] = C61
        features[i][26] = C62        
        
    return features

#X_train = np.load('train_set.npy')
#X_test = np.load('test_set.npy')
#X_train_feature = featureExtraction(X_train)

#X_test_feature = featureExtraction(X_test)
#np.save('train_set_feature.npy', X_train_feature)
#np.save('test_set_feature.npy', X_test_feature)

xmajorLocator   = MultipleLocator(150) 
xmajorFormatter = FormatStrFormatter('%1d') 
xminorLocator   = MultipleLocator(30) 

ymajorLocator   = MultipleLocator(5) 
ymajorFormatter = FormatStrFormatter('%1.1f') 
yminorLocator   = MultipleLocator(0.3) 

X = np.load('train_set.npy')
Y = np.load('train_label.npy')
Z = np.load('train_snr.npy')

n = 588
sample = X[n]
print(Y[n])
print(Z[n])
[f, p] = plt.subplots(nrows=5, ncols=2, figsize=(5,10))

for i in range(5):
    for j in range(2):
        p[i][j].xaxis.set_major_locator(xmajorLocator)  
        p[i][j].xaxis.set_major_formatter(xmajorFormatter)  
        
        p[i][j].yaxis.set_major_locator(ymajorLocator)  
        p[i][j].yaxis.set_major_formatter(ymajorFormatter)  
        
        
        p[i][j].xaxis.set_minor_locator(xminorLocator)  
        p[i][j].yaxis.set_minor_locator(yminorLocator)          


p[0][0].plot(sample[0][0:150])
p[0][0].plot(sample[1][0:150])
p[0][0].set_title('BPSK')

n = 592
print(Y[n])
print(Z[n])
sample = X[n]
p[1][0].plot(sample[0][0:150])
p[1][0].plot(sample[1][0:150])
p[1][0].set_title('QPSK')

n = 560
print(Y[n])
print(Z[n])
sample = X[n]
p[2][0].plot(sample[0][0:150])
p[2][0].plot(sample[1][0:150])
p[2][0].set_title('8PSK')

n = 768
print(Y[n])
print(Z[n])
sample = X[n]
p[3][0].plot(sample[0][0:150])
p[3][0].plot(sample[1][0:150])
p[3][0].set_title('QAM16')

n = 630
print(Y[n])
print(Z[n])
sample = X[n]
p[4][0].plot(sample[0][0:150])
p[4][0].plot(sample[1][0:150])
p[4][0].set_title('QAM64')



n = 202
sample = X[n]
print(Y[n])
print(Z[n])

p[0][1].plot(sample[0][0:150])
p[0][1].plot(sample[1][0:150])
p[0][1].set_title('BPSK')

n = 124
print(Y[n])
print(Z[n])
sample = X[n]
p[1][1].plot(sample[0][0:150])
p[1][1].plot(sample[1][0:150])
p[1][1].set_title('QPSK')

n = 309
print(Y[n])
print(Z[n])
sample = X[n]
p[2][1].plot(sample[0][0:150])
p[2][1].plot(sample[1][0:150])
p[2][1].set_title('8PSK')

n = 451
print(Y[n])
print(Z[n])
sample = X[n]
p[3][1].plot(sample[0][0:150])
p[3][1].plot(sample[1][0:150])
p[3][1].set_title('QAM16')

n = 672
print(Y[n])
print(Z[n])
sample = X[n]
p[4][1].plot(sample[0][0:150])
p[4][1].plot(sample[1][0:150])
p[4][1].set_title('QAM64')
plt.show()


