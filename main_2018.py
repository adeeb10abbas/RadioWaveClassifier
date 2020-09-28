import numpy as np
tr_labels = np.load("/home/arrowhead/dataset_2018/validation/labels.npy")
tr_signals = np.load("/home/arrowhead/dataset_2018/validation/signals.npy")
tr_snrs = np.load("/home/arrowhead/dataset_2018/validation/snrs.npy")

from pyts.approximation import PAA
from pyts.image import GADF, MTF, RecurrencePlots, GASF
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tqdm
import os
plt.ioff()
import csv
import multiprocessing
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

classes = ['32PSK',
 '16APSK',
 '32QAM',
 'FM',
 'GMSK',
 '32APSK',
 'OQPSK', #1
 '8ASK',
 'BPSK',
 '8PSK',
 'AM-SSB-SC',
 '4ASK',
 '16PSK',
 '64APSK',
 '128QAM',
 '128APSK',
 'AM-DSB-SC', #2
 'AM-SSB-WC',
 '64QAM',
 'QPSK',
 '256QAM',
 'AM-DSB-WC',
 'OOK',
 '16QAM']
all_labels = list()
for i in range(len(tr_labels)):
    idx = np.where(tr_labels[i]==1)[0][0]
    all_labels.append(classes[idx])

def runner(j):
    I, Q = map(list, zip(*tr_signals[j]))
    X = [I, Q]
    IMG_SIZE = 256
    
    
    encoder1 = GASF(IMG_SIZE)
    encoder2 = MTF(IMG_SIZE, n_bins=IMG_SIZE//15, quantiles='gaussian')
    encoder3 = GADF(IMG_SIZE)
    Amp = np.sqrt(np.add(np.square(X[0]), np.square([X[1]])))
    r = encoder1.fit_transform(X)
    g = encoder2.fit_transform(X)
    b = encoder3.fit_transform(Amp)
    shape = r.shape
    rgbArray = np.zeros((IMG_SIZE, IMG_SIZE, 3), 'uint8')

    rgbArray[..., 0] = r.reshape(shape)[1, :] * 256
    rgbArray[..., 1] = g.reshape(shape)[0, :] * 256
    rgbArray[..., 2] = b * 256
    label= all_labels[j]
    destination = "./datasets/test/%s/%s/%s"%(str(tr_snrs[j]), label, label+'_'+str(j)+'.png')
    print(destination)
    plt.imsave(destination, rgbArray)
    
def main():
    pool = multiprocessing.Pool(12)
    for _ in tqdm.tqdm(pool.imap_unordered(runner, range(len(tr_signals))), total=len(tr_signals)):
        pass
main()
# runner(0)
#http://127.0.0.1:8888/?token=b07a5ce5ebe034a6ced45b81ed071b39e757b334ae1820cd
