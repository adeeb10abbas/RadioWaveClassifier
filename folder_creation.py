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

import os
import numpy as np
tr_snrs = np.load("/home/arrowhead/dataset_2018/test/snrs.npy")
# pth_= './datasets/train'
pth = './datasets/test'
# os.makedirs(pth)
SNRS= [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
for i in SNRS:
    os.makedirs(pth+'/'+str(i))
    for j in classes:
        os.makedirs(pth+'/'+str(i)+'/'+j)
