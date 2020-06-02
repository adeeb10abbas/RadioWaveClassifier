
import pickle
import numpy as np
Xd = pickle.load(open("RML2016.10a_dict.pkl",'rb'), encoding='latin1')
print(len(Xd[list(Xd.keys())[0]]))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  
            lbl.append((mod,snr))
X = np.vstack(X)
#we have labels for each one of them woowoo
#mods are all the classes are 11
import gc
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GASF, GADF
import os
plt.ioff()
path = os.path.join(os.getcwd())

signals = X
for j in tqdm(range(len(X))):
    wave_type = lbl[j][0]
    #     first = signals[j][:][0]
    #     second = signals[j][:][1]
    values = signals[j][:]
    image_size = 36
    gasf = GASF(image_size)
    X_gasf = gasf.fit_transform(values)
    gadf = GADF(image_size)
    X_gadf = gadf.fit_transform(values)
    plt.figure(figsize=(12, 12))
    # plt.xticks([])
    # plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    wave_id = "%s_%d"%(wave_type, j) 
    fig = plt.imshow(X_gasf[0], cmap='rainbow', origin='lower')
    fig.figure.savefig("%s/dataset/%s/GASF/%s.png"%(path, lbl[j][0], wave_id), bbox_inches = 'tight', pad_inches = 0)
    fig1 = plt.imshow(X_gadf[0], cmap='rainbow', origin='lower')
    fig.figure.savefig("%s/dataset/%s/GADF/%s.png"%(path, lbl[j][0], wave_id), bbox_inches = 'tight', pad_inches = 0)
        # Clear the current axes.
    plt.cla()
        # Clear the current figure.
    plt.clf()
        # Closes all the figure windows.
    plt.close('all')
    gc.collect()
