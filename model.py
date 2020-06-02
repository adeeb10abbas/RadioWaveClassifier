from fastai import *
from fastai.vision import *
from fastai.callbacks import CSVLogger
import os
# print(os.getcwd())
path = os.getcwd() + '/dataset_128px_tr'
np.random.seed(43)
data = ImageDataBunch.from_folder(path, train= path+'/train', test= path+'/test', valid_pct=0.2, size=224, resize_method=ResizeMethod.SQUISH, num_workers=8, bs=32)
data.normalize(imagenet_stats)
# print(type(data))
# print(len(data.classes))
learn = cnn_learner(data, models.densenet121, metrics=[accuracy, error_rate], callback_fns=[CSVLogger])
learn.load('stg2_10eps')
# lr = learn.recorder.min_grad_lr
lr = 9.120108393559096e-07
learn.fit_one_cycle(10, lr)
learn.save('stage-3_with_lr')
# data.show_batch(rows=3, figsize=(7,6))
# learn.unfreeze()
# learn.lr_find()
# # learn.recorder.plot()
# learn.fit_one_cycle(60, max_lr=slice(1e-6,0.5e-1))
# learn.save('stage-3')
