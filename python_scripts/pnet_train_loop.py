'''
##+===========================##
### TIME DISTRIBUTED PointNet ##
##============================##

author: Russell Bate
russellbate@phas.ubc.ca
'''

print()
print('CUDA Environment Info')
## Configuration
#======================================
import os
slurm_tmpdir = os.environ['SLURM_TMPDIR']
slurm_job_id = os.environ['SLURM_JOB_ID']

datapath_prefix = slurm_tmpdir+'/train_data/'
module_path = slurm_tmpdir+'/scripts/'
log_file = ""
train_val_test_ratio = (70,15,15)
EPOCHS = 30
BATCH_SIZE = 500
LEARNING_RATE = 1e-2

print()
print('-- JOB METADATA --')
print('slurm job id: {}'.format(slurm_job_id))
print('batch size: {}'.format(BATCH_SIZE))
print('learning rate: {:2.3f}'.format(LEARNING_RATE))
print('epochs: {}'.format(EPOCHS))

print()
print('loading packages..')
## General imports
#======================================
import numpy as np
import pickle
import time as t
from time import perf_counter as cput
import logging


## local ML Packages
import sys
sys.path.append(module_path)
import deep_set_util as dsu
import pnet_models
from pnet_models import PointNet_gamma


## TensorFlow
#======================================
import tensorflow as tf
from tensorflow import keras

## IMPORTANT ## ====== ## DISABLE EAGER EXECUTION WITH TensorFlow!! ##
print()
print('disabling eager execution..')
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
print()

print("TensorFlow version {}".format(tf.__version__))
print("Keras version {}".format(keras.__version__))
print()


## Load Data
#======================================
print('loading data..')
X = np.load(datapath_prefix+'XR_norm.npy')
Y = np.load(datapath_prefix+'YR_segm_norm.npy')
print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))

Xbatch = np.ndarray.copy(X[:BATCH_SIZE,:,:])
print('Size of batches: {:4.2f} GB'.format(Xbatch.nbytes/1024**3))
del Xbatch

## Set Up Datasets
#======================================
train_num, val_num, test_num = dsu.tvt_num(X, tvt=train_val_test_ratio)
print()
print('Number of training samples: '+str(train_num))
print('Number of validation samples: '+str(val_num))
print('Number of test samples: '+str(test_num))

Y = np.atleast_3d(Y)

X_train = X[:train_num,:,:]
Y_train = Y[:train_num,:]

X_val = X[train_num:train_num+val_num,:,:]
Y_val = Y[train_num:train_num+val_num,:]

X_test = X[train_num+val_num:,:,:]
Y_test = Y[train_num+val_num:,:]

print('Y train: {}'.format(Y_train.shape))
print('X train: {}'.format(X_train.shape))
print()
print()


## Compile Model
#======================================
pnet = PointNet_gamma(shape=(X.shape[1], 4),
    name='PointNet_gamma')

pnet.compile(loss='mse', optimizer=keras.optimizers.Adam(
    learning_rate=LEARNING_RATE))
pnet.summary()
print()
print()


## Train Model
#======================================
t0 = cput()
history = pnet.fit(x=X_train, y=Y_train,
                           epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           validation_data=(X_val, Y_val),
                           verbose=1)
t1 = cput()

print()
print()
print('Time to train: {} (s)'.format(t1-t0))
print('{} (min)'.format((t1-t0)/60))
print('{} (hour)'.format((t1-t0)/3600))
print()


## Predictions
#======================================
t0 = cput()
predictions = pnet.predict(X_test)
t1 = cput()
print('Predictions: {}'.format(predictions.shape))
print()
print('Time to make predictions: {} (s)'.format(t1-t0))
print()

print('saving files..')
with open('results/history'+slurm_job_id+'.pickle', 'wb') as histfile:
    pickle.dump(history, histfile)
np.savez('results/target_preds'+slurm_job_id+'.npz',
    args=(Y_test, predictions), kwds=('target', 'prediction'))

print()
print('saving model..')
pnet.save('models/pnet_'+slurm_job_id)
print()
print('Finished')
print()
