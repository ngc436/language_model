# https://github.com/mukesh-mehta/VDCNN/blob/master/VDCNN_tf.py

import numpy as np
import pandas as pd

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session

#
set_session(session)

from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, Activation, Embedding, Input, Dense, Dropout, Lambda, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau

from metrics import calculate_all_metrics
from vis_tools import *

ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
print('Length of the alphabet')



