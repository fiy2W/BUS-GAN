# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 10
config.TRAIN.lr_init = 1e-2
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 500
config.TRAIN.g_init_model = 'checkpoint/g.npz'
config.TRAIN.g_init_log = 'logs/log_g_init'
config.TRAIN.g_init_samples_dir = 'samples/g_init'

config.TRAIN.ckpt_dir = 'checkpoint/'
config.TRAIN.g_model = 'checkpoint/g_semi.npz'
config.TRAIN.d_model1 = 'checkpoint/d1_semi.npz'
config.TRAIN.d_model2 = 'checkpoint/d2_semi.npz'

config.TRAIN.g_model_semi = 'checkpoint/g_semi.npz'
config.TRAIN.d_model1_semi = 'checkpoint/d1_semi.npz'
config.TRAIN.d_model2_semi = 'checkpoint/d2_semi.npz'

config.TRAIN.gan_log = 'logs/log_gan'
config.TRAIN.gan_samples_dir = 'samples/gan'

## adversarial learning
config.TRAIN.n_epoch = 5000
config.TRAIN.lr_decay = 0.2
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 5)

## train set location
config.TRAIN.img_path = 'data/image'
config.TRAIN.seg_path = 'data/seg'
config.TRAIN.img_list_path = 'data/list_100/list_0.csv'
config.TRAIN.img_list_path2 = 'data/list_100/list_1.csv'


config.VALID = edict()
## test set location
config.VALID.img_path = 'data/image'
config.VALID.seg_path = 'data/seg'
config.VALID.img_list_path = 'data/list/list.csv'

config.VALID.per_print = 5
config.VALID.result_dir = 'valid_result/'
