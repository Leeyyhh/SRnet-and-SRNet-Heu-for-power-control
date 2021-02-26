

import tensorflow as tf
import SRNET_Multiuser as SRNET

import sys

train=0
__console__ = sys.stdout



if train:
    tf.reset_default_graph()
    istrain=1  #1/0:  train/test
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        max=0
        train_proj=SRNET.PCNet(sess, 1)
        train_proj.train(istrain)
        sess.close()
        sys.stdout = __console__
else:
    import os

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #use cpu
        train_proj = SRNET.PCNet(sess, 0)
        train_proj.test_sample()
        sess.close()
        sys.stdout = __console__
