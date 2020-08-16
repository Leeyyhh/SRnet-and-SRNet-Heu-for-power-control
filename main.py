import tensorflow as tf
import numpy as np
import os
import SRNET

import sys
train=1
__console__ = sys.stdout
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



sess = tf.InteractiveSession()
for m in [1]:
    if train:
        max=0
        train_proj=SRNET.PCNet(sess, 1)
        train_proj.train(0)
        sess.close()
        sys.stdout = __console__
    else:
        train_proj = SRNET.PCNet(sess, 0)
        train_proj.test_sample()
        sess.close()
        sys.stdout = __console__