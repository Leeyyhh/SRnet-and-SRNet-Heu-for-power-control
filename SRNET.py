'''
@Description:power control DNN
@Author: Yinghan LI
@LastEditTime: 2019-07
'''

##???????projection??????????????????????????????/???????????????????????
# test if a nn can fit a max projection
import tensorflow as tf
import numpy as np
import os
import scipy
import time
import sys
import  scipy.io as io
import copy


class PCNet(object):
    def __init__(self, sess, train):
        # information that can be changed
        self.seeds=1  #random seeds
        self.sigma=1 #noise error
        self.Pmax=1 #the Maximum value of the neural network output
        self.users=3    # number of users
        self.radius=250     # cell radius
        self.shadow=8   #shadowing.
        self.method='SRNET'  # SRNET OR SRNET-HEU
        self.test_batch=10000
        if train:
            self.batch_size = 8000  #8000 training samples per epoch
        else:
            self.batch_size=10000  #10000 test samples
        self.epoch = 180000     #traning epoch
        self.lambda_val = 0.1 #lambda_val choose from 0.1 0.2 0.3 0.4 0.5
        self.layer_size=np.array([360, 360, 360, 360,6]) #layer size of the Network, 6 is the size of outputlayer


        #default information
        self.sess=sess
        self.Edge_range = [0, 3]  # test cell-edge region
        self.req_rate=tf.reshape(tf.ones([self.users])* self.lambda_val,[1,self.users,1])*tf.ones([self.batch_size,self.users,1])
        tf.set_random_seed(self.seeds)
        self.rng = np.random.RandomState(self.seeds)
        self.isTrain = train
        self.model_folder = self.generate_name()
        self.hidden_drop = tf.placeholder(tf.float32)
        self.q = tf.pow(2.0, self.req_rate) - 1

        #training process
        if train:
            name = format(
                './train/trainset_lambda%d' % (self.lambda_val*10))
            self.Hall = scipy.io.loadmat(name)['H']
            sizeH = np.size(self.Hall, axis=0)
            self.Hall = tf.sqrt(self.Hall)
            Hchosse = tf.random_crop(self.Hall[:sizeH - 10000, :, :], [self.batch_size, self.users, self.users])
            self.Hvalid = self.Hall[sizeH - self.batch_size:, :, :],
            self.trorval = tf.placeholder(tf.bool)
            self.H_input = tf.cond(self.trorval, lambda: Hchosse, lambda: self.Hvalid)
            self.H_input = tf.cast(self.H_input, tf.float32)
        else:
            self.H_input = tf.placeholder(tf.float32, [self.batch_size, self.users, self.users])
        self.Net_input = tf.reshape(self.H_input, [self.batch_size, self.users * self.users])
        self.Net_output =self.creat_SRNet()
        self.P_output = tf.reshape(self.Net_output[:, 0:self.users],[-1, self.users, 1])*self.Pmax

        self.d_output = tf.reshape(self.Net_output[:, self.users:self.users * 2], [-1, self.users, 1])
        self.B=self.genete_B()
        self.pointC=self.genete_pointC(self.B)
        self.pointE=self.generate_pointE(self.B,self.pointC)
        self.sum_rate,self.per_rate=self.calsum_rate(self.H_input,self.pointE,self.batch_size,self.users,self.sigma)
        # the number of unfeasible outputs
        self.unfeasible=self.check_feasible()
        self.loss=-self.sum_rate
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            with tf.variable_scope('opt'):
                # self.optimizer =  tf.train.AdamOptimizer(s_lr).minimize(self.loss, var_list=[var for var in tf.trainable_variables() ])
                self.train_op = tf.train.AdamOptimizer().minimize(self.loss,
                                                                  var_list=[var for var in tf.trainable_variables()])

        # vars = [var for var in tf.all_variables()]
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)  # 存变量
    def check_feasible(self):
        #Check whether the requirements of ouput P are met
        #label1 minimum sum rate requirement
        #label 2 requeirement-The power P is less than Pmax
        #label 2 requeirement-The power P is more than 0
        label1 = tf.reshape(tf.reduce_sum(
            tf.cast(tf.greater(tf.reshape(self.req_rate, [ self.batch_size, self.users]), self.per_rate + 1e-05), dtype=tf.int16),
            axis=1), [ self.batch_size])
        label2 = tf.reshape(tf.reduce_sum(tf.cast(tf.greater(self.pointE, self.Pmax+ 1e-05), dtype=tf.int16), axis=1), [ self.batch_size])
        label3 = tf.reshape(tf.reduce_sum(tf.cast(tf.greater(0.0, self.pointE + 1e-05), dtype=tf.int16), axis=1), [ self.batch_size])
        label = label1 + label2 + label3
        return label

    def calsum_rate(self,H, P_output, numh, usernum, sigma):
        h_square = tf.square(tf.reshape(tf.abs(H), (numh, usernum, usernum)))
        h_square = tf.cast(h_square, tf.float32)
        P_output = tf.cast(P_output, tf.float32)
        P_output_tans = tf.multiply(tf.ones((numh, usernum, usernum), dtype=tf.float32),
                                    tf.reshape(P_output, [numh, usernum, -1]))
        P_output_mul = tf.multiply(h_square, P_output_tans)
        mask = tf.eye(usernum)
        valid_rx_power = tf.reduce_sum(tf.multiply(P_output_mul, mask), axis=1)
        eye = tf.subtract(tf.ones([numh, usernum, usernum]),
                          tf.reshape(tf.eye(usernum, dtype=tf.float32), [-1, usernum, usernum]))
        P_output_down = tf.reduce_sum(tf.multiply(P_output_mul, eye), axis=1) + sigma
        # sum_rate0 = tf.log(tf.divide(P_output_up, P_output_down)) / tf.log(2.0)
        sum_rate0 = tf.log(1 + tf.divide(valid_rx_power, P_output_down)) / tf.log(2.0)
        sumrate = tf.reduce_mean(tf.reduce_sum(sum_rate0, axis=1))
        return sumrate, sum_rate0

    def generate_test_data(self):
        name = format(
            './test/testset_lambda%d' % (self.lambda_val*10))

        H = np.sqrt(scipy.io.loadmat(name)['Sample_H'])

        return H

    def genete_B(self):
        H2_input = self.H_input * self.H_input
        H2trans = tf.transpose(H2_input, perm=[0, 2, 1])
        diagH2 = H2_input * tf.eye(self.users)
        B = diagH2 - (H2trans - diagH2) * (self.q)

        return B
    def genete_pointC(self,B):
        Btrans = tf.transpose(B, perm=[0, 2, 1])
        sqrt_diaBBT = tf.sqrt(
            B @ Btrans * tf.reshape(tf.eye(self.users, dtype=tf.float32), [-1, self.users, self.users]))
        dmax = tf.reduce_min((self.Pmax - tf.matrix_inverse(B) @ self.q) / (
                    tf.matrix_inverse(B) @ sqrt_diaBBT @ tf.ones_like(self.P_output)), axis=[1, 2], keep_dims=True)

        pointC = tf.matrix_inverse(B) @ self.q * self.sigma + tf.matrix_inverse(B) @ sqrt_diaBBT @ ( self.d_output * dmax)
        return pointC

    def generate_pointE(self,B,point_C):
        resi_all = self.q-B@self.P_output
        x1 = tf.where(resi_all < 0, x=tf.zeros_like(resi_all), y=resi_all)
        yn = B @ (point_C - self.P_output)
        result = tf.reduce_max(tf.where(yn <= 0, x=tf.zeros_like(yn), y=tf.divide(x1 +1e-11, yn +1e-11)),axis=1)
        point_D = self.P_output + tf.reshape(result, [self.batch_size, 1, 1]) * (point_C - self.P_output )
        point_D=tf.reshape(point_D,[self.batch_size,self.users])
        point_E = tf.divide(point_D,  tf.reduce_max(point_D, axis=1, keep_dims=True))
        return point_E

    def generate_name(self):
        layername = ''
        for i in range(np.size(self.layer_size)):
            layername = layername + format('%d_' % self.layer_size[i])

        model_location = format(
                './SRNET_R%d_Edge%d_%d/' % (self.radius, self.Edge_range[0], self.Edge_range[1])) +  '/' + layername+format('lambda%d'%(self.lambda_val*10))


        if not os.path.exists(model_location):
            os.makedirs(model_location)
        return model_location

    def save_network_to_file(self,saver,i):

        model_name = format("%s/model.ckpt" % (self.model_folder))
        saver.save(self.sess, model_name,global_step=i)
        print("Save the network to a file.\n")
    def restore_network(self):
        save_dict = [var for var in tf.global_variables()  ]

        ckpt = tf.train.get_checkpoint_state(self.model_folder)
        print(self.model_folder)
        saver = tf.compat.v1.train.Saver(save_dict)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            print('sucess load all model')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('can not restore the model')
    def creat_SRNet(self):
        with tf.variable_scope('srnet'):
            n=self.Net_input
            for i in range(np.size(self.layer_size)-1):
                    n = tf.layers.dense(n, self.layer_size[i],
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer())
                    n = tf.layers.batch_normalization(n, training=self.isTrain)
                    n = tf.nn.relu(n)
                    n=tf.nn.dropout(n,self.hidden_drop)

            n = tf.layers.dense(n, self.layer_size[i+1],
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            n = tf.nn.sigmoid(n)
        return n

    def save_python_output(self):
        # generate txt of the output data
        class Logger(object):
            def __init__(self, filename="Default.log"):
                self.terminal = sys.stdout  # ????????????terminal
                self.log = open(filename, "a")

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)

            def flush(self):
                pass
        if self.isTrain:
            sys.stdout = Logger(self.model_folder + format('/train_model.txt' ))
        else:
            sys.stdout = Logger(self.model_folder + format('/test_model.txt' ))

    def train(self,ifrestore):
        self.sess.run(tf.global_variables_initializer())
        self.save_python_output()
        if ifrestore:
            self.restore_network()
        max_valid_sum=0

        # train projection network first
        for epoch in range(self.epoch):

            _,train_sumrate=self.sess.run([self.train_op,self.sum_rate],
                                          feed_dict={self.trorval: True,
                                               self.hidden_drop: np.array([1])})
            if epoch % 10000== 0 :
                valid_sumrate, label_val =self.validation()
                print('step: %d validation sumrate: %f train sumrate:%f unfeasible:%d' %(epoch , valid_sumrate, train_sumrate,np.sum(label_val)))
                # # # print(label_val)
                if valid_sumrate<(max_valid_sum-0.1):
                    self.save_network_to_file(self.saver,epoch)
                    break
            elif (epoch+1) % self.epoch== 0 and epoch>1:
                self.save_network_to_file(self.saver, epoch)

    def validation(self):
        sumrate, label_val = self.sess.run(
            [self.sum_rate, self.unfeasible],
            feed_dict={
                self.trorval: False, self.hidden_drop: np.array([1])
            })

        return sumrate, label_val

    def test_sample(self):
        self.save_python_output()
        self.sess.run(tf.global_variables_initializer())

        self.restore_network()
        time_start=time.time()
        input_h =self.generate_test_data()[:self.test_batch,:,:]
        c, label_val,pout = self.sess.run(
            [self.loss, self.unfeasible,self.P_output],
            feed_dict={
                self.H_input: input_h, self.hidden_drop: np.array([1])
            })
        time_end=time.time()
        print(c,np.sum(label_val),'time:',time_end-time_start)

