'''
@Description:power control DNN
@Author: Yinghan LI
@LastEditTime: 2019-07
'''
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
##???????projection???????''/???????????????????????/???????????????????????
# test if a nn can fit a max projection
import tensorflow as tf
import numpy as np
import os
import scipy
import time
import sys
import  scipy.io as io


class PCNet(object):
    def __init__(self, sess, train):
        # information that can be changed
        self.seeds=1  #random seeds
        self.sigma=1 #noise error
        self.Pmax=1 #the Maximum value of the neural network output
        self.percell_use=2
        self.cell_num=7 # number of cells
        self.clip=0 #gradient clip
        self.method_out=1
        self.users=self.cell_num*self.percell_use  # number of users
        self.radius=250     # cell radius
        self.shadow=8   #shadowing.
        self.method='SRNET'  # SRNET OR SRNET-HEU OR SRNET-PROJ
        self.test_batch=10000
        if train:
            self.batch_size = 1024 #train batchsize
        else:
            self.batch_size=self.test_batch #test batchsize
        self.epoch = 200000
        self.lambda_val = 0.5
        self.layer_size = np.array([3000,3000, 3000, self.users * 2 + self.cell_num])  # layer size of the Network, 6 is the size of outputlayer
        self.sess=sess
        self.Edge_range = [0, 3]  # test cell-edge region
        self.req_rate=tf.reshape(tf.ones([self.users])* self.lambda_val,[1,self.users,1])*tf.ones([self.batch_size,self.users,1])
        tf.set_random_seed(self.seeds)
        self.rng = np.random.RandomState(self.seeds)
        self.isTrain = train
        self.model_folder = self.generate_name()
        self.model_folder2 = self.model_folder
        self.hidden_drop = tf.placeholder(tf.float32)
        self.q = tf.pow(2.0, self.req_rate) - 1
        self.H_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.users, self.users])
        self.Net_input = tf.reshape(self.H_input, [self.batch_size, self.users * self.users])
        self.B,self.B1=self.genete_B()
        self.P0=tf.matrix_inverse(self.B1) @ self.q * self.sigma
        P_output,d_output =self.creat_SRNet()
        self.P_output = tf.reshape(P_output,[-1, self.users, 1])*self.Pmax
        self.d_output=tf.reshape(d_output,[-1, self.users, 1])

        self.pointC=self.genete_pointC( self.B)
        self.pointE=self.generate_pointE(self.B,self.pointC)
        self.sum_rate,self.per_rate=self.calsum_rate2(self.H_input,self.pointE,self.batch_size,self.users,self.sigma)
        self.unfeasible=self.check_feasible()
        self.loss=-self.sum_rate
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            with tf.variable_scope('opt'):
                  if self.clip==0:
                    optimizer = tf.train.AdamOptimizer()
                    grads = optimizer.compute_gradients(self.loss)
                    for i, (g, v) in enumerate(grads):
                        if g is not None:
                            grads[i] = (tf.clip_by_norm(g, 10), v)  # 阈值这里设为5
                    self.train_op = optimizer.apply_gradients(grads)
                  else:
                      self.train_op = tf.train.AdamOptimizer().minimize(self.loss, var_list=[var for var in
                                                                                           tf.trainable_variables()])

        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)  # 存变量

    def check_feasible(self):

        label1 = tf.reshape(tf.reduce_sum(tf.cast(tf.greater(tf.reshape(self.req_rate, [ self.batch_size, self.users]), self.per_rate + 1e-05), dtype=tf.int16), axis=1), [ self.batch_size])
        label2 = tf.reshape(tf.reduce_sum(tf.cast(tf.greater(self.pointE, self.Pmax+ 1e-05), dtype=tf.int16), axis=1), [ self.batch_size])
        label4 = tf.reshape(tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(tf.reshape(self.pointE,[self.batch_size,self.cell_num,self.percell_use]),2,keepdims=True), self.Pmax+ 1e-05), dtype=tf.int16), axis=1), [ self.batch_size])
        label3 = tf.reshape(tf.reduce_sum(tf.cast(tf.greater(0.0, self.pointE + 1e-05), dtype=tf.int16), axis=1), [ self.batch_size])
        label = label1 + label2 + label3+label4
        return label



    def calsum_rate2(self, H, P_output, numh, usernum, sigma):
        h_square = tf.square(tf.reshape(H, (numh, usernum, usernum)))
        h_square = tf.cast(h_square, tf.float32)
        P_output =tf.reshape( tf.cast(P_output, tf.float32), [numh, 1,usernum])
        mask = tf.reshape(tf.eye(usernum, dtype=tf.float32), [1, usernum, usernum])
        H_eye=h_square*mask
        H_2=h_square*(1-mask)
        valid_rx_power = P_output@H_eye
        P_output_down = P_output@H_2 + sigma
        sum_rate0 = tf.log(1 + (valid_rx_power/(P_output_down+1e-11))) / tf.log(2.0)
        sum_rate0 = tf.reshape(sum_rate0,[self.batch_size,self.users])
        sumrate = tf.reduce_mean(tf.reduce_sum(sum_rate0, axis=1))
        return sumrate, sum_rate0


    def generate_train_data(self):

        import h5py
        name = format(
            './train/Fin_An2r250Edge_range0_3straint%d_Sample_H.mat' % (self.lambda_val * 10))
        feature = h5py.File(name)  # 读取mat文件
        H =np.sqrt(np.transpose( feature['H'],[2,1,0]) ) # 读取mat文件中所有数据存储到array中
        name = format(
            './traindata_multicell/Fin_An2r250Edge_range0_3straint%d_Sample_H2.mat' % (self.lambda_val * 10))
        feature = h5py.File(name)  # 读取mat文件
        H=np.concatenate([H,np.sqrt(np.transpose( feature['H'],[2,1,0]))],0 ) # 读取mat文件中所有数据存储到array中])

        return H

    def generate_test_data(self):

        import h5py
        name = format(
            './traindata_multicell/test_8%dr250Edge_range0_3_H.mat' % (self.lambda_val*10))
        feature = h5py.File(name)  # 读取mat文件
        H = np.sqrt(np.transpose( feature['Sample_H'],[2,1,0]))   # 读取mat文件中所有数据存储到array中
        return H

    def genete_B(self):
        H2_input = self.H_input * self.H_input
        H2trans = tf.transpose(H2_input, perm=[0, 2, 1])
        diagH2 = H2_input * tf.eye(self.users)
        B1 = diagH2 - (H2trans - diagH2) * (self.q)
        return B1,B1
    def genete_pointC(self,B1):
        B1trans = tf.transpose(B1, perm=[0, 2, 1])
        sqrt_diaBBT = tf.sqrt(
            B1 @ B1trans * tf.reshape(tf.eye(self.users, dtype=tf.float32), [-1, self.users, self.users]))

        cellq=tf.reshape(tf.matrix_inverse(B1) @ self.q, [self.batch_size, self.cell_num, self.percell_use])
        Bcell=tf.reshape(tf.matrix_inverse(B1) @ sqrt_diaBBT @ tf.ones_like(self.P_output), [self.batch_size, self.cell_num, self.percell_use])
        dmax2=tf.reduce_min((self.Pmax -tf.reduce_sum(cellq,axis=2,keepdims=True) )/ (
            tf.reduce_sum( Bcell,axis=2,keepdims=True)), axis=[1, 2], keep_dims=True)
        pointC = tf.matrix_inverse(B1) @ self.q * self.sigma + tf.matrix_inverse(B1) @ sqrt_diaBBT @ (
                self.d_output * dmax2)

        return pointC

    def generate_pointE(self, B, point_C):
        resi_all = self.q - B @ self.P_output
        x1 = tf.where(resi_all < 0, x=tf.zeros_like(resi_all), y=resi_all)
        yn = B @ (point_C - self.P_output)
        result = tf.reduce_max(tf.where(yn <= 0, x=tf.zeros_like(yn), y=tf.divide(x1 + 1e-11, yn + 1e-11)), axis=1)
        point_D = self.P_output + tf.reshape(result, [self.batch_size, 1, 1]) * (point_C - self.P_output)
        point_D = tf.reshape(point_D, [self.batch_size, self.users])
        if self.isTrain==0:
            MAX = tf.reduce_min(
                1/ (tf.reduce_sum(tf.reshape(point_D, [self.batch_size, self.cell_num, self.percell_use]),2, keepdims=True)+1e-11), 1)

            point_D = point_D * MAX

        return point_D
    def generate_name(self):
        layername = ''
        for i in range(np.size(self.layer_size)):
            layername = layername + format('%d_' % self.layer_size[i])

        model_location = format(
            '.clip%d_out%d_R%d_Edge%d_%d/' % (self.clip,self.method_out,self.radius, self.Edge_range[0], self.Edge_range[1])) +  '/' + layername+\
                         format('lambda%d'%(self.lambda_val*10))
        if not os.path.exists(model_location):
            os.makedirs(model_location)
        return model_location

    def parametric_relu(self,_x,num):
        alphas = tf.get_variable(format('alpha%d'%num), _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    def save_network_to_file(self,saver,i):

        model_name = format("%s/model.ckpt" % (self.model_folder))
        saver.save(self.sess, model_name,global_step=i)
        print("Save the network to a file.\n")
    def restore_network(self):
        save_dict = [var for var in tf.global_variables()]
        ckpt = tf.train.get_checkpoint_state(self.model_folder2)
        print(ckpt)
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
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
                    n = tf.compat.v1.layers.batch_normalization(n, training=self.isTrain)
                    n = tf.nn.relu(n)
            n = tf.layers.dense(n, self.layer_size[i + 1], kernel_initializer=tf.contrib.layers.xavier_initializer())
            n = tf.compat.v1.layers.batch_normalization(n, training=self.isTrain)
            P_output = n[:, :self.users + self.cell_num]
            # we make the elements of output greater than the corresponding to the elements in P0 (smaller than P0 must be unfeasible)
            if self.method_out:
                P0_cell=tf.reduce_sum(tf.reshape(self.P0,[self.batch_size,self.cell_num,self.percell_use]),axis=2,keepdims=True)
                P_output=tf.reshape(tf.nn.softmax(tf.reshape(P_output,[self.batch_size*self.cell_num,3])),[-1,self.cell_num,self.percell_use+1])
                P_output=P_output [:,:,:self.percell_use]*(1-P0_cell)
                P_output=tf.reshape(P_output,[self.batch_size,self.users,1])+self.P0
            else:
                P_output=tf.nn.softmax(tf.reshape(P_output,[self.batch_size*self.cell_num,3]))
                P_output=P_output [:,:self.percell_use]

            d_output=tf.nn.sigmoid(n[:,self.users+self.cell_num:])

        return P_output,d_output

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
        H_all= self.generate_train_data()
        max_valid_sum=0
        num_all = np.size(H_all, axis=0)
        Htest=H_all[num_all-self.batch_size:num_all, :, :]

        for epoch in range(self.epoch):
            choosenum =  self.rng .randint(0, num_all-self.batch_size*2 , self.batch_size)
            _,train_sumrate,unfeasible=self.sess.run([self.train_op,self.sum_rate,self.unfeasible],
                                    feed_dict={self.H_input: H_all[choosenum, :, :],
                                                self.hidden_drop: np.array([1])})
            if epoch % 1000== 0 :
                valid_sumrate, label_val =self.validation(Htest)
                print('step: %d validation sumrate: %f train sumrate:%f unfeasible:%d  trainunfeasible:%d' %(epoch , valid_sumrate, train_sumrate,np.sum(label_val),np.sum(unfeasible)))
                if valid_sumrate>max_valid_sum and epoch>40000:
                    max_valid_sum=valid_sumrate
                    self.save_network_to_file(self.saver,epoch)
    def validation(self,Htest):
        sumrate, label_val = self.sess.run(
            [self.sum_rate, self.unfeasible,],
            feed_dict={
                self.H_input: Htest,self.hidden_drop: np.array([1])
            })

        return sumrate, label_val


    def test_sample(self):
        self.save_python_output()
        self.sess.run(tf.global_variables_initializer())
        self.restore_network()
        input_h =self.generate_test_data()
        time_start = time.time()
        pout,sumrate,pe = self.sess.run(
            [ self.P_output,self.sum_rate,self.pointE],
            feed_dict={
                self.H_input: input_h, self.hidden_drop: np.array([1])
            })

        time_end=time.time()
        # io.savemat('./pe.mat', {'pe': pe})
        # io.savemat('./pout.mat', {'pout': pout})
        print(time_end-time_start)
        print(sumrate)



