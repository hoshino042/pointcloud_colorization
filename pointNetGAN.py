import tensorflow as tf
import os, sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # the directory directly containing current file treated as base dir
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'Utils'))
from utils import *
import time
from ops import *
class point2color():
    def __init__(self,sess, flog,
                 batch_size = 32,
                 num_pts = 2048,
                 L1_lambda = 20,
                 epoch =100,
                 lr_d=0.001,
                 lr_g=0.0001):
        self.isSingleClass = True
        self.batch_size = batch_size
        self.num_pts = num_pts
        self.L1_lambda = L1_lambda
        self.epoch = epoch
        self.sess = sess
        self.flog = flog
        # batch_norm for generator
        self.g_bn_1 = Batch_Norm(name="g_bn_1")
        self.g_bn_2 = Batch_Norm(name="g_bn_2")
        self.g_bn_3 = Batch_Norm(name="g_bn_3")
        self.g_bn_4 = Batch_Norm(name="g_bn_4")
        self.g_bn_5 = Batch_Norm(name="g_bn_5")

        self.g_bn_seg_1 = Batch_Norm(name="g_bn_seg_1")
        self.g_bn_seg_2 = Batch_Norm(name="g_bn_seg_2")
        self.g_bn_seg_3 = Batch_Norm(name="g_bn_seg_3")

        # batch_norm for discriminator
        self.d_bn_1 = Batch_Norm(name="d_bn_1")
        self.d_bn_2 = Batch_Norm(name="d_bn_2")
        self.d_bn_3 = Batch_Norm(name="d_bn_3")
        self.d_bn_4 = Batch_Norm(name="d_bn_4")
        self.d_bn_5 = Batch_Norm(name="d_bn_5")

        self.d_bn_cls_1 = Batch_Norm(name = "d_bn_cls_1")
        self.d_bn_cls_2 = Batch_Norm(name = "d_bn_cls_2")
        self.lr_d = lr_d
        self.lr_g = lr_g

        self.build_model()

    def generator(self, point_cloud, bn_is_train, keep_prob):
        """
        
        :param point_cloud: input raw point cloud , should be of shape (batch_size, num_pts, 3)
        :param bn_is_train: an indicator whether its in training phase or in test phase. if true, parameters of bn would be updated.
        refer http://ruishu.io/2016/12/27/batchnorm/
        :return: generated rgb color list of shape (batch_size, num_pts, 3).
        """
        with tf.variable_scope("generator") as scope:

            input_image = tf.expand_dims(point_cloud, -1)  # batch_size x N x 3 x 1
            out1 = conv2d(input_=input_image, k_w=3, output_dim= 64, scope='g_conv1') # (32, N, 1, 64)
            out1 = tf.nn.relu(self.g_bn_1(out1, phase=bn_is_train))
            self.deadReLU_out1 = tf.summary.histogram("g_out1", out1)

            out2 = conv2d(input_=out1, output_dim=128, scope='g_conv2')  # (32, N, 1 , 128)
            out2 = tf.nn.relu(self.g_bn_2(out2, phase=bn_is_train))
            self.deadReLU_out2 = tf.summary.histogram("g_out2", out2)

            out3 = conv2d(input_=out2, output_dim=128, scope='g_conv3')  # (32, N, 1 , 128)
            out3 = tf.nn.relu(self.g_bn_3(out3, phase=bn_is_train))
            self.deadReLU_out3 = tf.summary.histogram("g_out3", out3)

            out4 = conv2d(input_=out3, output_dim=512, scope='g_conv4')  # (32, N, 1 , 512)
            out4 = tf.nn.relu(self.g_bn_4(out4, phase=bn_is_train))
            self.deadReLU_out4 = tf.summary.histogram("g_out4", out4)

            out5 = conv2d(input_=out4, output_dim=1024, scope='g_conv5')  # (32, N, 1 , 1024)
            out5 = tf.nn.relu(self.g_bn_5(out5, phase=bn_is_train))
            self.deadReLU_out5 = tf.summary.histogram("g_out5", out5)

            out_max = tf.nn.max_pool(out5,
                           ksize=[1, self.num_pts, 1, 1],
                           strides=[1, 2, 2, 1],
                           padding="VALID",
                           name="g_maxpool")
            # segmentation network
            expand = tf.tile(out_max, [1, self.num_pts, 1, 1])  # batch_size x N x 1 x 1024
            concat = tf.concat(axis=3, values=[expand, out1, out2, out3, out4, out5])  # batch_size x N x 1 x (2048 + 256 + 512 + 64)

            net2 = conv2d(input_=concat, output_dim=1024, scope='seg/g_conv1')
            net2 = tf.nn.relu(self.g_bn_seg_1(net2, phase=bn_is_train))
            self.deadReLU_net1 = tf.summary.histogram("g_net1", net2)
            net2 = tf.nn.dropout(net2, keep_prob=keep_prob, name = "seg/g_dp1")

            net2 = conv2d(input_=net2, output_dim=256, scope='seg/g_conv2')  # batch_size x N x 1 x 256
            net2 = tf.nn.relu(self.g_bn_seg_2(net2, phase=bn_is_train))
            self.deadReLU_net2 = tf.summary.histogram("g_net2", net2)
            net2 = tf.nn.dropout(net2, keep_prob=keep_prob, name="seg/g_dp2")

            net2 = conv2d(input_=net2, output_dim=128, scope='seg/g_conv3')  # batch_size x N x 1 x 128
            net2 = tf.nn.relu(self.g_bn_seg_3(net2, phase=bn_is_train))
            self.deadReLU_net3 = tf.summary.histogram("g_net3", net2)
            net2 = conv2d(input_=net2, output_dim=3, scope='seg/g_conv4')  # batch_size x N x 1 x 3
            net2 = tf.nn.tanh(net2)
            color = tf.reshape(net2, [self.batch_size, self.num_pts, 3])

            return color  # (batch_size, N, 3)

    def discriminator(self, point_cloud_color, reuse, bn_is_train):
        """ ConvNet baseline, input is BxNx6 point cloud """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            input_image = tf.expand_dims(point_cloud_color, -1)  # batch_size x N x 6 x 1
            net = conv2d(input_=input_image, output_dim= 64, k_w=6, scope='d_conv1')
            net = tf.nn.relu(self.d_bn_1(net, phase=bn_is_train))

            net = conv2d(input_=net, output_dim= 128, scope='d_conv2')
            net = tf.nn.relu(self.d_bn_2(net, phase=bn_is_train))

            net = conv2d(input_=net, output_dim=128, scope='d_conv3')
            net = tf.nn.relu(self.d_bn_3(net, phase=bn_is_train))

            net = conv2d(input_=net, output_dim=512, scope='d_conv4')
            net = tf.nn.relu(self.d_bn_4(net, phase=bn_is_train))

            net = conv2d(input_=net, output_dim=2048, scope='d_conv5')
            net = tf.nn.relu(self.d_bn_5(net, phase=bn_is_train))

            out_max = tf.nn.max_pool(net,
                                     ksize=[1, self.num_pts, 1, 1],
                                     strides=[1, 2, 2, 1],
                                     padding="VALID",
                                     name="d_maxpool")

            # classification network
            with tf.variable_scope("cls") as scope_cls:
                net = tf.reshape(out_max, [self.batch_size, -1])

                net = dense(net, 256, scope = "d_fc1")
                net = tf.nn.relu(self.d_bn_cls_1(net, phase = bn_is_train))

                net = dense(net, 256, scope = "d_fc2")
                net = tf.nn.relu(self.d_bn_cls_2(net, phase = bn_is_train))

                net = tf.nn.dropout(net, keep_prob=0.7, name="d_dp1")
                net_logit = dense(net, 1, scope = "cls/d_fc3")

                net = tf.nn.sigmoid(net_logit)
                return net_logit, net

    def build_model(self):
        pass

        self.real_pts_color_ph = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_pts, 6), name = "real_pts_color_ph")
        self.bn_is_train = tf.placeholder(dtype=tf.bool, shape=(), name = "bn_is_train")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name = "keep_prob")

        self.real_pts_ts = self.real_pts_color_ph[:, :, :3]
        self.real_color_ts = self.real_pts_color_ph[:, :, 3:]  # tensor of real point cloud color

        self.fake_color_ts = self.generator(self.real_pts_ts, self.bn_is_train, self.keep_prob)  # tensor of fake point cloud color TODO

        self.d_real_color_hist = tf.summary.histogram("real_color", self.real_color_ts)
        self.d_fake_color_hist = tf.summary.histogram("fake_color", self.fake_color_ts)
        self.real_pts_color_ts = tf.concat([self.real_pts_ts,
                                              self.real_color_ts], axis=-1)
        self.fake_pts_color_ts = tf.concat([self.real_pts_ts,
                                              self.fake_color_ts], axis=-1)

        # feedforward pass for discriminator
        self.D_real_logit, self.D_real = self.discriminator(self.real_pts_color_ts, reuse=False, bn_is_train=self.bn_is_train)
        self.D_fake_logit, self.D_fake = self.discriminator(self.fake_pts_color_ts, reuse=True, bn_is_train=self.bn_is_train)

        self.d_real = tf.reduce_mean(self.D_real)
        self.d_fake = tf.reduce_mean(self.D_fake)

        # define d loss and g loss, NS-GAN loss
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logit, labels=tf.ones_like(self.D_real_logit)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logit, labels=tf.zeros_like(self.D_fake_logit)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logit, labels=tf.ones_like(self.D_fake_logit))) \
                      + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_color_ts - self.fake_color_ts))

        # Add losses to summary
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.d_real_proba_sum = tf.summary.scalar("d_real_proba", self.d_real)
        self.d_fake_proba_sum = tf.summary.scalar("d_fake_proba", self.d_fake)

        # merge summary
        self.d_sum = tf.summary.merge([self.d_loss_sum,
                                       self.d_loss_fake_sum,
                                       self.d_loss_real_sum,
                                       self.d_real_color_hist,
                                       self.d_fake_color_hist,
                                       self.d_real_proba_sum,
                                       self.d_fake_proba_sum,
                                       self.deadReLU_net1,
                                       self.deadReLU_net2,
                                       self.deadReLU_net3,
                                       self.deadReLU_out1,
                                       self.deadReLU_out2,
                                       self.deadReLU_out3,
                                       self.deadReLU_out4,
                                       self.deadReLU_out5])
        self.g_sum = tf.summary.merge([self.g_loss_sum])

        # define trainable variables for g and d
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # for item in self.d_vars:
        #     print(item)
        #
        # for item in self.g_vars:
        #     print(item)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr_d, name="d_optim").minimize(self.d_loss, var_list=self.d_vars)
            self.g_optim = tf.train.AdamOptimizer(learning_rate=self.lr_g, name="g_optim").minimize(self.g_loss, var_list=self.g_vars)

    def train(self, train_dir, data, ndata, color, test_data, test_ndata, test_color):
        """
        
        :param train_dir: 
        :param data: a numpy array of shape (N, num_pts, 3)
        :param ndata: a numpy array of shape (N, num_pts, 3)
        :param color: a numpy array of shape (N, num_pts, 3)
        :return: 
        """
        ncolor = (color - 127.5) / 127.5
        test_ncolor = (test_color - 127.5) / 127.5

        ndata_ncolor = np.concatenate((ndata, ncolor), axis=-1)
        test_ndata_ncolor = np.concatenate((test_ndata, test_ncolor), axis=-1)
        assert ndata_ncolor.shape == (data.shape[0], data.shape[1], 6)
        show_all_variables()

        # define saver and summary writer
        self.saver = tf.train.Saver(max_to_keep=200)
        model_dir = os.path.join(train_dir, "model")
        os.mkdir(os.path.join(train_dir, "logs"))
        train_sum_dir = os.path.join(train_dir, "logs","train")
        test_sum_dir = os.path.join(train_dir, "logs","test")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(train_sum_dir):
            os.mkdir(train_sum_dir)
        if not os.path.exists(test_sum_dir):
            os.mkdir(test_sum_dir)
        self.train_writer = tf.summary.FileWriter(train_sum_dir, self.sess.graph)

        # initilize variables
        tf.global_variables_initializer().run()
        self.num_batches = ndata.shape[0] // self.batch_size

        # for visualization during training
        test_masks = np.random.choice(test_data.shape[0], self.batch_size, replace=False)
        batch_test_ndata = test_ndata_ncolor[test_masks]
        batch_test_color = test_color[test_masks]
        batch_test_data = test_data[test_masks]

        train_masks = np.random.choice(data.shape[0], self.batch_size, replace=False)
        batch_train_ndata_ncolor = ndata_ncolor[train_masks]
        batch_train_data = data[train_masks] # no normalized data
        batch_train_color = color[train_masks] # true color

        start_time = time.time()
        for epoch in range(self.epoch):
            # get batch data
            for idx in range(self.num_batches):
                global_step = epoch * self.num_batches + idx + 1
                masks = range(idx * self.batch_size, (idx+1)*self.batch_size)
                batch_ndata_ncolor = ndata_ncolor[masks]
                batch_data = data[masks]
                batch_color = color[masks]
                d_real, d_fake, d_sum_save, d_loss_print, = self.sess.run(
                    [self.d_real, self.d_fake, self.d_sum, self.d_loss], feed_dict={
                        self.real_pts_color_ph: batch_ndata_ncolor,
                        self.bn_is_train: True,
                        self.keep_prob: 0.8})
                if d_real < 0.7:
                     self.sess.run([self.d_optim], feed_dict={self.real_pts_color_ph: batch_ndata_ncolor,
                                                              self.bn_is_train: True,
                                                              self.keep_prob: 0.8})
                self.train_writer.add_summary(d_sum_save, global_step)
                g_sum_save, g_loss_print, _ = self.sess.run([self.g_sum, self.g_loss, self.g_optim], feed_dict={
                                                                    self.real_pts_color_ph: batch_ndata_ncolor,
                                                                    self.bn_is_train: True,
                                                                    self.keep_prob: 0.8})
                self.train_writer.add_summary(g_sum_save, global_step)

                g_sum_save, g_loss_print, _ = self.sess.run([self.g_sum, self.g_loss, self.g_optim], feed_dict={
                                                                    self.real_pts_color_ph: batch_ndata_ncolor,
                                                                    self.bn_is_train: True,
                                                                    self.keep_prob: 0.8})
                self.train_writer.add_summary(g_sum_save, global_step)

                period = time.time()-start_time
                printout(self.flog, "Training! epoch %3d/%3d batch%3d/%3d time: %2dh%2dm%2ds  d_loss: %.4f g_loss: %.4f d_real: %.4f d_fake: %.4f" % (
                    epoch+1, self.epoch, idx+1, self.num_batches,period // 3600, period // 60, period % 60 , d_loss_print, g_loss_print, d_real, d_fake))
            train_fake_color = self.sess.run(self.fake_color_ts, feed_dict={self.real_pts_color_ph: batch_train_ndata_ncolor,
                                                               self.bn_is_train: False, self.keep_prob: 0.8})
            train_fake_color256 = ((train_fake_color + 1) * 127.5).astype(np.int16)
            hex_train_fake_colors = [np_color_to_hex_str(color) for color in train_fake_color256]
            hex_train_true_colors = [np_color_to_hex_str(color) for color in batch_train_color]
            for idx, data in enumerate(batch_train_data):
                fname = os.path.join(train_sum_dir, "epoch{0}_{1}.png".format(epoch, idx))
                display_point(data, hex_train_true_colors[idx], hex_train_fake_colors[idx], fname=fname)
                printout(self.flog, "Saved! {}".format(fname))

            # save model and visualize colorization results on test data
            if epoch % 3 == 0:
                self.saver.save(self.sess, model_dir + "/model", epoch)
                test_fake_color = self.sess.run(self.fake_color_ts, feed_dict={self.real_pts_color_ph: batch_test_ndata,
                                                               self.bn_is_train: False, self.keep_prob: 0.8})
                # reconstruct to 0 - 255
                test_fake_color256 = ((test_fake_color + 1) * 127.5).astype(np.int16)  # (batch_size, N, 3)
                hex_fake_colors = [np_color_to_hex_str(color) for color in test_fake_color256]
                hex_true_colors = [np_color_to_hex_str(color) for color in batch_test_color]
                for idx, data in enumerate(batch_test_data):
                    # fname_ply = os.path.join(test_sum_dir, "epoch{0}_{1}.ply".format(epoch, idx))
                    # save_ply(data, test_fake_color256[idx], fname_ply)
                    fname = os.path.join(test_sum_dir, "epoch{0}_{1}.png".format(epoch, idx))
                    display_point(data, hex_true_colors[idx], hex_fake_colors[idx], fname=fname)
                    printout(self.flog, "Saved! {}".format(fname))


if __name__ == "__main__":
    pc = tf.placeholder(dtype=tf.float32, shape=(32, 2048, 3))
    bn_is_train = tf.placeholder(dtype=tf.bool, shape=())
    with tf.Session() as sess:
        GAN = point2color()
        # color = GAN.generator(pc, bn_is_train)
        # pts_clr = tf.concat([pc, color], axis= -1)
        # logit, net = GAN.discriminator(pts_clr,reuse=False, bn_is_train=bn_is_train)

        # for item in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        #     print(item)
        # for item in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
        #     print(item)
        pass