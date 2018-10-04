import tensorflow as tf
from utils import *
from ops import *
import time

class point2color():
    def __init__(self, sess, flog, batch_size=32, num_pts=2048, L1_lambda=20, epoch=100, lr_g=0.0001):
        self.isSingleClass = True
        self.batch_size = batch_size
        self.num_pts = num_pts
        self.L1_lambda = L1_lambda
        self.epoch = epoch
        self.sess = sess
        self.flog = flog
        # batch_norm for generator
        self.g_bn_1 = batch_norm(name="g_bn_1")
        self.g_bn_2 = batch_norm(name="g_bn_2")
        self.g_bn_3 = batch_norm(name="g_bn_3")
        self.g_bn_4 = batch_norm(name="g_bn_4")
        self.g_bn_5 = batch_norm(name="g_bn_5")

        self.g_bn_seg_1 = batch_norm(name="g_bn_seg_1")
        self.g_bn_seg_2 = batch_norm(name="g_bn_seg_2")
        self.g_bn_seg_3 = batch_norm(name="g_bn_seg_3")

        self.lr_g = lr_g

        self.build_model()
        pass

    def generator(self, point_cloud, bn_is_train, keep_prob):
        pass
        """

        :param point_cloud: input raw point cloud , should be of shape (batch_size, num_pts, 3)
        :param bn_is_train: an indicator whether its in training phase or in test phase. if true, parameters of bn would be updated.
        refer http://ruishu.io/2016/12/27/batchnorm/
        :return: generated rgb color list of shape (batch_size, num_pts, 3).
        """
        with tf.variable_scope("generator") as scope:
            input_image = tf.expand_dims(point_cloud, -1)  # batch_size x N x 3 x 1
            out1 = conv2d(input_=input_image, k_w=3, output_dim=64, scope='g_conv1')  # (32, N, 1, 64)
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
            concat = tf.concat(axis=3, values=[expand, out1, out2, out3, out4,
                                               out5])  # batch_size x N x 1 x (2048 + 256 + 512 + 64)

            net2 = conv2d(input_=concat, output_dim=1024, scope='seg/g_conv1')
            net2 = tf.nn.relu(self.g_bn_seg_1(net2, phase=bn_is_train))
            self.deadReLU_net1 = tf.summary.histogram("g_net1", net2)
            net2 = tf.nn.dropout(net2, keep_prob=keep_prob, name="seg/g_dp1")

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

    def build_model(self):
        self.real_pts_color_ph = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_pts, 6),
                                                name="real_pts_color_ph")
        self.bn_is_train = tf.placeholder(dtype=tf.bool, shape=(), name="bn_is_train")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name="keep_prob")

        self.real_pts_ts = self.real_pts_color_ph[:, :, :3]
        self.real_color_ts = self.real_pts_color_ph[:, :, 3:]  # tensor of real point cloud color

        self.fake_color_ts = self.generator(self.real_pts_ts, self.bn_is_train,
                                            self.keep_prob)  # tensor of fake point cloud color TODO


        # define d loss and g loss NS-GAN loss
        self.g_loss = tf.reduce_mean(tf.abs(self.real_color_ts - self.fake_color_ts))

        # Add losses to summary
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)


        # merge summary
        self.d_sum = tf.summary.merge([self.deadReLU_net1,
                                       self.deadReLU_net2,
                                       self.deadReLU_net3,
                                       self.deadReLU_out1,
                                       self.deadReLU_out2,
                                       self.deadReLU_out3,
                                       self.deadReLU_out4,
                                       self.deadReLU_out5])
        self.g_sum = tf.summary.merge([self.g_loss_sum])

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.g_optim = tf.train.AdamOptimizer(learning_rate=self.lr_g, name="g_optim").minimize(self.g_loss)

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
        print(np.mean(ncolor[0], axis=0))
        ndata_ncolor = np.concatenate((ndata, ncolor), axis=-1)
        test_ndata_ncolor = np.concatenate((test_ndata, test_ncolor), axis=-1)
        assert ndata_ncolor.shape == (data.shape[0], data.shape[1], 6)
        show_all_variables()
        # define saver and summary writer
        self.saver = tf.train.Saver(max_to_keep=200)
        model_dir = os.path.join(train_dir, "model")
        os.mkdir(os.path.join(train_dir, "logs"))
        train_sum_dir = os.path.join(train_dir, "logs/train")
        test_sum_dir = os.path.join(train_dir, "logs/test")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(train_sum_dir):
            os.mkdir(train_sum_dir)
        if not os.path.exists(test_sum_dir):
            os.mkdir(test_sum_dir)
        self.train_writer = tf.summary.FileWriter(train_sum_dir, self.sess.graph)  # 模型的图已经保存
        # self.train_writer.add_graph(self.sess.graph)
        # self.test_writer = tf.summary.FileWriter(test_sum_dir, self.sess.graph)

        # initilize variables
        tf.global_variables_initializer().run()
        # loop for epoch
        # self.saver.save(self.sess, model_dir, global_step=0)
        self.num_batches = ndata.shape[0] // self.batch_size
        start_time = time.time()

        test_masks = np.random.choice(test_data.shape[0], self.batch_size, replace=False)
        batch_test_ndata = test_ndata_ncolor[test_masks]
        batch_test_color = test_color[test_masks]
        batch_test_data = test_data[test_masks]

        train_masks = np.random.choice(data.shape[0], self.batch_size, replace=False)
        batch_train_ndata_ncolor = ndata_ncolor[train_masks]
        batch_train_data = data[train_masks]  # no normalized data
        batch_train_color = color[train_masks]  # true color
        for epoch in range(self.epoch):
            # get batch data
            for idx in range(self.num_batches):
                global_step = epoch * self.num_batches + idx + 1
                masks = range(idx * self.batch_size, (idx + 1) * self.batch_size)
                batch_ndata_ncolor = ndata_ncolor[masks]
                batch_data = data[masks]
                batch_color = color[masks]
                g_sum_save, g_loss_print, _ = self.sess.run([self.g_sum, self.g_loss, self.g_optim], feed_dict={
                    self.real_pts_color_ph: batch_ndata_ncolor,
                    self.bn_is_train: True,
                    self.keep_prob: 0.8})
                self.train_writer.add_summary(g_sum_save, global_step)

                period = time.time() - start_time
                printout(self.flog,
                         "Training! epoch %3d/%3d batch%3d/%3d time: %2dh%2dm%2ds  g_loss: %.4f" % (
                             epoch + 1, self.epoch, idx + 1, self.num_batches, period // 3600, period // 60,
                             period % 60, g_loss_print))
                # fake_color = self.sess.run(self.fake_color_ts, feed_dict = {self.real_pts_color_ph: batch_ndata_ncolor,
                #                                               self.bn_is_train: True})
                # fake_color256 = ((fake_color + 1) * 128).astype(np.int16) # (batch_size, N, 3)
                # hex_fake_color = np_color_to_hex_str(fake_color256[0])
                # hex_true_color = np_color_to_hex_str(batch_color[0])
                # display_point(batch_data[0], hex_true_color, hex_fake_color)
            # save model each 10 epoch
            train_fake_color = self.sess.run(self.fake_color_ts,
                                             feed_dict={self.real_pts_color_ph: batch_train_ndata_ncolor,
                                                        self.bn_is_train: False, self.keep_prob: 0.8})
            train_fake_color256 = ((train_fake_color + 1) * 127.5).astype(np.int16)
            hex_train_fake_colors = [np_color_to_hex_str(color) for color in train_fake_color256]
            hex_train_true_colors = [np_color_to_hex_str(color) for color in batch_train_color]
            for idx, data in enumerate(batch_train_data):
                fname = os.path.join(train_sum_dir, "epoch{0}_{1}.png".format(epoch, idx))
                display_point(data, hex_train_true_colors[idx], hex_train_fake_colors[idx], fname=fname)
                printout(self.flog, "Saved! {}".format(fname))

            if epoch % 1 == 0:
                self.saver.save(self.sess, model_dir + "/model", epoch)
                test_fake_color = self.sess.run(self.fake_color_ts, feed_dict={self.real_pts_color_ph: batch_test_ndata,
                                                                               self.bn_is_train: False,
                                                                               self.keep_prob: 0.8})
                # reconstruct to 0 - 255
                test_fake_color256 = ((test_fake_color + 1) * 127.5).astype(np.int16)  # (batch_size, N, 3)
                hex_fake_colors = [np_color_to_hex_str(color) for color in test_fake_color256]
                hex_true_colors = [np_color_to_hex_str(color) for color in batch_test_color]
                for idx, data in enumerate(batch_test_data):
                    fname_ply = os.path.join(test_sum_dir, "epoch{0}_{1}.ply".format(epoch, idx))
                    save_ply(data, test_fake_color256[idx], fname_ply)
                    fname = os.path.join(test_sum_dir, "epoch{0}_{1}.png".format(epoch, idx))
                    display_point(data, hex_true_colors[idx], hex_fake_colors[idx], fname=fname)
                    printout(self.flog, "Saved! {0} and {1}".format(fname, fname_ply))
                    # test for each 5 epoch