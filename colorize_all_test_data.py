import tensorflow as tf
import os
import time
from utils import *
import numpy as np


cat_list = [i for i in os.listdir("./data/category_h5py") if os.path.isdir(os.path.join("./data/category_h5py", i))]
NUM_PTS = 4096
if not os.path.exists("test_results"):
    os.mkdir("test_results")
test_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
test_dir = os.path.join("test_results", test_time)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

for cat in cat_list:
    cat_dir = os.path.join(test_dir, cat)
    cat_name = cat.split("_")[-1].split(",")[0]
    if not os.path.exists(cat_dir):
        os.mkdir(cat_dir)
    test_data, test_ndata, test_color = load_single_cat_h5(cat, NUM_PTS,"test","data", "ndata", "color")
    nb_samples = test_data.shape[0]
    modelPath = "./train_results/2018_07_10_16_27/{}/model/".format(cat)
    model_id = 180
    graph_file = os.path.join(modelPath, "model-" + str(model_id) + ".meta")
    variable_file = os.path.join(modelPath, "model-" + str(model_id))
    GAN_graph=tf.Graph()

    LOG_FOUT = open(os.path.join(cat_dir, 'log_test.txt'), 'w')
    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)
    log_string(graph_file)

    with tf.Session() as sess:
        try:
            saver = tf.train.import_meta_graph(graph_file)
            saver.restore(sess, variable_file)
        except:
            continue
        fake_pts = tf.get_default_graph().get_tensor_by_name("generator/Tanh:0")
        input_pt = tf.get_default_graph().get_tensor_by_name("real_pts_color_ph:0")
        batch_size = int(input_pt.get_shape()[0])
        bn_is_train = tf.get_default_graph().get_tensor_by_name("bn_is_train:0")

        total_batch = test_data.shape[0] // batch_size
        for i in range(total_batch):
            start_idx = batch_size * i
            end_idx = batch_size * (i+1)
            batch_test_ndata = test_ndata[start_idx:end_idx]
            batch_test_data = test_data[start_idx:end_idx]
            batch_test_color = test_color[start_idx:end_idx]
            batch_test_ndata_color = np.concatenate([batch_test_ndata, batch_test_color], axis=-1)
            fake_colored_pts = sess.run(fake_pts, feed_dict={input_pt: batch_test_ndata_color,
                                                         bn_is_train: False})
            fake_colored_pts = np.squeeze(fake_colored_pts)
            test_fake_color256 = ((fake_colored_pts + 127.5) * 127.5).astype(np.int16)  # (batch_size, N, 3)
            # show_id = 2
            for j in range(batch_size):
                fname_GT = os.path.join(cat_dir, "test_chair_GT_{}.png".format(i * batch_size + j))
                fname_input = os.path.join(cat_dir, "test_chair_input_{}.png".format(i * batch_size + j))
                fname_gen = os.path.join(cat_dir, "test_chair_gen_{}.png".format(i * batch_size + j))
                display_point(batch_test_data[j], test_fake_color256[j], fname=fname_gen)
                display_point(batch_test_data[j], batch_test_color[j], fname=fname_GT)
                display_point(batch_test_data[j], 127*np.ones_like(fake_colored_pts[j]), fname=fname_input)
                fout = os.path.join(cat_dir, "test_{0}_{1}.png".format(cat_name, i * batch_size + j))
                try:
                    horizontal_concatnate_pic(fout, fname_input, fname_GT, fname_gen)
                except:
                    continue
                os.remove(fname_GT)
                os.remove(fname_gen)
                os.remove(fname_input)
            LOG_FOUT.close()



