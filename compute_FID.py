from sklearn.manifold import TSNE
import os
import h5py
import numpy as np
from math import ceil, floor
from trained_cls.voxelNet import *
import configparser
import time
import shutil
import matplotlib.pyplot as plt
from metrics import *
from collections import namedtuple
import matplotlib as mpl
import tensorflow as tf
from utils import *

plt.style.use("seaborn")
np.random.seed(42)
category_name = []
# num_pts = 4096

NUM_PTS = 4096
# load generator and generate the colorized point cloud
genrated_point_cloud = []
generation_label = []
cat_list = [i for i in os.listdir("./Data/category_h5py") if os.path.isdir(os.path.join("./Data/category_h5py", i))]
for cat in cat_list:
    tf.reset_default_graph()
    cat_name = cat.split("_")[-1].split(",")[0]
    # load test data
    test_data, test_ndata, test_color, test_cid = load_single_cat_h5(cat, NUM_PTS,"test","data", "ndata", "color", "cid")
    nb_samples = test_data.shape[0]
    modelPath = "./train_results/2018_10_13_15_40_L1_loss_only/{}/model/".format(cat)
    model_id = 180
    graph_file = os.path.join(modelPath, "model-" + str(model_id) + ".meta")
    variable_file = os.path.join(modelPath, "model-" + str(model_id))
    GAN_graph=tf.Graph()
    with tf.Session() as sess:
        try:
            saver = tf.train.import_meta_graph(graph_file)
            saver.restore(sess, variable_file)
        except:
            continue
        fake_pts = tf.get_default_graph().get_tensor_by_name("generator/Tanh:0")
        input_pt = tf.get_default_graph().get_tensor_by_name("real_pts_color_ph:0")
        batch_size = int(input_pt.get_shape()[0])
        #batch_size = 2
        bn_is_train = tf.get_default_graph().get_tensor_by_name("bn_is_train:0")
        keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
        # keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")

        total_batch = test_data.shape[0] // batch_size
        for i in range(total_batch):
            start_idx = batch_size * i
            end_idx = batch_size * (i + 1)
            batch_test_ndata = test_ndata[start_idx:end_idx]
            batch_test_data = test_data[start_idx:end_idx]
            batch_test_color = test_color[start_idx:end_idx]
            batch_label = test_cid[start_idx:end_idx]
            batch_test_ndata_color = np.concatenate([batch_test_ndata, batch_test_color], axis=-1)
            fake_colored_pts = sess.run(fake_pts, feed_dict={input_pt: batch_test_ndata_color,
                                                             bn_is_train: False,
                                                             keep_prob: 1.0})
            fake_colored_pts = np.squeeze(fake_colored_pts)
            #display_point(batch_test_ndata[0], ((fake_colored_pts[0]+ 1) * 127.5).astype(np.int16))
            print(fake_colored_pts.shape, batch_test_ndata.shape)
            genrated_point_cloud.append(np.concatenate((batch_test_ndata, fake_colored_pts), axis=-1))
            generation_label.append(batch_label)

genrated_point_cloud = np.concatenate(genrated_point_cloud, 0)
generation_label = np.concatenate(generation_label, 0)
print(genrated_point_cloud.shape)
f = h5py.File("./Data/generation_L1_model_{}.hdf5".format(model_id), "w")

f.create_dataset(data=genrated_point_cloud, name="ndata_ncolor")
tf.reset_default_graph()
# load classifier and compute feature vectors
cls_dir = "trained_cls/model-20"

config = configparser.ConfigParser()
config.read("trained_cls/base_config.ini")
batch_size = config["hyperparameters"].getint("batch_size")
num_pts = config["hyperparameters"].getint("num_pts")

pc_pl, labels_pl = placeholder_inputs(batch_size,
                                          num_pts,
                                          num_feature=6)
bn_pl = tf.placeholder(dtype=tf.bool, shape=())
class_weights_pl = tf.placeholder(dtype=tf.float32, shape=(batch_size,))

logits, avg_out1, avg_out2, avg_out3, avg_out4, avg_out5, dense_1, dense_2, dense_3, dense_4\
        = get_model(pc_pl, bn_is_train=bn_pl, style_transfer_test=True)
#acc = tf.reduce_sum(tf.cast(tf.argmax(logits, axis=-1) == labels_pl, tf.int8))

out_FE_dense1 = tf.get_default_graph().get_tensor_by_name("cls/VFE-1/dense/BiasAdd:0") # batch_size, num_pts, 1024
out_FE_dense1_mean = tf.reduce_mean(out_FE_dense1, axis=1) # batch_size, 1024
out_FE_dense1_max = tf.reduce_max(out_FE_dense1, axis=1) # batch_size, 1024
out_FE_dense1_aggre = tf.concat(axis=1, values=[out_FE_dense1_max, out_FE_dense1_mean]) # batch_size, 2048

out_FE_dense2 = tf.get_default_graph().get_tensor_by_name("cls/VFE-2/dense/BiasAdd:0") # batch_size, num_pts, 1024
out_FE_dense2_mean = tf.reduce_mean(out_FE_dense2, axis=1) # batch_size, 1024
out_FE_dense2_max = tf.reduce_max(out_FE_dense2, axis=1) # batch_size, 1024
out_FE_dense2_aggre = tf.concat(axis=1, values=[out_FE_dense2_max, out_FE_dense2_mean]) # batch_size, 2048

out_FE_dense3 = tf.get_default_graph().get_tensor_by_name("cls/VFE-3/dense/BiasAdd:0") # batch_size, num_pts, 1024
out_FE_dense3_mean = tf.reduce_mean(out_FE_dense3, axis=1) # batch_size, 1024
out_FE_dense3_max = tf.reduce_max(out_FE_dense3, axis=1) # batch_size, 1024
out_FE_dense3_aggre = tf.concat(axis=1, values=[out_FE_dense3_max, out_FE_dense3_mean]) # batch_size, 2048

out_FE_dense4 = tf.get_default_graph().get_tensor_by_name("cls/VFE-4/dense/BiasAdd:0") # batch_size, num_pts, 1024
out_FE_dense4_mean = tf.reduce_mean(out_FE_dense4, axis=1) # batch_size, 1024
out_FE_dense4_max = tf.reduce_max(out_FE_dense4, axis=1) # batch_size, 1024
out_FE_dense4_aggre = tf.concat(axis=1, values=[out_FE_dense4_max, out_FE_dense4_mean]) # batch_size, 2048

out_FE_dense5 = tf.get_default_graph().get_tensor_by_name("cls/VFE-5/dense/BiasAdd:0") # batch_size, num_pts, 1024
out_FE_dense5_mean = tf.reduce_mean(out_FE_dense5, axis=1) # batch_size, 1024
out_FE_dense5_max = tf.reduce_max(out_FE_dense5, axis=1) # batch_size, 1024
out_FE_dense5_aggre = tf.concat(axis=1, values=[out_FE_dense5_max, out_FE_dense5_mean]) # batch_size, 4096

out_dense1 = tf.get_default_graph().get_tensor_by_name("cls/dense1/dense/BiasAdd:0") # batch_size, 1024

out_dense2 = tf.get_default_graph().get_tensor_by_name("cls/dense2/dense/BiasAdd:0") # batch_size, 1024

out_dense3 = tf.get_default_graph().get_tensor_by_name("cls/dense3/dense/BiasAdd:0") # batch_size, 1024

out_dense4 = tf.get_default_graph().get_tensor_by_name("cls/dense4/dense/BiasAdd:0") # batch_size, 1024

saver = tf.train.Saver()
test_data = h5py.File(os.path.join("./Data/test_data", "test_epoch_0.hdf5"), "r") #
test_ndata = test_data["ndata"][:]
test_color = test_data["color"][:]
test_label = test_data["cid"][:]
test_ncolor = (test_color - 127.5) / 127.5 #(-1, 1) # (num_samples, num_points, 3)

total_batch = floor(test_ndata.shape[0] / batch_size)

layers = ["out_FE_dense1_aggre",
        "out_FE_dense2_aggre",
        "out_FE_dense3_aggre",
        "out_FE_dense4_aggre",
        "out_FE_dense5_aggre",
        "out_dense1",
        "out_dense2",
        "out_dense3",
        "out_dense4"]
all_layer_output = {i:[] for i in layers}

stat = namedtuple("statistic", "mean cov")
GT_statistic = {i: None for i in layers}
with tf.Session() as sess:
    saver.restore(sess, cls_dir)
    total_acc = 0
    for current_batch in range(total_batch):
        start_idx = current_batch * batch_size
        end_idx = (current_batch + 1) * batch_size
        current_batch_test_ndata = test_ndata[start_idx:end_idx]
        current_batch_test_color = test_ncolor[start_idx:end_idx]
        current_batch_label = test_label[start_idx:end_idx]
        current_batch_test_ndata_ncolor = np.concatenate((current_batch_test_ndata, current_batch_test_color), axis=-1)
        batch_logits, batch_out_FE_dense1_aggre, batch_out_FE_dense2_aggre, batch_out_FE_dense3_aggre, batch_out_FE_dense4_aggre, \
        batch_out_FE_dense5_aggre, batch_out_dense1, batch_out_dense2, batch_out_dense3, batch_out_dense4 = \
            sess.run([logits, out_FE_dense1_aggre, out_FE_dense2_aggre, out_FE_dense3_aggre, out_FE_dense4_aggre,
                      out_FE_dense5_aggre,
                      out_dense1, out_dense2, out_dense3, out_dense4]
                     , feed_dict={pc_pl: current_batch_test_ndata_ncolor,
                                  bn_pl: False,
                                  labels_pl: current_batch_label})
        batch_acc = np.sum(np.argmax(batch_logits, axis=-1) == current_batch_label) / batch_size
        total_acc += batch_acc
        all_layer_output["out_FE_dense1_aggre"].append(batch_out_FE_dense1_aggre)
        all_layer_output["out_FE_dense2_aggre"].append(batch_out_FE_dense2_aggre)
        all_layer_output["out_FE_dense3_aggre"].append(batch_out_FE_dense3_aggre)
        all_layer_output["out_FE_dense4_aggre"].append(batch_out_FE_dense4_aggre)
        all_layer_output["out_FE_dense5_aggre"].append(batch_out_FE_dense5_aggre)
        all_layer_output["out_dense1"].append(batch_out_dense1)
        all_layer_output["out_dense2"].append(batch_out_dense2)
        all_layer_output["out_dense3"].append(batch_out_dense3)
        all_layer_output["out_dense4"].append(batch_out_dense4)

    print("test acc: {}".format(total_acc/total_batch))
    for key in all_layer_output.keys():
        all_layer_output[key] = np.concatenate(all_layer_output[key], axis=0)  # (num_samples, num_features)
        GT_statistic[key] = stat._make(compute_statistics(all_layer_output[key]))
    print(GT_statistic["out_dense1"].mean)

    FID = dict()
    total_batch = genrated_point_cloud.shape[0] // batch_size
    generation_output = {i: [] for i in layers}
    generation_statistic = {i: None for i in layers}
    generation_acc = 0
    for current_batch in range(total_batch):
        start_idx = current_batch * batch_size
        end_idx = (current_batch + 1) * batch_size
        current_batch_test_ndata_ncolor = genrated_point_cloud[start_idx:end_idx]
        current_batch_label = generation_label[start_idx:end_idx]
        batch_logits, batch_out_FE_dense1_aggre, batch_out_FE_dense2_aggre, batch_out_FE_dense3_aggre, batch_out_FE_dense4_aggre, \
        batch_out_FE_dense5_aggre, batch_out_dense1, batch_out_dense2, batch_out_dense3, batch_out_dense4 = \
            sess.run([logits, out_FE_dense1_aggre, out_FE_dense2_aggre, out_FE_dense3_aggre, out_FE_dense4_aggre,
                      out_FE_dense5_aggre,
                      out_dense1, out_dense2, out_dense3, out_dense4]
                     , feed_dict={pc_pl: current_batch_test_ndata_ncolor,
                                  bn_pl: False,
                                  labels_pl:current_batch_label})
        batch_acc = np.sum(np.argmax(batch_logits,axis=-1) == current_batch_label) / batch_size
        generation_acc += batch_acc
        generation_output["out_FE_dense1_aggre"].append(batch_out_FE_dense1_aggre)
        generation_output["out_FE_dense2_aggre"].append(batch_out_FE_dense2_aggre)
        generation_output["out_FE_dense3_aggre"].append(batch_out_FE_dense3_aggre)
        generation_output["out_FE_dense4_aggre"].append(batch_out_FE_dense4_aggre)
        generation_output["out_FE_dense5_aggre"].append(batch_out_FE_dense5_aggre)
        generation_output["out_dense1"].append(batch_out_dense1)
        generation_output["out_dense2"].append(batch_out_dense2)
        generation_output["out_dense3"].append(batch_out_dense3)
        generation_output["out_dense4"].append(batch_out_dense4)
    print("generation acc: {}".format(generation_acc / total_batch))
    for key in generation_output.keys():
        print(key)
        generation_output[key] = np.concatenate(generation_output[key], axis=0)  # (num_samples, num_features)
        generation_statistic[key] = stat._make(compute_statistics(generation_output[key]))
        FID[key] = calculate_frechet_distance(mu1=GT_statistic[key].mean, sigma1=GT_statistic[key].cov,
                                                          mu2=generation_statistic[key].mean,
                                                          sigma2=generation_statistic[key].cov)
    print(FID)
    plt.figure()
    plt.bar(list(range(9)), list(FID.values()))
    plt.show()



# compute FID