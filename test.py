import tensorflow as tf
import os
import time
from utils import *

cat = "03001627_chair"
NUM_PTS = 4096
batch_size = 8
test_data, test_ndata, test_color = load_single_cat_h5(cat, NUM_PTS,"test","data", "ndata", "color")
nb_samples = test_data.shape[0]
modelPath = "./train_results/2018_07_19_16_15/03001627_chair/model/"
model_id = 171

graph_file = os.path.join(modelPath, "model-" + str(model_id) + ".meta")
variable_file = os.path.join(modelPath, "model-" + str(model_id))

test_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
test_dir = os.path.join("./test_results", test_time)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

GAN_graph=tf.Graph()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(graph_file)
    saver.restore(sess, variable_file)

    fake_pts = tf.get_default_graph().get_tensor_by_name("generator/Tanh:0")
    input_pt = tf.get_default_graph().get_tensor_by_name("real_pts_color_ph:0")
    bn_is_train = tf.get_default_graph().get_tensor_by_name("bn_is_train:0")
    keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")

    masks = np.random.choice(nb_samples, batch_size, replace=False)
    batch_test_ndata = test_ndata[masks]
    batch_test_data = test_data[masks]
    batch_test_color = test_color[masks]
    batch_test_ndata_color = np.concatenate([batch_test_ndata, batch_test_color], axis=-1)
    all_ex = []

    fake_colored_pts = sess.run(fake_pts, feed_dict={input_pt: batch_test_ndata_color,
                                                     bn_is_train: False,
                                                     keep_prob:1.0})
    fake_colored_pts = np.squeeze(fake_colored_pts)
    test_fake_color256 = ((fake_colored_pts + 1) * 127.5).astype(np.int16)  # (batch_size, N, 3)
    hex_fake_colors = [np_color_to_hex_str(color) for color in test_fake_color256]
    hex_true_colors = [np_color_to_hex_str(color) for color in batch_test_color]
    for j in range(batch_size):
        fname = os.path.join(test_dir, "epoch{0}_id{1}_dp{2:.1f}.png".format(model_id,j, i/10.0))
        display_point(batch_test_data[j], hex_true_colors[j], hex_fake_colors[j], fname)


