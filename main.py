from pointNetGAN import point2color as GAN
from utils import *
import time, os
import numpy as np

NUM_PTS = 4096
#BATCH_SIZE = 200
EPOCH = 200
CAT_DIR = "./Data/category_h5py"
train_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
# train_dir = "./train_results/2018_07_10_16_27"
train_dir = os.path.join("./train_results", train_time)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

# K = 3
# hex_color = np_color_to_hex_str(color[K])
# display_point(data[K], hex_color)


# K = 2
# hex_color = np_color_to_hex_str(test_color[K])
# display_point(test_data[K], hex_color)

def main():
    for cat in os.listdir(CAT_DIR):
        if cat in os.listdir(train_dir):
            continue
        if "chair" in cat:
            train_cat_dir = os.path.join(train_dir, cat)
            if not os.path.exists(train_cat_dir):
                os.mkdir(train_cat_dir)
            flog = open(os.path.join(train_cat_dir, 'log.txt'), 'w')

            data, ndata, color, pid = load_single_cat_h5(cat, NUM_PTS, "train", "data", "ndata", "color", "pid")
            printout(flog, "Loading training data! {}".format(cat))
            test_data, test_ndata, test_color = load_single_cat_h5(cat, NUM_PTS,"test","data", "ndata", "color")
            printout(flog, "Loading test data! {}".format(cat))

            # data, ndata, color = data[:100], ndata[:100], color[:100]
            if test_ndata.shape[0]  > 8:
                BATCH_SIZE = 8
            elif test_ndata.shape[0] > 4:
                BATCH_SIZE = 4
            else:
                BATCH_SIZE = 2
            with tf.Session() as sess:
                ptGAN = GAN(sess, flog, num_pts=NUM_PTS, batch_size=BATCH_SIZE, epoch=EPOCH)
                ptGAN.train(train_cat_dir, data, ndata, color, test_data, test_ndata, test_color)
            tf.reset_default_graph()

if __name__ =="__main__":
    main()