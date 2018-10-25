# from pointNetGAN import point2color as GAN
from model import point2color as GAN
from utils import *
import time, os
import numpy as np
import configparser
import shutil
import socket

# download training data modified from shapenet
HOSTNAME = socket.gethostname()
DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'category_h5py')):
    www = 'https://drive.google.com/open?id=1hNBo71t_0jryztKalcn-e30EPeSNWj4i'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))



config = configparser.ConfigParser()
config.read("base_config.ini")
NUM_PTS = config["hyperparameters"].getint("num_pts")
EPOCH = config["hyperparameters"].getint("epoch")
LR = config["hyperparameters"].getint("epoch")
CAT_DIR = "./data/category_h5py"

train_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
train_dir = "./train_results/2018_07_10_16_27"
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

shutil.copyfile("base_config.ini", os.path.join(train_dir, "config.ini"))


def main():
    for cat in os.listdir(CAT_DIR):
        if cat in os.listdir(train_dir):
            continue
        train_cat_dir = os.path.join(train_dir, cat)
        if not os.path.exists(train_cat_dir):
            os.mkdir(train_cat_dir)
        flog = open(os.path.join(train_cat_dir, 'log.txt'), 'w')

        # load training data from hdf5
        data, ndata, color, pid = load_single_cat_h5(cat, NUM_PTS, "train", "data", "ndata", "color", "pid")
        printout(flog, "Loading training data! {}".format(cat))
        # load test data from hdf5
        test_data, test_ndata, test_color = load_single_cat_h5(cat, NUM_PTS,"test","data", "ndata", "color")
        printout(flog, "Loading test data! {}".format(cat))
        print(np.amin(ndata))

        use_samples = 640
        data, ndata, color = data[:use_samples], ndata[:use_samples], color[:use_samples]
        if test_ndata.shape[0]  > 8:
            BATCH_SIZE = 8
        elif test_ndata.shape[0] > 4:
            BATCH_SIZE = 4
        else:
            BATCH_SIZE = 2
        with tf.Session() as sess:
            ptGAN = GAN(sess, flog, num_pts=NUM_PTS,
                        batch_size=BATCH_SIZE,
                        epoch=EPOCH,
                        lr_g=LR)
            ptGAN.train(train_cat_dir, data, ndata, color, test_data, test_ndata, test_color)
        tf.reset_default_graph()

if __name__ =="__main__":
    main()