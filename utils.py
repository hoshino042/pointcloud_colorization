import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D
from pyntcloud import PyntCloud
import pandas as pd

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def directed_hausdorff_distance(ptsA, ptsB):
    """
    This function computes  directed hausdorff distance as h(A,B)=max{min{d(a,b)}}
    refer: http://cgm.cs.mcgill.ca/~godfried/teaching/cg-projects/98/normand/main.html
    :param ptsA: numpy array of shape (Na, Nf)
    :param ptsB: numpy array of shape (Nb, Nf)
    :return:
    """
    assert ptsA.ndim == ptsB.ndim
    ptsA, ptsB = np.squeeze(ptsA), np.squeeze(ptsB)
    list_min = []
    for pts in ptsA:
        list_min.append(np.min(np.sqrt(np.sum((ptsB - pts) ** 2, axis=1))))
    return max(list_min)


def hausdorff_distance(ptsA, ptsB):
    """
    This function computes  directed hausdorff distance as H(A, B) = max{h(A,B), h(B,A)}
    :param ptsA:
    :param ptsB:
    :return:
    """
    return max([directed_hausdorff_distance(ptsA, ptsB), directed_hausdorff_distance(ptsB, ptsA)])


def display_point(pts, color, color_label=None, fname=None):
    """

    :param pts:
    :param color:
    :param color_label:
    :param fname: save path and filename of the figue
    :return:
    """
    DPI =300
    PIX_h = 1000
    MARKER_SIZE = 5
    if color_label is None:
        PIX_w = PIX_h
    else:
        PIX_w = PIX_h * 2
    X = pts[:, 0]
    Y = pts[:, 2]
    Z = pts[:, 1]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    fig = plt.figure()
    fig.set_size_inches(PIX_w/DPI, PIX_h/DPI)
    plt.subplots_adjust(top=1.2, bottom=-0.2, right=1.5, left=-0.5, hspace=0, wspace=-0.7)
    plt.margins(0, 0)
    if color_label is None:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c=color, edgecolors="none", s=MARKER_SIZE, depthshade=True)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_aspect("equal")
        ax.grid("off")
        plt.axis('off')
        if fname:
            plt.savefig(fname, transparent=True, dpi=DPI)
            plt.close(fig)
        else:
            plt.show()
    else:
        ax = fig.add_subplot(121, projection='3d')
        bx = fig.add_subplot(122, projection='3d')
        ax.scatter(X, Y, Z, c=color, edgecolors="none", s=MARKER_SIZE, depthshade=True)
        bx.scatter(X, Y, Z, c=color_label, edgecolors="none", s=MARKER_SIZE, depthshade=True)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        bx.set_xlim(mid_x - max_range, mid_x + max_range)
        bx.set_ylim(mid_y - max_range, mid_y + max_range)
        bx.set_zlim(mid_z - max_range, mid_z + max_range)
        bx.patch.set_alpha(0)
        ax.set_aspect("equal")
        ax.grid("off")
        bx.set_aspect("equal")
        bx.grid("off")
        ax.axis('off')
        bx.axis("off")
        plt.axis('off')
        if fname:
            plt.savefig(fname, transparent=True, dpi=DPI)
            plt.close(fig)

        else:
            plt.show()


def int16_to_hex_str(color):
    hex_str = ""
    color_map = {i: str(i) for i in range(10)}
    color_map.update({10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "F"})
    # print(color_map)
    hex_str += color_map[color // 16]
    hex_str += color_map[color % 16]
    return hex_str


def rgb_to_hex_str(*rgb):
    hex_str = "#"
    for item in rgb:
        hex_str += int16_to_hex_str(item)
    return hex_str


def np_color_to_hex_str(color):
    """
    :param color: an numpy array of shape (N, 3)
    :return: a list of hex color strings
    """
    hex_list = []
    for rgb in color:
        hex_list.append(rgb_to_hex_str(rgb[0], rgb[1], rgb[2]))
    return hex_list


def load_h5(path, *kwd):
    f = h5py.File(path)
    list_ = []
    for item in kwd:
        list_.append(f[item][:])
        print("{0} of shape {1} loaded!".format(item, f[item][:].shape))
        if item == "ndata" or item == "data":
            pass# print(np.mean(f[item][:], axis=1))
        if item == "color":
            print("color is of type {}".format(f[item][:].dtype))
    return list_

def load_single_cat_h5(cat,num_pts,type, *kwd):
    fpath = os.path.join("./Data/category_h5py", cat, "PTS_{}".format(num_pts), "ply_data_{}.h5".format(type))
    return load_h5(fpath, *kwd)

def printout(flog, data): # follow up
    print(data)
    flog.write(data + '\n')

def save_ply(data, color, fname):
    color = color.astype(np.uint8)
    df1 = pd.DataFrame(data, columns=["x","y","z"])
    df2 = pd.DataFrame(color, columns=["red","green","blue"])
    pc = PyntCloud(pd.concat([df1, df2], axis=1))
    pc.to_file(fname)

if __name__ == "__main__":
    pass
    # a = np.array([[1, 2, 3],
    #               [2, 3, 4]])
    # b = np.array([[2, 2, 3],
    #               [3, 4, 5]])
    # print(directed_hausdorff_distance(a, b))
    # print(hausdorff_distance(a, b))
    # pts_path = r"F:\ShapeNetPartC\category_h5py\02691156_airplane,aeroplane,plane\PTS_4096\ply_data_val.h5"
    # output_dir = r"F:\ShapeNetPartC\category_h5py\02691156_airplane,aeroplane,plane\PTS_4096\g.png"
    # pts, color, pid = load_h5(pts_path, "data", "color", "pid")
    # K = 62
    # hex_color = color[K]
    # display_point(pts[K], np_color_to_hex_str(hex_color), pid[K], output_dir)
    #
