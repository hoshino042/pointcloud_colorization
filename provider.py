import h5py, os
data_dir = "Data/pts_2048"
with open("Data/train_hdf5_file_list.txt") as f:
    lines = f.readlines()
file_list = [line.strip() for line in lines]

print(os.getcwd())
def get_single_category(data_path, file_list):
    data_list, color_list, cid_list = [], [], []
    for item in file_list:
        f = h5py.File(os.path.join(data_path, item))
        data_list.append(f["data"][:])
        color_list.append(f["color"][:])
        cid_list.append(f["cid"][:])
        print(cid_list[0])



if __name__ =="__main__":
    print(file_list)
    get_single_category(data_dir, file_list, )