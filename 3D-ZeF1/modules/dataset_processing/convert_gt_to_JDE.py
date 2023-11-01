import os.path as osp
import numpy as np
import os
import random
import shutil

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

# /home/data/HJZ/MFT/train/cruise1/img1
# /home/data/HJZ/MFT/train/cruise1/gt
# /home/data/HJZ/MFT/train/cruise1/labels_with_ids
def format_jde():
    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imwidth=') + 8:seq_info.find('\nimheight')])
        seq_height = int(seq_info[seq_info.find('imheight=') + 9:seq_info.find('\nimext')])

        gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=str, delimiter=',', skiprows=1)

        seq_label_root = osp.join(label_root, seq, 'labels_with_ids')
        mkdirs(seq_label_root)

        for fid, frame_id, tid, hx, hy, tlx, tly, w, h, label in gt:
            tlx = float(tlx) + float(w) / 2
            tly = float(tly) + float(h) / 2
            label_fpath = osp.join(seq_label_root, '{0}.txt'.format(fid.split(".")[0]))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                int(tid), float(tlx) / seq_width, float(tly) / seq_height, float(w) / seq_width,
                                         float(h) / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)

def split(view):
    label_list = []
    for i in sorted(os.listdir(os.path.join(label_root, view, 'labels_with_ids'))):
        label_list.append(
            os.path.join(label_root, view, 'labels_with_ids', i)
        )
    img_list = []
    for i in sorted(os.listdir(os.path.join(label_root, view, 'img1'))):
        img_list.append(
            os.path.join(label_root, view, 'img1', i)
        )

    data_list = [_ for _ in range(len(img_list))]
    train_list = random.sample(data_list, int(len(data_list)*train_ratio))

    val_test = list(set(data_list).difference(set(train_list)))
    test_list = random.sample(val_test, int(len(data_list) * test_ratio))
    val_list = list(set(val_test).difference(set(test_list)))

    train_img = os.path.join(seq_root, f'{view}_train', 'img1')
    test_img = os.path.join(seq_root, f'{view}_test', 'img1')
    val_img = os.path.join(seq_root, f'{view}_val', 'img1')
    train_label = os.path.join(seq_root, f'{view}_train', 'labels_with_ids')
    test_label = os.path.join(seq_root, f'{view}_test', 'labels_with_ids')
    val_label = os.path.join(seq_root, f'{view}_val', 'labels_with_ids')
    if not os.path.exists(val_img):
        os.makedirs(train_img)
        os.makedirs(train_label)
        os.makedirs(val_img)
        os.makedirs(val_label)
        os.makedirs(test_img)
        os.makedirs(test_label)

    print(len(train_list))
    for idx in train_list:
        os.symlink(img_list[idx], os.path.join(train_img, img_list[idx].split("/")[-1]))
        os.symlink(label_list[idx], os.path.join(train_label, label_list[idx].split("/")[-1]))
        # shutil.copy(img_list[idx], train_img)
        # shutil.copy(label_list[idx], train_label)
    print(len(test_list))
    for idx in test_list:
        os.symlink(img_list[idx], os.path.join(test_img, img_list[idx].split("/")[-1]))
        os.symlink(label_list[idx], os.path.join(test_label, label_list[idx].split("/")[-1]))
    print(len(val_list))
    for idx in val_list:
        os.symlink(img_list[idx], os.path.join(val_img, img_list[idx].split("/")[-1]))
        os.symlink(label_list[idx], os.path.join(val_label, label_list[idx].split("/")[-1]))
    #     shutil.copy(img_list[idx], val_img)
    #     shutil.copy(label_list[idx], val_label)

seq_root = '/home/data/HJZ/3DZeF20/train/all'
label_root = '/home/data/HJZ/3DZeF20/train/all'
seqs = ['front', 'top']

if __name__ == '__main__':
    format_jde()
    train_ratio = 0.6
    test_ratio = 0.25
    val_ratio = 0.15
    for i in seqs:
        split(i)

