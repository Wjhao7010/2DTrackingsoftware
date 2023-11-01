import cv2
import argparse
import os.path
import sys

sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')
from common.utility import *

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--path", default='E:\\data\\3D_pre\\D1_T1', help="Path to the directory containing the video file")

    args = vars(ap.parse_args())

    if args.get("path", None) is None:
        print('No path was provided. Try again!')
        sys.exit()
    else:
        path = args["path"]

    backgrounds = os.path.join(path, 'background')
    mean_background = os.path.join(backgrounds, 'mean-background')
    background_list = os.listdir(backgrounds)
    camids = {}
    for ifile in background_list:
        icamid = ifile.split(".")[0].split("_")[-1]
        if icamid not in camids:
            camids[icamid] = [ifile]
        else:
            camids[icamid].append(ifile)

    for icamid, filelist in camids.items():
        bg_file = os.path.join(mean_background, f'background{icamid}.jpg')
        imgs = [cv2.imread(os.path.join(backgrounds, _)) for _ in filelist]
        print(f"Generating background_{icamid} from {len(imgs)} backgrounds")
        stack = np.stack(imgs)
        bg = np.median(stack, axis=0)
        cv2.imwrite(bg_file, bg)
