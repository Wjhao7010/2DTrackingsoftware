# coding: utf-8
import cv2
from common.utility import *
import argparse
import json

position = {
    "left": [],
    "right": [],
    "up": [],
    "down": []
}

def on_EVENT_BUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print("left")
        position['left'].append({
            "x": x,
            "y": y
        })
        print(xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)

    if event == cv2.EVENT_RBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print("right")
        print(xy)
        position['right'].append({
            "x": x,
            "y": y
        })
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)

    if event == cv2.EVENT_LBUTTONDBLCLK:
        xy = "%d,%d" % (x, y)
        print("up")
        print(xy)
        position['up'].append({
            "x": x,
            "y": y
        })
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
    if event == cv2.EVENT_RBUTTONDBLCLK:
        xy = "%d,%d" % (x, y)
        print("down")
        print(xy)
        position['down'].append({
            "x": x,
            "y": y
        })
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)

    if event == cv2.EVENT_MBUTTONDOWN:
        print(position)
        json_str = json.dumps(position)
        with open(path+'cam1_references.json', 'w') as json_file:
            json_file.write(json_str)

if __name__ == '__main__':
    # construct the argument parser and parse the arguments

    ap = argparse.ArgumentParser()
    ap.add_argument("--path",
                        # default="/home/data/HJZ/zef/0918-0919fish",
                        default="D:\\program\\PotPlayer\\Capture",
                        type=str, help="video shear dirpath"
                        )
    ap.add_argument("--view",
                    default="top",
                    type=str, help="video shear dirpath"
                    )
    args = vars(ap.parse_args())

    # args['path'] = 'E:\\data\\3D_pre\\exp_pre\\D01_20210918195000'
    path = args["path"]
    view = args["view"]

    # config = readConfig(path)
    # c = config['Detector']
    # downsample = c.getint('downsample_factor')  # How much to downsample the image by
    # downsample = 3
    img = cv2.imread(os.path.join(path,f"{view}.jpg"))
    if img.shape[0] == 1080:
        img = cv2.resize(img, (720, 1280))
    # img = img[::downsample, ::downsample].astype(np.uint8)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)

    # while (True):
    #     try:
    #         cv2.waitKey(100)
    #     except Exception:
    #         cv2.destroyWindow("image")
    #         break
    #
    # cv2.destroyAllWindow()