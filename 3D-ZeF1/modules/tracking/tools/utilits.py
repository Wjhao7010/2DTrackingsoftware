import numpy as np
import cv2
import pandas as pd


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def py_cpu_nms(detects, thresh):
    # 首先数据赋值和计算对应矩形框的面积
    # dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]
    if isinstance(detects, pd.DataFrame):
        detects = detects[['tl_x', 'tl_y', 'w', 'h', 'confidence']].copy()
        detects['br_x'] = detects['tl_x'] + detects['w']
        detects['br_y'] = detects['tl_y'] + detects['h']
        dets = detects[['tl_x', 'tl_y', 'br_x', 'br_y', 'confidence']].values
    else:
        dets = detects

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]

    # 这边的keep用于存放，NMS后剩余的方框
    keep = []

    # 取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下。
    index = scores.argsort()[::-1]
    # 上面这两句比如分数[0.72 0.8  0.92 0.72 0.81 0.9 ]
    #  对应的索引index[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。

    # index会剔除遍历过的方框，和合并过的方框。
    while index.size > 0:
        # 取出第一个方框进行和其他方框比对，看有没有可以合并的
        i = index[0]  # every time the first is the biggst, and add it directly

        # 因为我们这边分数已经按从大到小排列了。
        # 所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        # keep保留的是索引值，不是具体的分数。
        keep.append(i)

        # 计算交集的左上角和右下角
        # 这里要注意，比如x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        # 这边要注意，如果两个方框相交，X22-X11和Y22-Y11是正的。
        # 如果两个方框不相交，X22-X11和Y22-Y11是负的，我们把不相交的W和H设为0.
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        # 计算重叠面积就是上面说的交集面积。不相交因为W和H都是0，所以不相交面积为0
        overlaps = w * h
        # print('overlaps is',overlaps)

        # 这个就是IOU公式（交并比）。
        # 得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # print('ious is',ious)

        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        # 我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        # 我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法。
        idx = np.where(ious <= thresh)[0]
        # print(idx)

        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[idx + 1]  # because index start from 1
        # print(index)
    return keep


def filterFish(bbox, camId, ROI_Area, refine_value=0):
    def ROI_Extract():
        itl_x, itl_y, ibr_x, ibr_y = ROI_Area
        # btl_x, btl_y, bbr_x, bbr_y = bbox["tl_x"], bbox["tl_y"], bbox["tl_x"] + bbox["w"], bbox["tl_y"] + bbox["h"]
        c_x, c_y = bbox["c_x"], bbox["c_y"]

        if (c_x > itl_x + refine_value) and \
                (c_x < ibr_x) and \
                (c_y > itl_y + refine_value) and \
                (c_y < ibr_y):
            return True
        else:
            return False

    def Box_Size():
        # 检测框的宽高限制
        area = bbox['w'] * bbox['h']
        # if camNO in ['ch01', 'ch02', 'ch03']:
        # 这三个都是 1080像素的
        if camId == 1:
            max_bblen = 120
        else:
            max_bblen = 220
        if (area <= max_bblen * max_bblen) and (bbox['w'] < max_bblen) and (bbox['h'] < bbox['w']):
            return True
        else:
            return False

    return Box_Size() and ROI_Extract()
    # return True


def readDetectionCSV(df, camId, ROI_Area, downsample=1):
    kps = []
    bbs = []
    counter = 0
    for i, row in df.iterrows():
        bb = {"x": row["x"] / downsample,
              "y": row["y"] / downsample,
              "tl_x": row["tl_x"] / downsample,
              "tl_y": row["tl_y"] / downsample,
              "c_x": row["c_x"] / downsample,
              "c_y": row["c_y"] / downsample,
              "w": row["w"] / downsample,
              "h": row["h"] / downsample,
              "theta": row["theta"],
              "l_x": row["l_x"] / downsample,
              "l_y": row["l_y"] / downsample,
              "r_x": row["r_x"] / downsample,
              "r_y": row["r_y"] / downsample,
              "aa_tl_x": row["aa_tl_x"] / downsample,
              "aa_tl_y": row["aa_tl_y"] / downsample,
              "aa_w": row["aa_w"] / downsample,
              "aa_h": row["aa_h"] / downsample,
              "label": counter + 1,
              "time_str": row["time_str"]
              }

        bb["mean"] = np.asarray([row["c_x"] / downsample, row["c_y"] / downsample])

        bb["cov"] = np.eye(2)
        if "covar" in row:
            bb["cov"][0, 0] = row["var_x"] / (downsample ** 2)
            bb["cov"][1, 1] = row["var_y"] / (downsample ** 2)
            bb["cov"][0, 1] = bb["cov"][1, 0] = row["covar"] / (downsample ** 2)

        bb["confidence"] = 1.0
        if "confidence" in row:
            bb["confidence"] = row["confidence"]

        if filterFish(bb, camId, ROI_Area):
        # if True:
            bbs.append(bb)
            kps.append(
                cv2.KeyPoint(float(row["x"] / downsample), float(row["y"] / downsample), 1, 0, -1, -1, counter + 1))
            counter += 1
        else:
            continue
    return kps, bbs


def readTrackCSV(df, downsample=1):
    kps = []
    bbs = []
    counter = 0
    for i, row in df.iterrows():
        bb = {
            "id": row["id"],
              "x": row["x"] / downsample,
              "y": row["y"] / downsample,
              "tl_x": row["tl_x"] / downsample,
              "tl_y": row["tl_y"] / downsample,
              "c_x": row["c_x"] / downsample,
              "c_y": row["c_y"] / downsample,
              "w": row["w"] / downsample,
              "h": row["h"] / downsample,
              "theta": row["theta"],
              "l_x": row["l_x"] / downsample,
              "l_y": row["l_y"] / downsample,
              "r_x": row["r_x"] / downsample,
              "r_y": row["r_y"] / downsample,
              "aa_tl_x": row["aa_tl_x"] / downsample,
              "aa_tl_y": row["aa_tl_y"] / downsample,
              "aa_w": row["aa_w"] / downsample,
              "aa_h": row["aa_h"] / downsample,
              # "time_str": row["time_str"]
              }

        bbs.append(bb)
        kps.append(
            cv2.KeyPoint(
                float(row["x"] / downsample),
                float(row["y"] / downsample),
                1, 0, -1, -1, counter + 1
            )
        )
        counter += 1
    return kps, bbs
