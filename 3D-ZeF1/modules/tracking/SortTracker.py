import os, sys

sys.path.append("../../")
sys.path.append("../")
sys.path.append(".")
import argparse
from filterpy.kalman import KalmanFilter
from modules.tracking.tools.utilits import *
from common.utility import *

np.random.seed(0)


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box. [x1, y1, x2, y2]
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # self.kf.F是状态变换模型
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1]
            ]
        )
        # self.kf.H是观测函数
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]
            ]
        )
        # self.kf.R为测量噪声矩阵
        self.kf.R[2:, 2:] *= 10.
        # self.kf.P为协方差矩阵
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        # self.kf.Q为过程噪声矩阵
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    # 跟踪器数量为0则直接构造结果。
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections[:, :4], trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    # 记录未匹配的检测框及轨迹
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # 返回当前边界框估计值。
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


def save2Dict(frameNo, bbox, identities=None, bbs=None):
    cur_frame_track_info = []
    for i, box in enumerate(bbox):
        id = int(identities[i]) if identities is not None else 0

        max_dist = 100000000000000
        bb = None
        for ib in bbs:
            tl_x = ib['tl_x']
            tl_y = ib['tl_y']
            br_x = ib['tl_x'] + ib['w']
            br_y = ib['tl_y'] + ib['h']
            dist = (tl_x - box[0]) ** 2 + (tl_y - box[1]) ** 2 + (br_x - box[2]) ** 2 + (br_y - box[3]) ** 2
            if dist < max_dist:
                bb = ib
                max_dist = dist
        if bb is None:
            cur_frame_track_info.append({
                'frame': frameNo,
                'id': np.nan,
                'cam': np.nan,
                'tl_x': np.nan,
                'tl_y': np.nan,
                'c_x': np.nan,
                'c_y': np.nan,
                'w': np.nan,
                'h': np.nan,
                'theta': np.nan,
                'aa_tl_x': np.nan,
                'aa_tl_y': np.nan,
                'aa_w': np.nan,
                'aa_h': np.nan,
                'l_x': np.nan,
                'l_y': np.nan,
                'r_x': np.nan,
                'r_y': np.nan,
            })
        else:
            cur_frame_track_info.append({
                'frame': frameNo,
                'id': id,
                "x": bb["x"],
                "y": bb["y"],
                'tl_x': bb['tl_x'],
                'tl_y': bb['tl_y'],
                'c_x': bb['c_x'],
                'c_y': bb['c_y'],
                'w': bb['w'],
                'h': bb['h'],
                'theta': bb['theta'],
                'aa_tl_x': bb['aa_tl_x'],
                'aa_tl_y': bb['aa_tl_y'],
                'aa_w': bb['aa_w'],
                'aa_h': bb['aa_h'],
                'l_x': bb['l_x'],
                'l_y': bb['l_y'],
                'r_x': bb['r_x'],
                'r_y': bb['r_y'],
            })
    return cur_frame_track_info


if __name__ == "__main__":
    import os, sys
    import pandas as pd

    ap = argparse.ArgumentParser()
    if (platform.system() == 'Windows'):
        ap.add_argument("-f", "--root_path", default="E:\\data\\3D_pre\\")
    elif (platform.system() == 'Linux'):
        ap.add_argument("-f", "--root_path", default="/home/data/HJZ/zef")

    ap.add_argument("-DT", "--DayTank", default="D1_T1", type=str)
    ap.add_argument("-RN", "--RegionName", default="2_6PPD1ppm", type=str)
    ap.add_argument("-nms", "--nsm_threshold", default=0.05, type=float)
    ap.add_argument("-saveVid", "--saveVideo", default=False)
    ap.add_argument("-df", "--detection_filename",
                    default="2021_10_11_22_29_59_ch01.csv",
                    help="Path to folder")

    args = vars(ap.parse_args())

    root_path = args["root_path"]
    DayTank = args["DayTank"]
    nsm_threshold = args["nsm_threshold"]
    RegionName = args["RegionName"]
    saveVideo = args["saveVideo"]
    detection_filename = args["detection_filename"]

    # 特殊处理
    if detection_filename in [
        '2021_10_12_08_49_59_ch02.csv',
        '2021_10_12_09_09_59_ch02.csv',
        '2021_10_12_09_29_59_ch02.csv',
        '2021_10_12_09_49_59_ch02.csv'
    ]:
        downsample = 1.5
    else:
        downsample = 1

    config_path = os.path.join(root_path, DayTank)

    camNO = detection_filename.split(".")[0].split("_")[-1]
    video_nameT = '_'.join(detection_filename.split(".")[0].split("_")[: -1])
    camId = camera_id_map[camNO]

    track_filename = detection_filename

    detection_file = os.path.join(
        root_path, DayTank, 'bg_processed',
        RegionName, detection_filename
    )
    trafloder = os.path.join(root_path, DayTank, 'sortTracker', RegionName)
    track_file = os.path.join(trafloder, track_filename)
    if not os.path.isdir(trafloder):
        os.makedirs(trafloder)

    # 以视频的形式存储结果
    if saveVideo:
        video_filename = detection_filename.replace(".csv", ".avi")
        vid_file = os.path.join(root_path, DayTank, 'cut_video', video_filename)
        vid_out_file = os.path.join(trafloder, video_filename)
        cap = cv2.VideoCapture(vid_file)
        fps = 25  # 视频帧率
        out_vid_size = (640, 480)
        videoWriter = cv2.VideoWriter(vid_out_file, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, out_vid_size)

    ROI_Area = load_EXP_region_pos_setting(config_path, camNO)[RegionName]



    # if os.path.isfile(detection_file):
    #     df_fish = pd.read_csv(detection_file, sep=",", index_col=None)
    #     df_fish = df_fish.loc[:, ~df_fish.columns.str.contains('^Unnamed')]
    #     if DayTank in ['D1_T2', 'D1_T3', 'D1_T4', 'D1_T5']:
    #         observe_frames = 3500
    #     else:
    #         observe_frames = 6000
    #     if df_fish.shape[0] < observe_frames:
    #         error_log = os.path.join(config_path, "error.txt")
    #         error_f = open(error_log, "a+")
    #         print(f"number of tracker_data is {df_fish.shape[0]} less than {observe_frames}")
    #         error_f.write(os.path.join(DayTank, 'cut_video', detection_filename) + f"/{RegionName}\n")
    #         # sys.exit()
    # else:
    #     # 没有检测的视频文件。需要记录在error.txt中
    #     print("Detections file found '{}' not found. Ending program".format(detection_file))
    #     error_log = os.path.join(config_path, "error.txt")
    #     error_f = open(error_log, "a+")
    #     error_f.write(os.path.join(DayTank, 'cut_video', detection_filename) + f"/{RegionName}\n")
    #     sys.exit()

    if not os.path.isfile(detection_file):
        print(f"detection file: {detection_file} not exits")
        exit(-1)
    df_fish = pd.read_csv(detection_file, sep=",", index_col=None)
    df_fish = df_fish.loc[:, ~df_fish.columns.str.contains('^Unnamed')]

    frameList = np.linspace(
        1, df_fish["frame"].max(), df_fish["frame"].max() - df_fish["frame"].min() + 1,
        True, dtype=np.int32
    )

    tracker = Sort(max_age=8, min_hits=2, iou_threshold=0.15)

    # 生成当前track中的所有trackid <class 'numpy.ndarray'>

    all_track_info = []

    for frameCount in frameList:
        if saveVideo:
            ret, frame = cap.read()

        if frameCount % 500 == 0:
            print(f"Frame: {frameCount} in {detection_filename}")

        fish_row = df_fish.loc[(df_fish['frame'] == frameCount)]
        index = py_cpu_nms(fish_row, nsm_threshold)
        fish_row = fish_row.iloc[index]

        if len(fish_row) == 0:
            cur_frame_infos = [{
                'frame': frameCount,
                'id': np.nan,
                'tl_x': np.nan,
                'tl_y': np.nan,
                'c_x': np.nan,
                'c_y': np.nan,
                'w': np.nan,
                'h': np.nan,
                'theta': np.nan,
                'aa_tl_x': np.nan,
                'aa_tl_y': np.nan,
                'aa_w': np.nan,
                'aa_h': np.nan,
                'l_x': np.nan,
                'l_y': np.nan,
                'r_x': np.nan,
                'r_y': np.nan,
            }]
            all_track_info.extend(cur_frame_infos)
            continue

        # 从检测目标中毒数据
        kps, bbs = readDetectionCSV(fish_row, camId, ROI_Area, downsample=downsample)
        np_box = []

        box_info = {}
        for ib in bbs:
            tl_x = ib['aa_tl_x']
            tl_y = ib['aa_tl_y']
            br_x = ib['aa_tl_x'] + ib['aa_w']
            br_y = ib['aa_tl_y'] + ib['aa_h']

            np_box.append([tl_x, tl_y, br_x, br_y])
        np_box = np.array(np_box)

        if len(np_box) == 0:
            cur_frame_infos = [{
                'frame': frameCount,
                'id': np.nan,
                'tl_x': np.nan,
                'tl_y': np.nan,
                'c_x': np.nan,
                'c_y': np.nan,
                'w': np.nan,
                'h': np.nan,
                'theta': np.nan,
                'aa_tl_x': np.nan,
                'aa_tl_y': np.nan,
                'aa_w': np.nan,
                'aa_h': np.nan,
                'l_x': np.nan,
                'l_y': np.nan,
                'r_x': np.nan,
                'r_y': np.nan,
            }]
            all_track_info.extend(cur_frame_infos)
            continue

        track_result = tracker.update(np_box)
        if track_result.shape[0] != 0:
            bbox_xyxy = track_result[:, :4]
            identities = track_result[:, -1]

            if saveVideo:
                for i in range(bbox_xyxy.shape[0]):
                    try:
                        cv2.rectangle(frame, (int(bbox_xyxy[i][0]), int(bbox_xyxy[i][1])),
                                      (int(bbox_xyxy[i][2]), int(bbox_xyxy[i][3])), (0, 0, 0), 1)
                        cv2.putText(frame, str(identities[i]),
                                    (int(bbox_xyxy[i][0]), int(bbox_xyxy[i][1])),
                                    cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 0, 255),
                                    1)
                    except:
                        continue
                out_frame = cv2.resize(frame, out_vid_size)
                videoWriter.write(out_frame)

            # 当前帧中的所有轨迹信息
            cur_frame_infos = save2Dict(frameCount, bbox_xyxy, identities, bbs)
        else:
            cur_frame_infos = [{
                'frame': frameCount,
                'id': np.nan,
                'tl_x': np.nan,
                'tl_y': np.nan,
                'c_x': np.nan,
                'c_y': np.nan,
                'w': np.nan,
                'h': np.nan,
                'theta': np.nan,
                'aa_tl_x': np.nan,
                'aa_tl_y': np.nan,
                'aa_w': np.nan,
                'aa_h': np.nan,
                'l_x': np.nan,
                'l_y': np.nan,
                'r_x': np.nan,
                'r_y': np.nan,
            }]
        all_track_info.extend(cur_frame_infos)

    if len(all_track_info) > 0:
        output_df = pd.DataFrame(all_track_info)
        output_df['cam'] = camId
        print(f"saved tracking result in {track_file}")
        if saveVideo:
            print(f"results are saved in {vid_out_file}")

        output_df.to_csv(track_file, sep=",")
    else:
        print("no track")
