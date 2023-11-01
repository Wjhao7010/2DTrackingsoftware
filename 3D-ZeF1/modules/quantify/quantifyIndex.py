import cv2
import numpy as np
import math, sys

sys.path.append("../../")
sys.path.append("../")
sys.path.append(".")
import argparse
from common.utility import *


# =============================== 计算距离 =============================#
def cal_id_velocity(track_id_info, map_ratio=1):
    # 计算相邻两帧之间的差
    diff_id_data = track_id_info[['frame', 'c_x', 'c_y']].diff(1).dropna()
    dist = diff_id_data['c_x'] ** 2 + diff_id_data['c_y'] ** 2
    diff_frame_list = diff_id_data['frame'].values
    # 这里要除以帧差
    frame_v = np.sqrt(dist) / diff_frame_list
    # 除以图像到真是世界的映射比例
    id_velocity = frame_v.mean() / map_ratio
    return id_velocity


def cal_group_velocity(track_info, trackid_list, map_ratio=1):
    group_velocity = []
    # 每帧的均值
    # id_velocity = {}
    for track_id in trackid_list:
        # 筛选id为 track_id的数据
        track_id_info = track_info[track_info['id'] == track_id]
        v = cal_id_velocity(track_id_info, map_ratio)
        # id_velocity[track_id] = v
        group_velocity.append(v)
    group_v = np.mean(group_velocity)
    return group_v


def cal_id_path(track_id_info, map_ratio=1):
    # 计算相邻两帧之间的差
    diff_id_data = track_id_info[['frame', 'c_x', 'c_y']].diff(1).dropna()
    dist_sq = diff_id_data['c_x'] ** 2 + diff_id_data['c_y'] ** 2
    dist = dist_sq ** (1 / 2.0)
    id_dist = dist.sum() / map_ratio
    return id_dist


def cal_group_path(track_info, trackid_list, map_ratio=1):
    group_path = []
    for track_id in trackid_list:
        track_id_info = track_info[track_info['id'] == track_id]
        id_path = cal_id_path(track_id_info, map_ratio)
        group_path.append(id_path)
    group_v = np.mean(group_path)
    return group_v


# =============================== 计算change angle =============================#

def cal_id_change_angle(track_id_info, interval=(0, 10, 20, 30, 40, 50, 60, 70, 80, 91)):
    # 筛选id为 track_id的数据
    change_angle = track_id_info['theta'].diff(1)
    change_angle.dropna(inplace=True)

    id_dist_angle = {}
    for i in range(len(interval)-1):
        low_bound = interval[i]
        up_bound = interval[i+1]
        condition = (change_angle.abs() >= low_bound) & (change_angle.abs() < up_bound)
        id_dist_angle[f"{low_bound}_{up_bound}"] = change_angle[condition].shape[0] / change_angle.shape[0]
    return id_dist_angle


def cal_id_moving_angle(track_id_info, interval=(0, 15, 45, 91)):
    # 筛选id为 track_id的数据
    moving_angle = track_id_info[['theta', 'w', 'h']].copy()
    moving_angle.dropna(inplace=True)

    id_dist_angle = {}
    for i in range(len(interval)-1):
        low_bound = interval[i]
        up_bound = interval[i+1]
        condition = (moving_angle['theta'].abs() >= low_bound) & (moving_angle['theta'].abs() < up_bound) & (moving_angle['w'] / moving_angle['h'] > 3)
        id_dist_angle[f"{low_bound}_{up_bound}"] = moving_angle[condition].shape[0] / moving_angle.shape[0]
    return id_dist_angle

def cal_group_angle(track_info, trackid_list, interval=(0, 10, 20, 30, 40, 50, 60, 70, 80, 91), type='change'):
    group_angle = {}
    for track_id in trackid_list:
        track_id_info = track_info[track_info['id'] == track_id]
        # 计算角度的变化率
        if type == 'change':
            id_angle = cal_id_change_angle(track_id_info, interval)
        # 计算角度的分布情况
        else:
            id_angle = cal_id_moving_angle(track_id_info, interval)
        for irange, angle in id_angle.items():
            if irange not in group_angle:
                group_angle[irange] = []
            group_angle[irange].append(id_angle[irange])

    for irange, anglelist in group_angle.items():
        group_angle[irange] = np.mean(group_angle[irange])
    return group_angle


# =============================== 计算区域分部 =============================#
def get_block_area(water_ImgPos, board_coef=5.0):
    out_tl_x, out_tl_y, out_br_x, out_br_y = water_ImgPos
    w = out_br_x - out_tl_x
    h = out_br_y - out_tl_y

    pos_area = {}
    w_iblock = w / board_coef
    h_iblock = h / board_coef

    for row in range(int(board_coef)):
        inner_tl_x = out_tl_x
        inner_tl_y = out_tl_y + row * h_iblock
        for col in range(int(board_coef)):
            tl_x = inner_tl_x + w_iblock * col
            br_x = inner_tl_x + w_iblock * (col + 1)
            tl_y = inner_tl_y
            br_y = inner_tl_y + h_iblock
            pos_area[(tl_x, tl_y, br_x, br_y)] = f'block_{int(row * board_coef + col + 1)}'
    return pos_area


def cal_POSDist(pos_area, it_tracker_data):
    pos_dist = {}
    for area_pos, iblock in pos_area.items():
        pos_dist[iblock] = 0
        inner_tl_x = area_pos[0]
        inner_tl_y = area_pos[1]
        inner_br_x = area_pos[2]
        inner_br_y = area_pos[3]

        # condition1 = (it_tracker_data['l_x'] >= inner_tl_x) | \
        #              (it_tracker_data['c_x'] >= inner_tl_x) | \
        #              (it_tracker_data['r_x'] >= inner_tl_x)
        condition1 = (it_tracker_data['c_x'] >= inner_tl_x)
        # condition2 = (it_tracker_data['l_y'] >= inner_tl_y) | \
        #              (it_tracker_data['c_y'] >= inner_tl_y) | \
        #              (it_tracker_data['r_y'] >= inner_tl_y)
        condition2 = (it_tracker_data['c_y'] >= inner_tl_y)
        # condition3 = (it_tracker_data['l_x'] <= inner_br_x) | \
        #              (it_tracker_data['c_x'] <= inner_br_x) | \
        #              (it_tracker_data['r_x'] <= inner_br_x)
        condition3 = (it_tracker_data['c_x'] <= inner_br_x)
        # condition4 = (it_tracker_data['l_y'] <= inner_br_y) | \
        #              (it_tracker_data['c_y'] <= inner_br_y) | \
        #              (it_tracker_data['r_y'] <= inner_br_y)
        condition4 = (it_tracker_data['c_y'] <= inner_br_y)
        board_data = it_tracker_data[(condition1) & (condition2) & (condition3) & (condition4)]
        counter = board_data.shape[0] / it_tracker_data.shape[0] if it_tracker_data.shape[0] > 0 else np.nan
        pos_dist[iblock] += counter
    return pos_dist


# =============================== 顶部时间 =============================#
def cal_top_prec(water_ImgPos, it_tracker_data, deep_coef=4):
    itl_x, itl_y, ibr_x, ibr_y = water_ImgPos
    water_Depth = ibr_y - itl_y
    top_water_line = itl_y + water_Depth // deep_coef

    top_data = it_tracker_data[it_tracker_data['c_y'] <= top_water_line]
    return top_data.shape[0] / it_tracker_data.shape[0] if it_tracker_data.shape[0] != 0 else np.nan


# =============================== body angle =============================#
def cal_body_angle(it_tracker_data, angle_range):
    min1, max1 = [int(_) for _ in angle_range.split("_")]
    min2 = -max1
    max2 = -min1
    filter_condition = (
            (
                    ((it_tracker_data['theta'] <= max1) & (it_tracker_data['theta'] >= min1)) |
                    ((it_tracker_data['theta'] <= max2) & (it_tracker_data['theta'] >= min2))
            ) &
            (
                (it_tracker_data['w'] / it_tracker_data['h'] > 3)
            )
    )
    angle_data = it_tracker_data[filter_condition]
    return angle_data.shape[0] / it_tracker_data.shape[0] if it_tracker_data.shape[0] != 0 else np.nan


# =============================== 计算所有指标 =============================#
def calculateIndex(
        cfg_path, camNO, RegionName, tracker_data,
        turning_angle_interval=(0, 10, 20, 30, 40, 50, 60, 70, 80, 91),
        up_down_angle_interval=(0, 15, 45, 91),
        index_col=(
                'time_str',
                # 'dist', 'velocity', 'turning_angle',
                'top_time', 'up_down_angle'
        ), deep_coef=4, window_size=3
):
    # windows_size 窗口中的总路径，速度/每帧

    frameList = np.linspace(
        int(tracker_data["frame"].min()), int(tracker_data["frame"].max()),
        int(tracker_data["frame"].max()) - int(tracker_data["frame"].min()) + 1,
        True, dtype=np.int32
    )
    # 生成当前track中的所有frameid <class 'numpy.ndarray'>
    unique_frameid = list(frameList)

    water_ImgPos = load_EXP_region_pos_setting(
        cfg_path, camNO
    )[RegionName]

    index_info = {}
    for i in index_col:
        index_info[i] = []

    for idx_frameid in range(min(unique_frameid), max(unique_frameid) - window_size + 1, window_size):

        if (idx_frameid + 1) % 100 == 0:
            print(f"processing frame_id {idx_frameid}")
        # 存储每一帧中目标跟踪的信息，可能有多个目标
        # 取出 idx_frameid 需要的偏移量
        frameid_id_list = unique_frameid[idx_frameid: idx_frameid + window_size]
        # 偏移量对应的行索引
        trackid_Index_id_no = tracker_data['frame'].isin(frameid_id_list)
        # 当前帧偏移量 对应的 数据
        track_info = tracker_data[trackid_Index_id_no].copy()

        # 统计trackid出现的次数
        track_id_counts = track_info['id'].value_counts()
        # 找出出现次数等于window_size的track id
        trackid_list = track_id_counts[track_id_counts == window_size].index.tolist()
        # 保留下trackid的信息
        track_info = track_info[track_info['id'].isin(trackid_list)]
        # 计算node的特征表达

        # 如果选取的数据中切换过于频繁，导致无数据，则跳过
        if track_info.shape[0] == 0:
            print(f"no data in time: {idx_frameid} to {idx_frameid + window_size}")
            continue

        # 顶部视图计算 距离 速度
        # 求和
        index_info['time_str'].append(max(frameid_id_list))
        camId = camera_id_map[camNO]

        if 'dist' in index_col:
            map_ratio = load_dist_map(cfg_path, camId, RegionName)
            v = cal_group_path(track_info, trackid_list, map_ratio)
            # 选取当前 win_size 中最大的值作为win_size的代表， 这个window_size中走过的路径之和
            index_info['dist'].append(v)

        if 'velocity' in index_col:
            map_ratio = load_dist_map(cfg_path, camId, RegionName)
            v = cal_group_velocity(track_info, trackid_list, map_ratio)
            # 选取当前 win_size 中最大的值作为win_size的代表，这个window_size 中的每帧的速度均值
            index_info['velocity'].append(v)

        if 'turning_angle' in index_col:
            distangle = cal_group_angle(track_info, trackid_list, interval=turning_angle_interval, type='change')
            for block, count in distangle.items():
                if 'turning_angle_' + block not in index_info:
                    index_info['turning_angle_' + block] = []
                index_info['turning_angle_' + block].append(count)

        if 'up_down_angle' in index_col:
            distangle = cal_group_angle(track_info, trackid_list, interval=up_down_angle_interval, type='no_change')
            for block, count in distangle.items():
                if 'up_down_angle_' + block not in index_info:
                    index_info['up_down_angle_' + block] = []
                index_info['up_down_angle_' + block].append(count)

        # 顶部视图计算 距离 速度
        if 'pos_distribution' in index_col:
            pos_area = get_block_area(water_ImgPos, board_coef=board_coef)
            distpos = cal_POSDist(pos_area, track_info)
            for block, count in distpos.items():
                if 'pos_distribution_' + block not in index_info:
                    index_info['pos_distribution_' + block] = []
                index_info['pos_distribution_' + block].append(count)

        # 侧视图计算水面的时间 针对视角时 身体的角度
        if 'top_time' in index_col:
            top_prec = cal_top_prec(water_ImgPos, track_info, deep_coef=deep_coef)
            index_info['top_time'].append(top_prec)

    if 'turning_angle' in index_info:
        del index_info['turning_angle']
    if 'up_down_angle' in index_info:
        del index_info['up_down_angle']
    return index_info


def statisticIndex(IndexValue, time_interval=1, fps=25):
    # 跟踪数据
    IndexValue = pd.DataFrame(IndexValue).sort_values('time_str')
    IndexValue['timestamp_min_id'] = IndexValue['time_str'] // (fps * time_interval)
    statistic_way = {}
    for icol in IndexValue.columns.tolist():
        if 'turning_angle' in icol:
            statistic_way[icol] = 'mean'
        elif 'dist' in icol:
            statistic_way[icol] = 'sum'
        elif 'velocity' == icol:
            statistic_way[icol] = 'mean'
        elif 'top_time' == icol:
            statistic_way[icol] = 'mean'
        elif 'up_down_angle' in icol:
            statistic_way[icol] = 'mean'
        else:
            continue

    index_val = IndexValue.groupby('timestamp_min_id').agg(statistic_way)

    if 'velocity' in IndexValue.columns.tolist():
        mean_v = IndexValue[['velocity', 'timestamp_min_id']].groupby('timestamp_min_id').mean()
        index_val['dist_cal'] = mean_v * fps * time_interval
    return index_val


if __name__ == '__main__':
    edge_feature = ['dispersion_delaunay']
    node_feature = [
        'velocity', 'distance'
    ]
    ap = argparse.ArgumentParser()

    ap.add_argument("-rp", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("-DT", "--DayTank", default="D1_T1", type=str)
    # ap.add_argument("-fn", "--fish_num", default=1, type=int)
    ap.add_argument("-trker", "--tracker", default="finalTrack", type=str)
    ap.add_argument("-RN", "--RegionName", default="1_6PPD1ppm", type=str)
    ap.add_argument("-win_size", "--window_size", default=25, type=int)
    ap.add_argument("-tf", "--track_filename",
                    default="2021_10_11_21_49_59_ch03.csv",
                    help="Path to folder")

    args = vars(ap.parse_args())

    # fish_num = args["fish_num"]
    # 这里需要确定一下鱼体的长途
    fish_length = math.sqrt(((95 - 86) ** 2 + (429 - 476) ** 2))

    root_path = args["root_path"]
    DayTank = args["DayTank"]
    tracker = args["tracker"]
    RegionName = args["RegionName"]
    window_size = args["window_size"]
    track_filename = args["track_filename"]

    turning_angle_interval = (0, 10, 20, 30, 40, 50, 60, 70, 80, 91)
    up_down_angle_interval = (0, 15, 45, 91)

    cfg_path = os.path.join(root_path, DayTank)

    tracker_file = os.path.join(root_path, DayTank, tracker, RegionName, track_filename)
    camNO = tracker_file.split(".")[0].split("_")[-1]

    data = pd.read_csv(tracker_file, index_col=None)
    if data.shape[0] == 0:
        exit(-1)

    IndexValue = calculateIndex(
        cfg_path, camNO, RegionName, data,
        turning_angle_interval=turning_angle_interval,
        up_down_angle_interval=up_down_angle_interval,
        window_size=window_size
    )
    print(IndexValue)

    StatisticIndex = statisticIndex(IndexValue)
    print(StatisticIndex)
    # 输出指标的文件
    outputPath = os.path.join(root_path, DayTank, 'indicators', RegionName)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    indicator_file = track_filename
    outputfile = os.path.join(outputPath, indicator_file)
