'''
get_tankDragMap函数对D1_T5做了特殊化处理
'''
# 计算的是暴露第x小时时，A区域的活动指标
import sys

sys.path.append("../../")
sys.path.append("../")
sys.path.append(".")
from modules.quantify.quantifyIndex import calculateIndex, statisticIndex
import os
import argparse
from common.utility import *
import json


def get_exposureTIndex(exposeT):
    trace_filename_list = get_filname(root_path, drag_name, exposeT)

    trace_file_list = []
    for ifile in trace_filename_list:
        trace_file_list.extend(get_trackFileInfo(root_path, tracker, ifile))
    print(trace_file_list)

    top_df = pd.DataFrame()
    side_df = pd.DataFrame()
    for itrackerInfo in trace_file_list:
        print(f"processing file {itrackerInfo['filepath']} at exposure time {exposeT}")

        data = pd.read_csv(itrackerInfo['filepath'], index_col=None)
        if data.shape[0] == 0:
            continue

        data.fillna(method='pad', axis=0, inplace=True)

        cfg_path = os.path.join(itrackerInfo['root_path'], itrackerInfo['DayTank'])

        region_name = itrackerInfo['region_name']
        camNO = itrackerInfo['camNO']
        camId = camera_id_map[camNO]

        if camId == 1:

            IndexValue = calculateIndex(
                cfg_path, camNO, region_name,
                data, index_col=top_index, deep_coef=4, window_size=3
            )
            statistic_val = statisticIndex(IndexValue, time_interval=1, fps=25)
            statistic_val['filename'] = itrackerInfo['filename']
            statistic_val['exp_time'] = itrackerInfo['exp_time']
            statistic_val['DayTank'] = itrackerInfo['DayTank']
            statistic_val['region_name'] = itrackerInfo['region_name']
            statistic_val['camNO'] = itrackerInfo['camNO']

            top_df = pd.concat([top_df, statistic_val])
        else:
            IndexValue = calculateIndex(
                cfg_path, camNO, region_name,
                data, index_col=side_index, deep_coef=4, window_size=3
            )
            statistic_val = statisticIndex(IndexValue, time_interval=1, fps=25)
            statistic_val.reset_index(inplace=True)
            statistic_val['filename'] = itrackerInfo['filename']
            statistic_val['exp_time'] = itrackerInfo['exp_time']
            statistic_val['DayTank'] = itrackerInfo['DayTank']
            statistic_val['region_name'] = itrackerInfo['region_name']
            statistic_val['camNO'] = itrackerInfo['camNO']

            side_df = pd.concat([side_df, statistic_val])

    return top_df, side_df


if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument("-rp", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("-drag", "--drag_name", default="6PPD1ppm", type=str)
    ap.add_argument("-epT", "--exposeT", default='0', type=str, help='time 0')
    ap.add_argument("-trker", "--tracker", default="finalTrack", type=str)

    exp_keys = ['DayTank', 'camNO', 'exp_time', 'filename', 'region_name']

    # 区域分成  board_coef * board_coef 块， top + h/ deep_coef之上 认为是水面
    board_coef = 5.0
    deep_coef = 4
    # 'time' 是索引
    top_index = ['time_str', 'velocity', 'dist', 'turning_angle']
    side_index = ['time_str', 'top_time', 'up_down_angle']
    assert 'time_str' in top_index, 'key:time not in top_index'
    assert 'time_str' in side_index, 'key:time not in side_index'
    args = vars(ap.parse_args())

    root_path = args['root_path']
    drag_name = args['drag_name']
    tracker = args['tracker']
    exposeTimes = args['exposeT']

    if not os.path.exists(os.path.join(root_path, 'drag_result', drag_name)):
        os.makedirs(os.path.join(root_path, 'drag_result', drag_name))
    top_index_file = os.path.join(root_path, 'drag_result', drag_name, f'{exposeTimes}_top.csv')
    side_index_file = os.path.join(root_path, 'drag_result', drag_name, f'{exposeTimes}_side.csv')

    drag_cfg_file = os.path.join(root_path, f'drag_config.json')
    if not os.path.isfile(drag_cfg_file):
        trace_filename_list = get_tankDragMap(root_path)
        with open(drag_cfg_file, 'w', encoding='utf8') as fp:
            json.dump(trace_filename_list, fp, ensure_ascii=False)
    else:
        with open(drag_cfg_file, 'r', encoding='utf8') as fp:
            trace_filename_list = json.load(fp)

    process_list = []
    for ifile in os.listdir(os.path.join(root_path, 'drag_result', drag_name)):
        process_list.append(os.path.join(root_path, 'drag_result', drag_name, ifile))
    if (top_index_file in process_list) and (side_index_file in process_list):
        print("file has been process")
        exit(0)

    top_df, side_df = get_exposureTIndex(int(exposeTimes))
    top_df['exposure_time'] = exposeTimes
    side_df['exposure_time'] = exposeTimes

    top_df.to_csv(top_index_file, index=False, sep=",")
    side_df.to_csv(side_index_file, index=False, sep=",")
    print(f"drag: time {exposeTimes} of {drag_name}'s results are saved in \n {side_index_file} and {top_index_file}")
