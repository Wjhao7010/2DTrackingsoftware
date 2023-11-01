import argparse
import os
from common.utility import *

tank_drag_map = {
    'D1_T1': '6PPD1ppm', 'D1_T2': '6PPD500ppb', 'D1_T3': '6PPD50ppb',
    'D2_T1': '6PPD1ppm', 'D2_T2': '6PPD500ppb', 'D2_T3': '6PPD50ppb',
    'D3_T1': '6PPD1ppm', 'D3_T2': '6PPD500ppb', 'D3_T3': '6PPD50ppb',
    'D1_T4': 'RJ',       'D1_T5': 'CK',         'D4_T1': '4Hydroxy500ppb',
    'D2_T4': 'RJ',       'D2_T5': 'CK',         'D5_T1': '4Hydroxy500ppb',
    'D3_T4': 'RJ',       'D3_T5': 'CK',         'D6_T1': '4Hydroxy500ppb',
    'D4_T2': '4Hydroxy50ppb', 'D4_T4': '6PPDQ500ppb', 'D4_T5': '6PPDQ50ppb',
    'D5_T2': '4Hydroxy50ppb', 'D5_T4': '6PPDQ500ppb', 'D5_T5': '6PPDQ50ppb',
    'D6_T2': '4Hydroxy50ppb', 'D6_T4': '6PPDQ500ppb', 'D6_T5': '6PPDQ50ppb',
    'D7_T1': '4Hydroxy1ppb', 'D4_T3': '6PPDQ1ppm',
    'D7_T2': '4Hydroxy1ppb', 'D7_T5': '6PPDQ1ppm',
    'D7_T4': '4Hydroxy1ppb', 'D8_T2': '6PPDQ1ppm',
    'D8_T4': '6PPDQ1ppm'

}
drag_tank_map = {
    '6PPD1ppm': {
        'D1_T1': {},
        'D2_T1': {},
        'D3_T1': {}
    },
    '6PPD500ppb': {
        'D1_T2': {},
        'D2_T2': {},
        'D3_T2': {}
    },
    '6PPD50ppb': {
        'D1_T3': {},
        'D2_T3': {},
        'D3_T3': {}
    },
    'RJ': {
        'D1_T4': {},
        'D2_T4': {},
        'D3_T4': {}
    },
    'CK': {
        'D1_T5': {},
        'D2_T5': {},
        'D3_T5': {}
    },
    '4Hydroxy500ppb': {
        'D4_T1': {},
        'D5_T1': {},
        'D6_T1': {}
    },
    '4Hydroxy50ppb': {
        'D4_T2': {},
        'D5_T2': {},
        'D6_T2': {}
    },
    '6PPDQ500ppb': {
        'D4_T4': {},
        'D5_T4': {},
        'D6_T4': {}
    },
    '6PPDQ50ppb': {
        'D4_T5': {},
        'D5_T5': {},
        'D6_T5': {}
    },
    '4Hydroxy1ppb': {
        'D7_T1': {},
        'D7_T2': {},
        'D7_T4': {}
    },
    '6PPDQ1ppm': {
        'D4_T3': {},
        'D7_T5': {},
        'D8_T2': {},
        'D8_T4': {}
    },
}

def get_tank_drag_map(root_path):
    '''

    :param root_path:
    :return: {
        ppd1: {
            D1_T1: {
                expose_T1: [file1, file2, file3,....],
                expose_T2: [file1, file2, file3,....],
            }
        }
    }
    '''
    for dayTank, drag in tank_drag_map.items():
        video_path = os.path.join(root_path, dayTank, 'cut_video')
        cut_video_file = sorted([_.split(".")[0] for _ in os.listdir(video_path) if _.endswith('avi')])
        drag_tank_map[drag][dayTank] = {}

        if dayTank == 'D1_T5':
            drag_tank_map[drag][dayTank][0] = [
                '2021_10_11_21_40_00_ch13',
                '2021_10_11_21_40_00_ch14',
                '2021_10_11_21_40_00_ch16'
            ]
            for idx, video_name in enumerate(cut_video_file):
                if idx // (3*5)+1 not in drag_tank_map[drag][dayTank]:
                    drag_tank_map[drag][dayTank][idx // (3*5)+1] = [video_name]
                else:
                    drag_tank_map[drag][dayTank][idx // (3*5)+1].append(video_name)
        else:
            group_file = {}
            for idx, video_name in enumerate(cut_video_file):
                if len(cut_video_file) > 150: # 3分钟划分
                    if idx // (3*5) not in drag_tank_map[drag][dayTank]:
                        drag_tank_map[drag][dayTank][idx // (3*5)] = [video_name]
                    else:
                        drag_tank_map[drag][dayTank][idx // (3*5)].append(video_name)
                else:
                    if idx // (3*3) not in drag_tank_map[drag][dayTank]:
                        drag_tank_map[drag][dayTank][idx // (3*3)] = [video_name]
                    else:
                        drag_tank_map[drag][dayTank][idx // (3*3)].append(video_name)
    return drag_tank_map

def get_TrackFilePath(root_path, processer, filename):
    for iDayTank, drag in tank_drag_map.items():
        for no in ['1_', '2_', '3_', '4_']:
            trackfloder = os.path.join(root_path, iDayTank, processer, no+drag)
            trackfiles = os.listdir(trackfloder)
            if filename in trackfiles:
                return os.path.join(trackfloder, filename)

def get_filname(drag_name, exposeT):
    trace_filename_list = []
    for DayTank, exposeInfo in get_tank_drag_map(root_path)[drag_name].items():
        for iexposeT, file_list in exposeInfo.items():
            if iexposeT == exposeT:
                trace_filename_list.extend(file_list)
    return trace_filename_list

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-rp", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("-drag", "--drag_name", default="4Hydroxy500ppm", type=str)
    ap.add_argument("-epT", "--exposeT", default=1, type=int, help='0,1,2,...,12')
    ap.add_argument("-trker", "--tracker", default="processed", type=str)

    args = vars(ap.parse_args())
    root_path = args['root_path']
    drag_name = args['drag_name']
    tracker = args['tracker']
    exposeT = args['exposeT']

    trace_filename_list = get_filname(drag_name, exposeT)
    trace_file_list = []
    for ifile in trace_filename_list:
        trace_file_list.append(get_TrackFilePath(root_path, tracker, ifile))
    print(trace_file_list)



