import datetime
import os, sys
from multiprocessing import Pool

sys.path.append("../../")
sys.path.append(".")
from common.utility import *


# python中的多线程无法利用多核优势，
# 如果想要充分地使用多核CPU的资源，
# 在python中大部分情况需要使用多进程。
# Python提供了multiprocessing。

def execCmd(cmd):
    try:
        print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
    except:
        print('%s\t 运行失败' % (cmd))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--DayTank", default="D1_T4", type=str)
    ap.add_argument("--gpuno", default="1", type=str)
    args = ap.parse_args()
    # 是否需要并行运行
    if_parallel = True

    ips = getIPAddrs()
    if '10.2.151.127' in ips:
        exp_floder = f'/home/huangjinze/code/data/zef/{args.DayTank}'
    elif '10.2.151.128' in ips:
        exp_floder = f'/home/huangjinze/code/data/zef/{args.DayTank}'
    elif '10.2.151.129' in ips:
        exp_floder = f'/home/data/HJZ/zef/{args.DayTank}'
    else:
        exp_floder = os.path.join(f"E:\\data\\3D_pre\\{args.DayTank}")
    save_floder = os.path.join(exp_floder, 'cut_video')
    if not os.path.exists(save_floder):
        os.makedirs(save_floder)
    # D01_2021_09_18_20_59_59.mp4
    config_folder = exp_floder

    # 一个video要切 cut_number 段
    # 没一段长 split_time秒

    split_time = 3*60+10
    # split_time = 1
    process_list = os.listdir(save_floder)
    print(len(process_list))

    raw_video_list = []

    for video_name in os.listdir(exp_floder):
        if video_name.endswith(".mp4") or video_name.endswith(".mkv") or video_name.endswith(".avi") or video_name.endswith(".mov"):
            camNO = video_name.split("_")[0]
            start_time_list = load_Video_start_time(config_folder, video_name)
            for start_time in start_time_list:
                print(f"{start_time}_{camNO}.avi")
                # 从文件名获取视频开始时刻
                if f"{start_time}_{camNO}.avi" in process_list:
                    continue
                else:
                    raw_video_list.append([exp_floder, video_name, start_time])

    print(raw_video_list)

    # 吧 raw_video_list 中的元组按照 start_time 进行排序
    cmds = []
    for idx, (root, ivideo, start_time) in enumerate(sorted(raw_video_list, key=lambda cmd_info: cmd_info[1])):
        # save_path = os.path.join(save_floder, f"{camNO}_{start_time}")
        # gpuno = idx % len(gpustr)
        cmd_str = f"python modules/dataset_processing/CutTankjump.py " \
            f"--exp_floder {root} " \
            f"--save_path {save_floder} " \
            f"--video_name {ivideo} " \
            f"--start_time {start_time} " \
            f"--split_time {split_time} " \
            f"--gpuno {args.gpuno} "
            # f"--gpuno {gpustr[gpuno]} "
        cmds.append(cmd_str)

    print("*****************************************************")
    print(f"{len(cmds)} are need to be processed")
    print("*****************************************************")
    # exit(333)
    if if_parallel:
        # 并行
        pool = Pool(4)
        pool.map(execCmd, cmds)
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()
    else:
        # 串行
        for cmd in cmds:
            try:
                print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
                os.system(cmd)
                print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
            except:
                print('%s\t 运行失败' % (cmd))
