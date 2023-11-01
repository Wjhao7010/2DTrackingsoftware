import datetime
import os
from multiprocessing import Pool
import sys
import argparse


sys.path.append("../../")
sys.path.append("../")
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--DayTank", default="D1_T1", type=str)
    args = ap.parse_args()
    # 是否需要并行运行
    if_parallel = False
    # gpustr = [4,5,6,7]
    if (platform.system() == 'Windows'):
        exp_floder = os.path.join(f"E:\\data\\3D_pre\\{args.DayTank}")
        code_path = "E:/Project/PyQt5/ZeF/3D-ZeF1"
    elif (platform.system() == 'Linux'):
        try:
            code_path = "."
            exp_floder = os.path.join(f"/home/data/HJZ/zef/{args.DayTank}")
        except:
            code_path = "."
            exp_floder = os.path.join(f"/home/huangjinze/code/data/zef/{args.DayTank}")

    processed_list = []
    format_video_names = []

    for files in os.listdir(f"{exp_floder}/cut_video/"):
        if files.endswith(".avi"):
            format_video_names.append(f"{exp_floder}/cut_video/{files}")

    # 针对每个隔，创建文件夹 1_1，并且添加该文件夹下的数据文件名
    process_floder = f"{exp_floder}/background/"

    if not os.path.isdir(process_floder):
        os.makedirs(process_floder)

    for files in os.listdir(process_floder):
        processed_list.append(files)

    print("check processed list: ")
    print(processed_list)
    print("============================================")
    # 需要执行的命令列表
    cmds = []
    for idx, ivideo in enumerate(format_video_names):
        video_floder, video_name = os.path.split(ivideo)
        camNO = video_name.split(".")[0].split("_")[-1]
        camId = camera_id_map[camNO]
        video_nameT = '_'.join(video_name.split(".")[0].split("_")[: -1])
        bg_filename = f'{video_nameT}_cam{camId}.jpg'
        if bg_filename in processed_list:
            continue
        else:
            cmd_str = f"python {code_path}/modules/detection/ExtractBackground.py " \
                      f"--path {exp_floder} " \
                      f"--video_name {video_name} "

            cmds.append(cmd_str)

    # cmds = list(set(cmds))
    print("*****************************************************")
    print(f"{len(cmds)} are need to be processed")
    if (platform.system() == 'Windows'):
        print(cmds)
        # exit(333)
    print("*****************************************************")

    # if if_parallel:
    #     # 并行
    #     if (platform.system() == 'Windows'):
    #         print(cmds)
    #         pool = Pool(2)
    #     else:
    #         pool = Pool(10)
    #     pool.map(execCmd, cmds)
    #     pool.close()  # 关闭进程池，不再接受新的进程
    #     pool.join()
    # else:
    #     # 串行
    #     for cmd in cmds:
    #         try:
    #             print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
    #             os.system(cmd)
    #             print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
    #         except:
    #             print('%s\t 运行失败' % (cmd))
