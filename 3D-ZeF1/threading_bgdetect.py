import datetime
import os
from multiprocessing import Pool
import sys

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
    ap.add_argument("--DayTank", default="D1_T1", type=str)
    args = ap.parse_args()
    # 是否需要并行运行
    if_parallel = False

    ip_list = getIPAddrs()
    print(f"platform.system is {platform.system()}")
    print(f"ip list is {ip_list}")
    base_dir = getBaseDir(ip_list)

    exp_floder = os.path.join(base_dir, args.DayTank)
    config_folder = exp_floder
    camera_id_list = load_Cam_list(config_folder)
    region_names = load_EXP_region_name(config_folder)

    processed_list = []
    format_video_names = []

    for files in os.listdir(f"{exp_floder}/cut_video/"):
        if files.endswith(".avi"):
            format_video_names.append(f"{exp_floder}/cut_video/{files}")

    # 针对每个隔，创建文件夹 1_1，并且添加该文件夹下的数据文件名
    for iregion in region_names:
        process_floder = f"{exp_floder}/bg_processed/{iregion}"
        if not os.path.isdir(process_floder):
            os.makedirs(process_floder)

        # E:\data\3D_pre\0918-0919fish\processed\1_1/2021_09_18_19_52_03_D01.csv
        for files in os.listdir(process_floder):
            processed_list.append(f"{process_floder}/{files}".replace("\\", '/'))

    print("check processed list: ")
    print(processed_list)
    print("============================================")
    # 需要执行的命令列表
    command_log = os.path.join(config_folder, "command_list.txt")
    cammand_f = open(command_log, "w+")
    cmds = []
    for idx, ivideo in enumerate(format_video_names):
        video_floder, video_name = os.path.split(ivideo)
        region_str = ''
        for iregion in region_names:
            # process_floder: E:\data\3D_pre\0918-0919fish\processed
            # iregion: 2_CK
            # video_name: 2021_09_19_01_15_59_D02.avi
            process_floder = f"{exp_floder}/bg_processed/{iregion}"
            process_name = os.path.join(
                video_floder.replace("cut_video", 'bg_processed'),
                iregion,
                video_name.replace(".avi", ".csv")
            )
            camNO = video_name.split(".")[0].split("_")[-1]
            Exp_region_pos = load_EXP_region_pos_setting(config_folder, camNO)
            if iregion not in list(Exp_region_pos.keys()):
                print(f"{iregion} not in Exp_region_pos.keys")
                continue
            if process_name.replace("\\", '/') in processed_list:
                print(f"{process_name} has been processed")
                # continue
            else:
                cmd_str = f"python modules/detection/BgDetector.py " \
                          f"--path {exp_floder} " \
                          f"--video_name {video_name} " \
                          f"--region_name {iregion} "

                cammand_f.write(f"{cmd_str} -Debug\n")
                cmds.append(cmd_str)
                print(f"debug commands are saved in {config_folder}")

    cmds = list(set(cmds))
    print("*****************************************************")
    print(f"{len(cmds)} are need to be processed")
    if (platform.system() == 'Windows'):
        cmds_str = '\n'.join(cmds)
        print(cmds_str)
        # exit(333)
    print("*****************************************************")

    if if_parallel:
        # 并行
        if (platform.system() == 'Windows'):
            print(cmds)
            pool = Pool(4)
        else:
            pool = Pool(10)
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
