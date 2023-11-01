import os, sys
from multiprocessing import Pool

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
    # 是否需要并行运行
    import argparse

    if_parallel = False

    ap = argparse.ArgumentParser()

    ap.add_argument("--code_path", default="E:\\Project\\PyQt5\\ZeF\\3D-ZeF1", type=str)
    ap.add_argument("--data_path", default="E:\\data\\3D_pre", type=str)
    ap.add_argument("--DayTank", default="D1_T1", type=str)
    ap.add_argument("--tracker", default="sortTracker", type=str)
    # ap.add_argument("--tracker", default="Hungary", type=str)
    ap.add_argument("--detector", default="bg_processed", type=str)
    args = ap.parse_args()

    if (platform.system() == 'Windows'):
        code_path = args.code_path
        data_path = args.data_path
        exp_floder = os.path.join(f"{data_path}\\{args.DayTank}")
        detect_floder = os.path.join(f"{data_path}\\{args.DayTank}\\{args.detector}")
        tracker_floder = os.path.join(f"{data_path}\\{args.DayTank}\\{args.tracker}")
    elif (platform.system() == 'Linux'):
        code_path = args.code_path
        ip_list = getIPAddrs()
        print(f"platform.system is {platform.system()}")
        print(f"ip list is {ip_list}")
        data_path = getBaseDir(ip_list)
        exp_floder = os.path.join(f"{data_path}/{args.DayTank}")
        detect_floder = os.path.join(f"{data_path}/{args.DayTank}/{args.detector}")
        tracker_floder = os.path.join(f"{data_path}/{args.DayTank}/{args.tracker}")

    config_folder = exp_floder
    region_names = load_EXP_region_name(config_folder)
    processed_list = []
    detect_files = []

    for region_name in region_names:
        processed_file = os.path.join(tracker_floder, region_name)
        if not os.path.isdir(processed_file):
            os.makedirs(processed_file)

        for itrack_file in os.listdir(processed_file):
            if itrack_file.endswith(".csv"):
                itrack_filepath = os.path.join(processed_file, itrack_file)
                processed_list.append(itrack_filepath)

        for file in os.listdir(os.path.join(detect_floder, region_name)):
            if not file.endswith('.csv'):
                continue
            camNO = file.split(".")[0].split("_")[-1]
            Exp_region_pos = load_EXP_region_pos_setting(config_folder, camNO)
            if region_name not in list(Exp_region_pos.keys()):
                print(f"{region_name} not in Exp_region_pos.keys")
                continue
            else:
                detect_file = os.path.join(detect_floder, region_name, file)
                detect_files.append(detect_file)

    # 需要执行的命令列表
    cmds = []
    for idetfile in sorted(detect_files):
        # itrackfile = idetfile.replace("processed", tracker_name)
        detect_path, filename = os.path.split(idetfile)
        trackpath = detect_path.replace(args.detector, args.tracker)

        trackfile = os.path.join(trackpath, filename)
        # print(trackfile)
        # itrackfile = os.path.join(track_path, track_filename)
        if trackfile in processed_list:
            print(f"{trackfile} has been processed")
            continue
        else:
            detect_path = detect_path.replace("\\", "/")
            RegionName = detect_path.split("/")[-1]
            cmd_str = f"python {code_path}/modules/tracking/{args.tracker}.py " \
                      f"--root_path {data_path} " \
                      f"--DayTank {args.DayTank} " \
                      f"--RegionName {RegionName} " \
                      f"--detection_filename {filename} "
            cmds.append(cmd_str)

    print("*****************************************************")
    print(f"{len(cmds)} are need to be processed")
    cmds_str = '\n'.join(cmds)
    print(cmds_str)
    # exit(3333)
    print("*****************************************************")

    # if if_parallel:
    #     # 并行
    #     pool = Pool(4)
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
