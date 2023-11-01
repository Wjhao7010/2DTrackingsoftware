import os, sys
from multiprocessing import Pool

from common.utility import *

def execCmd(cmd):
    try:
        print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
    except:
        print('%s\t 运行失败' % (cmd))

if __name__ == '__main__':
    # 是否需要并行运行
    if_parallel = False
    # 是否需要并行运行
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("--drag_names", default="6PPD1ppm", type=str)
    ap.add_argument("--code_path", default="E:\\Project\\PyQt5\\ZeF\\3D-ZeF1", type=str)
    ap.add_argument("--data_path", default="E:\\data\\3D_pre", type=str)
    # ap.add_argument("--drag_names", default="4Hydroxy1ppm,6PPD1ppm,6PPD500ppb,6PPD50ppb,RJ,CK,4Hydroxy500ppb,4Hydroxy50ppb,6PPDQ500ppb,6PPDQ50ppb,6PPDQ1ppm", type=str)
    # ap.add_argument("--drag_names", default="4Hydroxy500ppb,4Hydroxy50ppb,6PPDQ500ppb,6PPDQ50ppb,6PPDQ1ppm", type=str)
    ap.add_argument("--exposureT", default="0,1,2,3,4,5,6,7,8,9,10,11", type=str)
    ap.add_argument("--tracker", default="finalTrack", type=str)
    args = ap.parse_args()

    drag_names = args.drag_names.split(",")
    if type(drag_names) is not list:
        drag_names = [drag_names]
    exposureT = args.exposureT
    tracker = args.tracker

    if (platform.system() == 'Windows'):
        code_path = args.code_path
        exp_floder = args.data_path

    elif (platform.system() == 'Linux'):
        code_path = "."
        ip_list = getIPAddrs()
        print(f"platform.system is {platform.system()}")
        print(f"ip list is {ip_list}")
        exp_floder = getBaseDir(ip_list)

    cmds = []
    for drag in drag_names:
        for expT in exposureT.split(","):
            cmd_str = f"python {code_path}/modules/analysis/indexAnalysis.py " \
                    f"-rp {exp_floder} " \
                    f"-drag {drag} " \
                    f"-epT {expT} " \
                    f"-trker {tracker} "
            cmds.append(cmd_str)

    print("*****************************************************")
    print(f"{len(cmds)} are need to be processed")
    print("*****************************************************")
    if if_parallel:
        # 并行
        pool = Pool(20)
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


