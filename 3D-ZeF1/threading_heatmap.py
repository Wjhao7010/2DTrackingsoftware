import os, sys
from multiprocessing import Pool
sys.path.append("../../")
sys.path.append("../")
sys.path.append(".")
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
    import argparse
    if_parallel = False

    ap = argparse.ArgumentParser()

    ap.add_argument("-c", "--code_path", default="E:\\Project\\PyQt5\\ZeF\\3D-ZeF1")
    ap.add_argument("-f", "--root_path", default="E:\\data\\3D_pre\\")
    # ap.add_argument("-DT", "--DayTank", default="D4_T4", type=str)
    # ap.add_argument("--RegionName", default='1_6PPDQ500ppb', help="region name of experience")
    ap.add_argument("--tracker", default='finalTrack', help="tracker name of experience")
    ap.add_argument("--exposureT", default='0,1,2,3,4,5,6,7,8,9,10,11', type=str)
    ap.add_argument("--pic_type", default='hillmap', type=str)
    args = ap.parse_args()

    if(platform.system()=='Windows'):
        code_path = args.code_path
        root_path = args.root_path

    # 需要执行的命令列表
    cmds = []
    for DayTank, drag_name in tank_drag_map.items():
        for ireg_no in ['1', '2', '3', '4']:
            cmd_str = f"python {code_path}/modules/visualization/heatmapShow.py " \
                f"--root_path {root_path} " \
                f"--DayTank {DayTank} " \
                f"--RegionName {ireg_no}_{drag_name} " \
                f"--tracker {args.tracker} " \
                f"--exposureT {args.exposureT} " \
                f"--pic_type {args.pic_type} "
            cmds.append(cmd_str)

    print("*****************************************************")
    print(f"{len(cmds)} are need to be processed")
    print(cmds)
    # exit(333)
    print("*****************************************************")

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


