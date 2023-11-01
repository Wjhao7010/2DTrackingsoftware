import datetime
import sys, os
sys.path.append(".")
sys.path.append("../")
from common.utility import *


if __name__ == '__main__':

    for itank in [
        "D1_T1", "D1_T2", "D1_T3", "D1_T4", "D1_T5",
        "D2_T1", "D2_T2", "D2_T3", "D2_T4", "D2_T5",
        "D3_T1", "D3_T2", "D3_T3", "D3_T4", "D3_T5",
        "D4_T1", "D4_T2", "D4_T3", "D4_T4", "D4_T5",
        "D5_T1", "D5_T2", "D5_T4", "D5_T5",
        "D6_T1", "D6_T2", "D6_T4", "D6_T5",
        "D7_T1", "D7_T2", "D7_T4", "D7_T5",
        "D8_T2", "D8_T4",
    ]:

        ips = getIPAddrs()

        if '10.2.151.127' in ips:
            base_dir = '/home/huangjinze/code/data/zef'
        elif '10.2.151.128' in ips:
            base_dir = '/home/huangjinze/code/data/zef'
        elif '10.2.151.129' in ips:
            base_dir = '/home/data/HJZ/zef'

        cmd = f"mkdir {base_dir}/{itank}/bak && " \
              f"mv {base_dir}/{itank}/bg_processed {base_dir}/{itank}/bak/ && " \
              f"mv {base_dir}/{itank}/gapfill {base_dir}/{itank}/bak/ && " \
              f"mv {base_dir}/{itank}/processed {base_dir}/{itank}/bak/ && " \
              f"mv {base_dir}/{itank}/sortTracker {base_dir}/{itank}/bak/ && " \
              f"mv {base_dir}/{itank}/sortTracker-refine {base_dir}/{itank}/bak/ "
        print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
