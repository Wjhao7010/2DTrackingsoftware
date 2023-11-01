'''
展示权重的重要性得分
'''

import os
import matplotlib.pyplot as plt
import pandas as pd

from pyecharts import options as opts
from pyecharts.charts import Timeline, Bar, HeatMap, Line, Page
from pyecharts.faker import Faker
from pyecharts.globals import ThemeType
import numpy as np

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, '../data/fishBehavior')
plt.rc('font', family='Times New Roman')
fontsize = 12.5

ANGLE_NAME = ['Angle_'+ str(_) for _ in range(0, 180, 10)]

ACC_NAME = ['AccSpeed_'+str(_) for _ in range(25)]


#####################################################################################

def getHeatMap(data, name):

    def formatHeatmapData(rdata):
        heat_data = []
        # rdata = np.around(rdata, decimals=3)
        for t in range(rdata.shape[0]):
            for a in range(rdata.shape[1]):
                heat_data.append([t, a, rdata[t][a]])
        return heat_data


    c = (
        HeatMap()
    )
    c.add_xaxis(ACC_NAME)
    for region_name, v in data.items():
        heat_data = formatHeatmapData(v.values)
        c.add_yaxis(
            region_name,
            ANGLE_NAME,
            heat_data,
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
    c.set_global_opts(
        title_opts=opts.TitleOpts(title=name),
        visualmap_opts=opts.VisualMapOpts(),
    )
    return c



if __name__ == '__main__':
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser()

    ap.add_argument("-tid", "--t_ID", default="D01")
    ap.add_argument("-lid", "--l_ID", default="D02")
    ap.add_argument("-rid", "--r_ID", default="D04")

    ap.add_argument("-iP", "--indicatorPath", default="E:\\data\\3D_pre/exp_pre/indicators_joint/")
    ap.add_argument("-o", "--outputPath", default="E:\\data\\3D_pre/exp_pre/results/")

    args = vars(ap.parse_args())

    interval = 10
    outputPath = args["outputPath"]
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    files = os.listdir(args["indicatorPath"])
    all_nos = []
    for ifile in files:
        no = ifile.split("_")[0]
        start_no, end_no, _ = no.split("-")
        str_start_no = start_no.zfill(4)
        str_end_no = end_no.zfill(4)
        if (str_start_no, str_end_no) in all_nos:
            continue
        else:
            all_nos.append((str_start_no, str_end_no))
    all_nos.sort()
    time_list = [_ for _ in range(0, int(all_nos[-1][1]))]

    total_data = {}
    name_list = [
        "1_1",
        "2_CK",
        "3_1",
        "4_1"
    ]

    all_data = []

    for idx, ino in enumerate(all_nos):
        if idx % interval == 0:
            acc_angle_data = {region_name: None for region_name in name_list}

            for RegionName in name_list:
                indicator_file = os.path.join(args["indicatorPath"], str(int(ino[0]))+"-"+str(int(ino[1]))+ "-joint_" + RegionName)
                print(indicator_file)
                data = pd.read_csv(indicator_file, index_col=0)
                acc_angle_data[RegionName] = data
                # velocity and distance
            all_data.append(acc_angle_data)
        else:
            continue

    page = Page(layout=Page.SimplePageLayout)
    for idx, idata in enumerate(all_data):
        page.add(
            getHeatMap(idata, f'time: {idx*interval}'),
        )
    page.render("angle_velocity_JointDistribution.html")