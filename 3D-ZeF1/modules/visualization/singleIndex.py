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

ANGLE_NAME = ['Angle_0.0', 'Angle_20.0', 'Angle_40.0', 'Angle_60.0', 'Angle_80.0', 'Angle_100.0', 'Angle_120.0',
             'Angle_140.0', 'Angle_160.0']

ACC_NAME = ['AccSpeed_0.0','AccSpeed_2.0','AccSpeed_4.0','AccSpeed_6.0','AccSpeed_8.0']

def format_data(data: pd.DataFrame, time_list: list, name_list: list) -> dict:
    data = data.T.to_dict()
    fdata = {}
    for t_id, vdata in data.items():
        fdata[time_list[t_id]] = [v for region, v in vdata.items()]

    for min_t in time_list:
        temp = fdata[min_t]
        for i in range(len(temp)):
            fdata[min_t][i] = {"name": name_list[i], "value": temp[i]}
    return fdata


#####################################################################################
# 2002 - 2011 年的数据
def get_year_overlap_chart(total_data, time_mim: int) -> Bar:
    bar = (
        Bar()
            .add_xaxis(xaxis_data=name_list)

    )
    bar.add_yaxis(
        series_name="velocity",
        y_axis=total_data["velocity"][time_mim],
        is_selected=True,
        label_opts=opts.LabelOpts(is_show=False),
        stack=f'stack1'
    )
    bar.add_yaxis(
        series_name="distance",
        y_axis=total_data["distance"][time_mim],
        is_selected=True,
        label_opts=opts.LabelOpts(is_show=False),
        stack=f'stack1'
    )
    bar.add_yaxis(
        series_name="velocity",
        y_axis=total_data["velocity"][time_mim],
        is_selected=True,
        label_opts=opts.LabelOpts(is_show=False),
        stack=f'stack2'
    )
    bar.add_yaxis(
        series_name="distance",
        y_axis=total_data["distance"][time_mim],
        is_selected=True,
        label_opts=opts.LabelOpts(is_show=False),
        stack=f'stack2'
    )

    # print(total_data["bottom_time"][time_mim])
    # print(Faker.values())
    # exit(33)
    # bar.add_yaxis("moving time", [31, 58, 80, 26], stack="stack1", category_gap="50%")
    # bar.add_yaxis("static time", [31, 58, 80, 26], stack="stack1", category_gap="50%")

    bar.set_global_opts(
        title_opts=opts.TitleOpts(
            title="{}分钟后，斑马鱼运动指标".format(time_mim)
        ),
        datazoom_opts=opts.DataZoomOpts(),
        tooltip_opts=opts.TooltipOpts(
            is_show=True, trigger="axis", axis_pointer_type="shadow"
        ),
    )

    return bar


def getLine(v_data, name):
    l = (
        Line()
            .add_xaxis(xaxis_data=[str(_) for _ in time_list])
            .add_yaxis(
            series_name="1_1",
            y_axis=v_data['1_1'],
            label_opts=opts.LabelOpts(is_show=False),
        )
            .add_yaxis(
            series_name="2_CK",
            y_axis=v_data['2_CK'],
            label_opts=opts.LabelOpts(is_show=False),
        )
            .add_yaxis(
            series_name="3_1",
            y_axis=v_data['3_1'],
            label_opts=opts.LabelOpts(is_show=False),
        )
            .add_yaxis(
            series_name="4_1",
            y_axis=v_data['4_1'],
            label_opts=opts.LabelOpts(is_show=False),
        )
            .set_series_opts(
                areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
            title_opts=opts.TitleOpts(title=name),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=opts.DataZoomOpts(),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )
    return l


def getStackBar(top_data, bottom_time, name1, name2, name):
    def format(t):
        region = {}
        for i in name_list:
            td = t[i].values
            list1 = []
            for v in td:
                list1.append({
                    "value": v,
                    "percent": v,
                })
            region[i] = list1
        return region

    td = format(top_data)
    bd = format(bottom_time)

    c = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add_xaxis(["Time " + str(_) + ":" + "/".join(name_list) for _ in time_list])
    )
    for idx, i in enumerate(name_list):
        c.add_yaxis(name1, td[i], stack=f'stack{idx}')
        c.add_yaxis(name2, bd[i], stack=f'stack{idx}')

    c.set_series_opts(
        label_opts=opts.LabelOpts(is_show=False)
    )
    c.set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
        datazoom_opts=opts.DataZoomOpts(),

        title_opts=opts.TitleOpts(title=name)
    )
    return c

def getHeatMap(data, time_list, name):

    def formatHeatmapData(rdata):
        heat_data = []
        rdata = np.around(rdata, decimals=3)
        for t in range(rdata.shape[0]):
            for a in range(rdata.shape[1]):
                heat_data.append([t, a, rdata[t][a]])
        return heat_data


    c = (
        HeatMap()
    )
    c.add_xaxis(time_list)
    for region_name, v in data.items():
        heat_data = formatHeatmapData(data[region_name].values)
        if 'Acceleration' in name:
            c.add_yaxis(
                region_name,
                ACC_NAME,
                heat_data,
                label_opts=opts.LabelOpts(is_show=True, position="inside"),
            )
        elif 'Angle' in name:
            c.add_yaxis(
                region_name,
                ANGLE_NAME,
                heat_data,
                label_opts=opts.LabelOpts(is_show=True, position="inside"),
            )
    c.set_global_opts(
        title_opts=opts.TitleOpts(title=name),
        datazoom_opts=opts.DataZoomOpts(),
        visualmap_opts=opts.VisualMapOpts(min_=0, max_=1),
    )
    return c



if __name__ == '__main__':
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser()

    ap.add_argument("-tid", "--t_ID", default="D01")
    ap.add_argument("-lid", "--l_ID", default="D02")
    ap.add_argument("-rid", "--r_ID", default="D04")

    ap.add_argument("-iP", "--indicatorPath", default="E:\\data\\3D_pre/exp_pre/indicators/")
    ap.add_argument("-o", "--outputPath", default="E:\\data\\3D_pre/exp_pre/results/")

    args = vars(ap.parse_args())

    outputPath = args["outputPath"]
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    files = os.listdir(args["indicatorPath"])
    all_nos = []
    for ifile in files:
        no = ifile.split("_")[0]
        start_no, end_no = no.split("-")
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
    v_data = pd.DataFrame()
    d_data = pd.DataFrame()
    top_data = pd.DataFrame()
    bottom_time = pd.DataFrame()
    stop_time = pd.DataFrame()
    moving_time = pd.DataFrame()
    angle_data = {region_name: None for region_name in name_list}
    acc_data = {region_name: None for region_name in name_list}

    for ino in all_nos:
        no_v_data = pd.DataFrame()
        no_d_data = pd.DataFrame()
        no_top_data = pd.DataFrame()
        no_bottom_time = pd.DataFrame()
        no_stop_time = pd.DataFrame()
        no_moving_time = pd.DataFrame()

        for RegionName in name_list:
            indicator_file = os.path.join(args["indicatorPath"], str(int(ino[0]))+"-"+str(int(ino[1]))+ "_" + RegionName)
            print(indicator_file)
            data = pd.read_csv(indicator_file)
            # velocity and distance
            no_v_data = pd.concat([no_v_data, data[['velocity']]], axis=1)
            no_d_data = pd.concat([no_d_data, data[['distance']]], axis=1)
            no_top_data = pd.concat([no_top_data, data[['top_time']]], axis=1)
            no_bottom_time = pd.concat([no_bottom_time, data[['bottom_time']]], axis=1)
            no_stop_time = pd.concat([no_stop_time, data[['stop_time']]], axis=1)

        v_data = pd.concat([v_data, no_v_data], axis=0)
        d_data = pd.concat([d_data, no_d_data], axis=0)
        top_data = pd.concat([top_data, no_top_data], axis=0)
        bottom_time = pd.concat([bottom_time, no_bottom_time], axis=0)
        stop_time = pd.concat([stop_time, no_stop_time], axis=0)

        # print(no_angle_data['1_1'])
        # print(angle_data['1_1'])

    moving_time = 1 - stop_time
    # print(moving_time)
    # print(moving_time.to_dict())
    v_data.columns = name_list
    d_data.columns = name_list
    top_data.columns = name_list
    bottom_time.columns = name_list
    stop_time.columns = name_list
    moving_time.columns = name_list

    print(v_data)
    # print(1- moving_time)
    # exit(333)

    page = Page(layout=Page.SimplePageLayout)
    page.add(
        getLine(v_data, 'velocity'),
        getLine(d_data, 'distance'),
        getLine(top_data, 'top'),
        getLine(bottom_time, 'bottom'),
        getLine(moving_time, 'move'),
        getLine(1 - moving_time, 'static'),
        # getHeatMap(angle_data, time_list, 'Angle Distribution'),
        # getHeatMap(acc_data, time_list, 'Acceleration Distribution')
    )
    page.render("page_simple_layout.html")
    # total_data["velocity"] = format_data(data=v_data, time_list=time_list, name_list=name_list)
    # total_data["distance"] = format_data(data=d_data, time_list=time_list, name_list=name_list)
    # total_data["top_time"] = format_data(data=top_data, time_list=time_list, name_list=name_list)
    # total_data["bottom_time"] = format_data(data=bottom_time, time_list=time_list, name_list=name_list)
    # total_data["stopping_time"] = format_data(data=stop_time, time_list=time_list, name_list=name_list)
    # total_data["moving_time"] = format_data(data=moving_time, time_list=time_list, name_list=name_list)

    # 生成时间轴的图
    # timeline = Timeline(init_opts=opts.InitOpts(width="1600px", height="800px"))
    #
    # for y in time_list:
    #     timeline.add(get_year_overlap_chart(total_data, time_mim=y), time_point=str(y))
    #
    # # 1.0.0 版本的 add_schema 暂时没有补上 return self 所以只能这么写着
    # timeline.add_schema(is_auto_play=False, play_interval=1000)
    # timeline.render("finance_indices_2002.html")
