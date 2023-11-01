import sys

import pandas as pd

sys.path.append("../../")
sys.path.append("../")
sys.path.append(".")
import os, sys
import argparse
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import pyecharts.options as opts
from pyecharts.charts import Boxplot
from pyecharts.commons.utils import JsCode

sys.path.append(".")
sys.path.append('../../')
from common.utility import *

plt.rcParams['font.sans-serif'] = "Microsoft YaHei"
plt.rcParams['font.size'] = 15

# color = [
#     '#2e75b6',
#     '#ffcc66', '#ff9966', '#ff6600'
# ]
color = [
    '#2e75b6',
    '#548235', '#facb6c', '#cf6107'
]

class BasicStatistic(object):
    def __init__(self,
                 index_data, dist_map_list=('velocity', 'dist', 'dist_cal'),
                 info_col=('exposure_time', 'DayTank', 'camNO', 'exp_time', 'filename', 'region_name')
                 ):
        self.index_data = index_data
        self.info_col = info_col
        self.column_list = index_data.columns.tolist()
        self.setting_info = {}

        # map_list = list(set(self.column_list).intersection(set(dist_map_list)))
        # if len(map_list) > 0:
        #     self.img2Word(map_list)
        print(self.index_data)

    # def img2Word(self, map_list):
    #     # 对于每一行，通过列名name访问对应的元素
    #     self.index_data.reset_index(inplace=True)
    #     for irow in self.index_data.index.tolist():
    #         DayTank = self.index_data.iloc[irow]['DayTank']
    #         region_no = self.index_data.iloc[irow]['region_name'].split("_")[0]
    #         camId = camera_id_map[self.index_data.iloc[irow]['camNO']]
    #         if f"{DayTank}_{region_no}_{str(camId)}" in self.setting_info:
    #             ratio = self.setting_info[f"{DayTank}_{region_no}_{str(camId)}"]
    #         else:
    #             cfg_path = os.path.join(root_path, DayTank)
    #             config = readConfig(cfg_path)
    #             if camId == 1:
    #                 ratio = config['Aquarium'].getfloat(f"top_ratio_{region_no}")
    #             else:
    #                 ratio = config['Aquarium'].getfloat(f"side_ratio_{region_no}")
    #             self.setting_info[f"{DayTank}_{region_no}_{str(camId)}"] = ratio
    #         scale_data = self.index_data.iloc[irow][map_list] / ratio
    #         self.index_data.loc[irow, map_list] = scale_data.copy()

    def groupbyExpData(self, stat_index=None):

        view_way = {}
        for k in self.index_data.columns.tolist():
            if k in stat_index:
                view_way[k] = stat_index[k]
        table = self.index_data.groupby(['exposure_time', 'region_name', 'DayTank']).agg(view_way)

        return table

    def formatBoxplot(self, col_data, col_name):
        col_data.reset_index(inplace=True)
        group_flag = list(set(col_data['exposure_time'].values))
        new_frame = []
        for idx, iflag in enumerate(group_flag):
            iflag_data = col_data[col_data['exposure_time'] == iflag][col_name]
            new_frame.append(iflag_data.values)

        return new_frame, group_flag

    def formatStackedBar(self, col_data, col_name):
        col_data.reset_index(inplace=True)

        group_flag = list(set(col_data['exposure_time'].values))
        mean_dict = {}
        std_dict = {}
        for idx, iflag in enumerate(group_flag):
            iflag_data = col_data[col_data['exposure_time'] == iflag][col_name]
            for angle in col_name:
                if angle not in mean_dict:
                    mean_dict[angle] = []
                if angle not in std_dict:
                    std_dict[angle] = []
                mean_dict[angle].append(iflag_data[[angle]].mean().values[0])
                std_dict[angle].append(iflag_data[[angle]].std().values[0])

        return mean_dict, std_dict, group_flag


def drawTimeBoxes(box_data, labels, axis_name, title):
    # Random test data

    fig, ax1 = plt.subplots(figsize=(9, 7), constrained_layout=True)

    # rectangular box plot
    ax1.boxplot(box_data,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=labels)  # will be used to label x-ticks
    ax1.set_title(drag_name + ":" + title)
    ax1.yaxis.grid(True)
    ax1.set_xlabel('exposure time')
    ax1.set_ylabel(axis_name)
    plt.savefig(os.path.join(plt_save_path, title + ".png"), dpi=300)  # 保存图片
    # plt.show()


def drawStackedBar(labels, mean_dict, std_dict, column_names, title):
    labels = [_ + 1 for _ in labels]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    for idx, iindex in enumerate(column_names):
        imeans = []
        istd = []
        for im, iistd in zip(mean_dict[iindex], std_dict[iindex]):
            imeans.append(im * 100)
            if im - iistd >= 0:
                istd.append(iistd * 100)
            else:
                istd.append(im * 100)

        label_name = f"{iindex.split('_')[-2:][0]}°-{iindex.split('_')[-2:][1]}°".replace("91", "90")

        if idx == 0:
            ax.bar(labels, imeans, yerr=istd, label=label_name)
        else:
            last_mean_value = mean_dict[column_names[idx - 1]]
            ax.bar(
                labels, imeans, yerr=istd,
                label=label_name, bottom=last_mean_value
            )

    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Time (every 1 hour)')

    ax.set_yticks([_ * 10 for _ in range(11)])
    ax.set_yticklabels([_ * 10 for _ in range(11)])
    ax.set_ylabel(f'{title} (%)')

    ax.set_title(drag_name)
    ax.legend()
    fig.subplots_adjust(bottom=0.2)
    print(os.path.join(plt_save_path, title + ".png"))
    plt.tight_layout()
    plt.savefig(os.path.join(plt_save_path, title + ".png"))  # 保存图片
    # plt.show()


def drawLines(labels, mean_data, std_data, column_names, title, annotations=None):
    # example data
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ls = 'dotted'
    markers = ['o', '^', 'v', 'D']
    index_names = []
    x = np.arange(len(labels)) + 1
    for idx, iindex in enumerate(column_names):
        means_data = [_ * 100 for _ in mean_data[iindex]]
        stds_data = [_ * 100 for _ in std_data[iindex]]
        xerr = 0.1
        # standard error bars
        ax.errorbar(
            x, means_data, xerr=xerr, yerr=stds_data,
            linestyle=ls, marker=markers[idx],
            color=color[idx]
        )

        ax.set_yticks(annotations['y_ticks'])
        index_names.append(f"{iindex.split('_')[-2:][0]}°-{iindex.split('_')[-2:][1]}°".replace("91", "90"))

        if annotations is not None:
            if drag_name in annotations:
                if iindex in annotations[drag_name]:
                    annos = annotations[drag_name][iindex]
                    for ianno in annos:
                        plt.annotate(
                            r'$%s$' % ianno['sig'], xy=(ianno['x0'], ianno['y0']),
                            xycoords='data', xytext=(0, 0),
                            textcoords='offset points',
                            fontsize=15,
                            color=color[idx]
                        )

    ax.legend(loc="best", ncol=1, labels=[_.replace("ppb", " μg/L").replace("ppm", "000 μg/L") for _ in index_names])
    ax.set_xlabel('Time (every 1 hour)')
    ax.set_xticks([_ + 1 for _ in range(12)])
    ax.set_xticklabels([_ + 1 for _ in range(12)])
    ax.set_title(drag_name.replace("ppb", " μg/L").replace("ppm", "000 μg/L"))

    ax.set_ylabel('Ratio of up down angle (%)')

    ax.set_yticks(annotations['y_ticks'])
    ax.set_yticklabels(annotations['yticklabels'])
    plt.tight_layout()
    # filename = os.path.join(plt_save_path, f"{drag_name}_{title}_{'_'.join(column_names)}.png")
    filename = f"E:\\data\\3D_pre\\drag_result\\figure\\{drag_name}_{title}_{'_'.join(column_names)}.png"
    print(f"save data in {filename}")
    # plt.show()
    plt.savefig(filename, dpi=600)


def drawPieBar(angle_0_30, angle_30_90, angle_30_40, angle_40_50, angle_50_60, angle_60_90, expT):
    # make figure and assign axis objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    fig.subplots_adjust(wspace=0)

    # pie chart parameters
    # ratio = [0.9596, 0.02197]
    ratios = [angle_30_90, angle_0_30]
    labels = ['30°- 90°', '0°- 30°']
    explode = [0, 0.1]
    # rotate so that first wedge is split by the x-axis
    angle = -180 * ratios[0]
    ax1.pie(
        ratios, autopct='%1.3f%%',
        startangle=angle,
        labels=labels,
        labeldistance=1.1,
        explode=explode
    )

    # bar chart parameters

    xpos = 0
    bottom = 0
    # ratios = [30_40, 40_50, 50_60, 60_90]
    ratios = [angle_30_40, angle_40_50, angle_50_60, angle_60_90]
    width = .2
    colors = [[.1, .3, .3], [.1, .3, .7], [.1, .3, .5], [.1, .3, .9]]

    for j in range(len(ratios)):
        height = ratios[j]
        ax2.bar(xpos, height, width, bottom=bottom, color=colors[j])
        ypos = bottom + ax2.patches[j].get_height() / 2
        bottom += height
        ax2.text(xpos, ypos, "%1.3f%%" % (ax2.patches[j].get_height() * 100),
                 ha='center')

    ax2.set_title('Angle change ratio (%)')
    ax2.legend(('30°- 40°', '40°- 50°', '50°- 60°', 'Over 60°'))
    ax2.axis('off')
    ax2.set_xlim(- 2.5 * width, 2.5 * width)

    # use ConnectionPatch to draw lines between the two plots
    # get the wedge data
    theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2
    center, r = ax1.patches[0].center, ax1.patches[0].r
    bar_height = sum([item.get_height() for item in ax2.patches])

    # draw top connecting line
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    con.set_linewidth(4)
    ax2.add_artist(con)

    # draw bottom connecting line
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    ax2.add_artist(con)
    con.set_linewidth(4)

    # plt.show()
    plt.savefig(os.path.join(plt_save_path, str(expT) + "_body_angle.png"), dpi=600)


def drawBar(x_label, error, y_pos, y_label_name, x_label_name='Time (every 1 hour)', title=""):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.bar(x_label, y_pos, yerr=error, align='center')

    ax.set_xticks(x_label)
    ax.set_xticklabels(x_label, rotation=45)
    ax.set_yticks([_ * 10 for _ in range(11)])
    ax.set_yticklabels([_ * 10 for _ in range(11)])
    ax.set_xlabel(x_label_name)
    ax.set_ylabel(f'{y_label_name} (%)')
    ax.set_title(title)
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.show()


def drawTransmiteeBar(
        x_label, error, y_pos,
        y_label_name, title="", unit="",
        annos=None
):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
    ax.bar(
        [_ for _ in range(len(x_label))],
        y_pos,
        yerr=error,
        align='center',
        color=[
            '#9dc3e6',
            '#ffd966', '#ffd966', '#ffd966',
            '#f4b183', '#f4b183', '#f4b183',
            '#f08080', '#f08080', '#f08080',
        ]
    )

    if annos is not None:
        for anno in annos:
            x0 = (anno['xstart'] + anno['xend']) / 2.0 + 0.5
            y0 = anno['yend']
            plt.annotate(r'$%s$' % anno['sig'], xy=(x0, y0), xycoords='data', xytext=(-15, +1),
                         textcoords='offset points', fontsize=17, color="red")

            x = np.arange(anno['xstart'], anno['xend'] + 0.1, 0.1)
            y = np.ones_like((x)) * anno['yend']
            plt.plot(x, y, label="$x$", color="black", linewidth=1)

            y = np.arange(anno['ystart'], anno['yend'], anno['yend'] - anno['ystart'] - 0.00001)
            x = np.ones_like((y)) * anno['xstart']
            plt.plot(x, y, label="$y$", color="black", linewidth=1)

            y = np.arange(anno['ystart'], anno['yend'], anno['yend'] - anno['ystart'] - 0.00001)
            x = np.ones_like((y)) * anno['xend']
            plt.plot(x, y, label="$y$", color="black", linewidth=1)

    ax.set_xticks(range(len(x_label)))
    x_label = [_.replace("ppb", " μg/L").replace("ppm", "000 μg/L") for _ in list(data['drag_name'])]
    ax.set_xticklabels(x_label, rotation=45, ha='right')
    if title == 'top_time':
        ax.set_ylabel('Ratio of water surface retention (%)')
        ax.set_yticks([_ * 10 for _ in range(11)])
        ax.set_yticklabels([_ * 10 for _ in range(11)])
    elif title == '0_15':
        ax.set_ylabel(f'Ratio of up down angle {title.split("_")[0]}°-{title.split("_")[1].replace("91", "90")}°(%)')
        ax.set_yticks([_ * 10 for _ in range(11)])
        ax.set_yticklabels([_ * 10 for _ in range(11)])
    elif title == '15_45':
        ax.set_ylabel(f'Ratio of up down angle {title.split("_")[0]}°-{title.split("_")[1].replace("91", "90")}°(%)')
        ax.set_yticks([_ * 10 for _ in range(3)])
        ax.set_yticklabels([_ * 10 for _ in range(3)])
    elif title == '45_91':
        ax.set_ylabel(f'Ratio of up down angle {title.split("_")[0]}°-{title.split("_")[1].replace("91", "90")}°(%)')
        ax.set_yticks([_ * 5 for _ in range(3)])
        ax.set_yticklabels([_ * 5 for _ in range(3)])
    else:
        ax.set_ylabel(f'{y_label_name} ({unit})')

    # ax.legend(loc="best",ncol=2,fontsize="x-small")
    # ax.set_title(title)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    # plt.show()
    print(f"data saved in {plt_save_path} with name: stat_" + str(title) + ".png")
    plt.savefig(os.path.join(plt_save_path, "stat_" + str(title) + ".png"), dpi=600)


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-rp", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("-drag", "--drag_name",
                    # default="CK",
                    # default="4Hydroxy50ppb",
                    # default="4Hydroxy500ppb",
                    # default="4Hydroxy1ppm",
                    # default="6PPD50ppb",
                    # default="6PPD500ppb",
                    default="6PPD1ppm",
                    # default="6PPDQ50ppb",
                    # default="6PPDQ500ppb",
                    # default="6PPDQ1ppm",
                    type=str)
    ap.add_argument("--exposureT", default="0,1,2,3,4,5,6,7,8,9,10,11", type=str)
    # ap.add_argument("--exposureT", default="0", type=str)

    args = vars(ap.parse_args())

    root_path = args['root_path']
    drag_name = args['drag_name']
    exp_time = args['exposureT'].split(",")

    turning_angle_interval = (0, 10, 20, 30, 40, 50, 60, 70, 80, 91)
    up_down_angle_interval = (0, 15, 45, 91)

    # 单指标多时间图存储路径
    plt_save_path = os.path.join(root_path, 'drag_result', 'figure', drag_name)
    if not os.path.exists(plt_save_path):
        os.makedirs(plt_save_path)

    # 单指标多时间图存储路径
    integration_save_path = os.path.join(root_path, 'drag_result', 'multiple', drag_name)
    if not os.path.exists(integration_save_path):
        os.makedirs(integration_save_path)

    # 原始数据读入路径
    index_path = os.path.join(root_path, 'drag_result', 'single', drag_name)

    info_col = ['exposure_time', 'DayTank', 'camNO', 'exp_time', 'filename', 'region_name']

    side_col = info_col + ['top_time', 'up_down_angle']
    side_angle_col = []
    if 'up_down_angle' in side_col:
        side_col.remove('up_down_angle')
        for i in range(len(up_down_angle_interval) - 1):
            low_bound = up_down_angle_interval[i]
            up_bound = up_down_angle_interval[i + 1]
            side_angle_col.append(f"up_down_angle_{low_bound}_{up_bound}")
    side_col = side_col + side_angle_col

    top_col = info_col + ['dist', 'velocity', 'dist_cal', 'turning_angle']
    top_angle_col = []
    if 'turning_angle' in top_col:
        top_col.remove('turning_angle')
        for i in range(len(turning_angle_interval) - 1):
            low_bound = turning_angle_interval[i]
            up_bound = turning_angle_interval[i + 1]
            top_angle_col.append(f"turning_angle_{low_bound}_{up_bound}")
    top_col = top_col + top_angle_col

    all_col = side_col + top_col
    statistic_way = {}

    for icol in all_col:
        if 'turning_angle' in icol:
            statistic_way[icol] = 'mean'
        elif 'dist' in icol:
            statistic_way[icol] = 'sum'
        elif 'velocity' == icol:
            statistic_way[icol] = 'mean'
        elif 'top_time' == icol:
            statistic_way[icol] = 'mean'
        elif 'up_down_angle' in icol:
            statistic_way[icol] = 'mean'
        else:
            continue

    top_result_data = pd.DataFrame()
    side_result_data = pd.DataFrame()

    for ifile in exp_time:
        tindex_data = pd.read_csv(os.path.join(index_path, ifile + "_top.csv"))
        sindex_data = pd.read_csv(os.path.join(index_path, ifile + "_side.csv"))

        top_index_data = tindex_data.copy()[top_col]
        top_index_data.dropna(axis=0, how='any', inplace=True)
        side_index_data = sindex_data.copy()[side_col]
        side_index_data.dropna(axis=0, how='any', inplace=True)

        top_result_data = pd.concat([top_result_data, top_index_data])
        side_result_data = pd.concat([side_result_data, side_index_data])

    Tanalyzer = BasicStatistic(top_result_data)
    Sanalyzer = BasicStatistic(side_result_data)
    Tmean_data = Tanalyzer.groupbyExpData(stat_index=statistic_way)
    Smean_data = Sanalyzer.groupbyExpData(stat_index=statistic_way)

    Tmean_data.to_csv(os.path.join(integration_save_path, 'top_view.csv'), sep=',')
    Smean_data.to_csv(os.path.join(integration_save_path, 'side_view.csv'), sep=',')

    # # 绘制速度箱线图
    # box_plot_data, label = Tanalyzer.formatBoxplot(Tmean_data[['velocity']], 'velocity')
    # drawTimeBoxes(box_plot_data, label, 'velocity', 'velocity')

    # # 绘制运动路径箱线图
    # box_plot_data, label = Tanalyzer.formatBoxplot(Tmean_data[['dist']], 'dist')
    # drawTimeBoxes(box_plot_data, label, 'dist', 'dist')

    # # 绘制运动路径箱线图
    # box_plot_data, label = Tanalyzer.formatBoxplot(Tmean_data[['dist_cal']], 'dist_cal')
    # drawTimeBoxes(box_plot_data, label, 'dist_cal', 'dist_cal')

    # ====================================================================================== #
    # # 上下浮动的角度 stack bar图
    # column_names = side_angle_col
    # mean_data, std_data, label = Sanalyzer.formatStackedBar(
    #     Smean_data[column_names], column_names
    # )
    #
    # # drawStackedBar(label, mean_data, std_data, column_names, 'Ratio of up down angle')
    # drawLines(label, mean_data, std_data, column_names, 'Ratio of up down angle', annotations={
    #     'y_ticks': [_ * 10 for _ in range(11)],
    #     'yticklabels': [_ * 10 for _ in range(11)],
    #     '4Hydroxy50ppb': {
    #         'up_down_angle_0_15': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 60,
    #             } for _ in [6, 9, 10]
    #         ],
    #         'up_down_angle_45_91': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 66,
    #             } for _ in [4, 5, 6, 7, 8, 9, 10, 11, 12]
    #         ]
    #     },
    #     '4Hydroxy500ppb': {
    #         'up_down_angle_45_91': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 66,
    #             } for _ in [5,6,7,8,9,10,11,12]
    #         ]
    #     },
    #     '4Hydroxy1ppm': {
    #         'up_down_angle_0_15': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 60,
    #             } for _ in [8,12]
    #         ],
    #         'up_down_angle_45_91': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 66,
    #             } for _ in [4,5,6,7,8,9,10,11,12]
    #         ]
    #     },
    #     '6PPD50ppb': {
    #         'up_down_angle_45_91': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 60,
    #             } for _ in [11]
    #         ]
    #     },
    #     '6PPD500ppb': {
    #         'up_down_angle_15_45': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 63,
    #             } for _ in [8,9,10,11]
    #         ],
    #         'up_down_angle_45_91': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 66,
    #             } for _ in [8,10]
    #         ]
    #     },
    #     '6PPD1ppm': {
    #         'up_down_angle_0_15': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 60,
    #             } for _ in [5,7,9,10,11]
    #         ],
    #         'up_down_angle_15_45': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 63,
    #             } for _ in [2,3,4,5,6,7,8,9,10,11,12]
    #         ],
    #         'up_down_angle_45_91': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 66,
    #             } for _ in [3,4,5,6,7,8,9,10,11,12]
    #         ]
    #     },
    #     '6PPDQ500ppb': {
    #         'up_down_angle_45_91': [
    #             {
    #                 "sig": "*",
    #                 "x0": _,
    #                 "y0": 63,
    #             } for _ in [5,10,12]
    #         ]
    #     },
    #
    # })
    # exit(3)
    # 绘制转向角的角度 stack bar图

    # new_col_names = ['turning_angle_0_30', 'turning_angle_30_90', 'turning_angle_60_90']
    # Tmean_data['turning_angle_0_30'] = Tmean_data['turning_angle_0_10'] + Tmean_data['turning_angle_10_20'] + \
    #                                    Tmean_data['turning_angle_20_30']
    # Tmean_data['turning_angle_30_90'] = 1 - Tmean_data['turning_angle_0_30']
    #
    # Tmean_data['turning_angle_60_90'] = Tmean_data['turning_angle_30_90'] - \
    #                                     Tmean_data['turning_angle_30_40'] - \
    #                                     Tmean_data['turning_angle_40_50'] - \
    #                                     Tmean_data['turning_angle_50_60']
    #
    # column_names = top_angle_col + new_col_names
    # for i in exp_time:
    #     mean_dict, _, group_flag = Tanalyzer.formatStackedBar(
    #         Tmean_data[column_names], column_names
    #     )
    #     angle_0_30 = mean_dict['turning_angle_0_30'][int(i)]
    #     angle_30_90 = mean_dict['turning_angle_30_90'][int(i)]
    #
    #     angle_30_40 = mean_dict['turning_angle_30_40'][int(i)]
    #     angle_40_50 = mean_dict['turning_angle_40_50'][int(i)]
    #     angle_50_60 = mean_dict['turning_angle_50_60'][int(i)]
    #     angle_60_90 = mean_dict['turning_angle_60_90'][int(i)]
    #     drawPieBar(angle_0_30, angle_30_90, angle_30_40, angle_40_50, angle_50_60, angle_60_90, i)

    # 绘制柱状图
    # np_data = {}
    # column_names = ['top_time']
    # mean_data, std_data, label = Sanalyzer.formatStackedBar(
    #     Smean_data[column_names], column_names
    # )
    # y_pos = [_*100 for _ in mean_data[column_names[0]]]
    # error = [_*100 for _ in std_data[column_names[0]]]
    # x_label = [_+1 for _ in label]
    # np_data['top_time_per'] = np.array(y_pos)
    # drawBar(x_label, error, y_pos, column_names, 'percent', title='Ratio of water surface retention')

    # ===================================DA=======================================#
    # data = pd.read_csv(
    #     os.path.join(root_path, 'drag_result', "dopamine.csv"),
    # )
    # marked_list = []
    # x_label = [_.replace("ppb", " μg/L").replace("ppm", "000 μg/L") for _ in list(data['drag_name'])]
    # error = list(data["SD"])
    # y_pos = list(data["DA平均值"])
    # drawTransmiteeBar(
    #     x_label, error, y_pos,
    #     y_label_name="Transmitte",
    #     title="DA",
    #     unit="ng/ml"
    # )

    # =====================================aminobutyric_acid================================ #
    # data = pd.read_csv(
    #     os.path.join(root_path, 'drag_result', "aminobutyric_acid.csv"),
    # )
    #
    # x_label = [_.replace("ppb", " μg/L").replace("ppm", "000 μg/L") for _ in list(data['drag_name'])]
    # error = list(data["SD"])
    # y_pos = list(data["平均值"])
    # drawTransmiteeBar(
    #     x_label, error, y_pos,
    #     y_label_name="Transmitte",
    #     title="GABA",
    #     unit="$\mu$mol/g wet weight",
    #     annos=[
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPD1ppm'.replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 19,
    #             "yend": 19.5,
    #         }, {
    #             "xstart": x_label.index('6PPD500ppb'.replace("ppb", " μg/L")),
    #             "xend": x_label.index('6PPD1ppm'.replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 17.5,
    #             "yend": 18,
    #         },
    #     ]
    # )


    # =====================================acetylcholine================================ #

    # data = pd.read_csv(
    #     os.path.join(root_path, 'drag_result', "acetylcholine.csv"),
    # )
    #
    # x_label = [_.replace("ppb", " μg/L").replace("ppm", "000 μg/L") for _ in list(data['drag_name'])]
    # error = list(data["SD1"])
    # y_pos = list(data["Ach平均值"])
    # drawTransmiteeBar(
    #     x_label, error, y_pos,
    #     y_label_name="Transmitte",
    #     title="ACh",
    #     unit="$\mu$g/mogprot",
    #     annos=[
    #         {
    #             "xstart": x_label.index('4Hydroxy50ppb'.replace("ppb", " μg/L")),
    #             "xend": x_label.index('4Hydroxy500ppb'.replace("ppb", " μg/L")),
    #             "sig": "*",
    #             "ystart": 128,
    #             "yend": 130,
    #         },
    #         {
    #             "xstart": x_label.index('4Hydroxy50ppb'.replace("ppb", " μg/L")),
    #             "xend": x_label.index('4Hydroxy1ppm'.replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 134,
    #             "yend": 136,
    #         },
    #
    #         {
    #             "xstart": x_label.index('6PPD50ppb'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "xend": x_label.index('6PPD1ppm'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 111,
    #             "yend": 113,
    #         },
    #         {
    #             "xstart": x_label.index('6PPD500ppb'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "xend": x_label.index('6PPD1ppm'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 105,
    #             "yend": 107,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPD1ppm'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 117,
    #             "yend": 119,
    #         },
    #
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPDQ50ppb'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 141,
    #             "yend": 143,
    #         },
    #         {
    #             "xstart": x_label.index('6PPDQ50ppb'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "xend": x_label.index('6PPDQ500ppb'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 128,
    #             "yend": 130,
    #         },
    #         {
    #             "xstart": x_label.index('6PPDQ50ppb'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "xend": x_label.index('6PPDQ1ppm'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 134,
    #             "yend": 136,
    #         },
    #         {
    #             "xstart": x_label.index('6PPDQ500ppb'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "xend": x_label.index('6PPDQ1ppm'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 117,
    #             "yend": 119,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPDQ500ppb'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 147,
    #             "yend": 149,
    #         }, {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPDQ1ppm'.replace("ppb", " μg/L").replace("ppm", "000 μg/L")),
    #             "sig": "*",
    #             "ystart": 153,
    #             "yend": 155,
    #         }
    #     ]
    # )

#
    # # # velocity
    # index_name = 'velocity'
    # data = pd.read_csv(
    #     os.path.join(root_path, 'drag_result', "behavior_index.csv"),
    # )
    # if index_name == 'velocity':
    #     x_label = list(data['drag_name'])
    #     error = list(data[f"std_{index_name}"])
    #     y_pos = list(data[f"mean_{index_name}"])
    #     y_label_name = 'velocity'
    #     unit = "mm/s"
    #     annos = [
    #         {
    #             "xstart": x_label.index('6PPD50ppb'),
    #             "xend": x_label.index('6PPD1ppm'),
    #             "sig": "*",
    #             "ystart": 0.83,
    #             "yend": 0.85,
    #         },
    #         {
    #             "xstart": x_label.index('6PPD500ppb'),
    #             "xend": x_label.index('6PPD1ppm'),
    #             "sig": "*",
    #             "ystart": 0.75,
    #             "yend": 0.78,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPD1ppm'),
    #             "sig": "*",
    #             "ystart": 0.9,
    #             "yend": 0.92,
    #         }
    #     ]
    # elif index_name == 'dist':
    #     x_label = list(data['drag_name'])
    #     error = list(data[f"std_{index_name}"])
    #     y_pos = list(data[f"mean_{index_name}"])
    #     y_label_name = 'distance'
    #     unit = "mm"
    #     annos = [
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPD1ppm'),
    #             "sig": "*",
    #             "ystart": 18000,
    #             "yend": 18500,
    #         }
    #     ]
    # elif index_name == 'top_time':
    #     x_label = list(data['drag_name'])
    #     error = [_ * 100 for _ in list(data[f"std_{index_name}"])]
    #     y_pos = [_ * 100 for _ in list(data[f"mean_{index_name}"])]
    #     y_label_name = 'Ratio of water surface retention'
    #     unit = "%"
    #     annos = [
    #         {
    #             "xstart": x_label.index('4Hydroxy50ppb'),
    #             "xend": x_label.index('4Hydroxy500ppb'),
    #             "sig": "*",
    #             "ystart": 52,
    #             "yend": 54,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('4Hydroxy500ppb'),
    #             "sig": "*",
    #             "ystart": 58,
    #             "yend": 60,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('4Hydroxy1ppm'),
    #             "sig": "*",
    #             "ystart": 64,
    #             "yend": 66,
    #         }, {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPD50ppb'),
    #             "sig": "*",
    #             "ystart": 70,
    #             "yend": 72,
    #         }, {
    #             "xstart": x_label.index('6PPD50ppb'),
    #             "xend": x_label.index('6PPD500ppb'),
    #             "sig": "*",
    #             "ystart": 58,
    #             "yend": 60,
    #         }, {
    #             "xstart": x_label.index('6PPD50ppb'),
    #             "xend": x_label.index('6PPD1ppm'),
    #             "sig": "*",
    #             "ystart": 64,
    #             "yend": 66
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPD500ppb'),
    #             "sig": "*",
    #             "ystart": 78,
    #             "yend": 80
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPD1ppm'),
    #             "sig": "*",
    #             "ystart": 86,
    #             "yend": 88
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPDQ50ppb'),
    #             "sig": "*",
    #             "ystart": 93,
    #             "yend": 95,
    #
    #         },
    #         {
    #             "xstart": x_label.index('6PPDQ50ppb'),
    #             "xend": x_label.index('6PPDQ500ppb'),
    #             "sig": "*",
    #             "ystart": 64,
    #             "yend": 66,
    #         },
    #         {
    #             "xstart": x_label.index('6PPDQ50ppb'),
    #             "xend": x_label.index('6PPDQ1ppm'),
    #             "sig": "*",
    #             "ystart": 70,
    #             "yend": 72,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPDQ500ppb'),
    #             "sig": "*",
    #             "ystart": 99,
    #             "yend": 101,
    #         },
    #         {
    #             "xstart": x_label.index('6PPDQ500ppb'),
    #             "xend": x_label.index('6PPDQ1ppm'),
    #             "sig": "*",
    #             "ystart": 76,
    #             "yend": 78,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPDQ1ppm'),
    #             "sig": "*",
    #
    #             "ystart": 104,
    #             "yend": 106,
    #         },
    #
    #     ]
    # elif index_name == 'meandering':
    #     x_label = list(data['drag_name'])
    #     error = list(data[f"std_{index_name}"])
    #     y_pos = list(data[f"mean_{index_name}"])
    #     y_label_name = 'meandering'
    #     unit = "%"
    #     annos = None
    # elif index_name == '0_15':
    #     x_label = list(data['drag_name'])
    #     error = [_ * 100 for _ in list(data[f"std_{index_name}"])]
    #     y_pos = [_ * 100 for _ in list(data[f"mean_{index_name}"])]
    #     y_label_name = 'Ratio of up down angle'
    #     unit = "%"
    #     annos = [
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('4Hydroxy50ppb'),
    #             "sig": "*",
    #             "ystart": 48,
    #             "yend": 50,
    #         },
    #         {
    #             "xstart": x_label.index('4Hydroxy50ppb'),
    #             "xend": x_label.index('4Hydroxy500ppb'),
    #             "sig": "*",
    #             "ystart": 41,
    #             "yend": 43,
    #         },
    #         {
    #             "xstart": x_label.index('4Hydroxy50ppb'),
    #             "xend": x_label.index('4Hydroxy1ppm'),
    #             "sig": "*",
    #             "ystart": 55,
    #             "yend": 57,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('4Hydroxy500ppb'),
    #             "sig": "*",
    #             "ystart": 60,
    #             "yend": 62,
    #         },
    #         {
    #             "xstart": x_label.index('4Hydroxy500ppb'),
    #             "xend": x_label.index('4Hydroxy1ppm'),
    #             "sig": "*",
    #             "ystart": 48,
    #             "yend": 50,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPD50ppb'),
    #             "sig": "*",
    #             "ystart": 67,
    #             "yend": 69,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPDQ50ppb'),
    #             "sig": "*",
    #             "ystart": 73,
    #             "yend": 75,
    #         }
    #         , {
    #             "xstart": x_label.index('6PPDQ50ppb'),
    #             "xend": x_label.index('6PPDQ500ppb'),
    #             "sig": "*",
    #             "ystart": 48,
    #             "yend": 50,
    #         }, {
    #             "xstart": x_label.index('6PPDQ50ppb'),
    #             "xend": x_label.index('6PPDQ1ppm'),
    #             "sig": "*",
    #             "ystart": 60,
    #             "yend": 62,
    #         }, {
    #             "xstart": x_label.index('6PPDQ500ppb'),
    #             "xend": x_label.index('6PPDQ1ppm'),
    #             "sig": "*",
    #             "ystart": 55,
    #             "yend": 57,
    #         }, {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPDQ1ppm'),
    #             "sig": "*",
    #             "ystart": 79,
    #             "yend": 81,
    #         }
    #     ]
    # elif index_name == '15_45':
    #     x_label = list(data['drag_name'])
    #     error = [_ * 100 for _ in list(data[f"std_{index_name}"])]
    #     y_pos = [_ * 100 for _ in list(data[f"mean_{index_name}"])]
    #     y_label_name = 'Ratio of up down angle'
    #     unit = "%"
    #     annos = [
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('4Hydroxy50ppb'),
    #             "sig": "*",
    #             "ystart": 11,
    #             "yend": 11.5,
    #         },
    #         {
    #             "xstart": x_label.index('4Hydroxy50ppb'),
    #             "xend": x_label.index('4Hydroxy500ppb'),
    #             "sig": "*",
    #             "ystart": 12.5,
    #             "yend": 13,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('4Hydroxy500ppb'),
    #             "sig": "*",
    #             "ystart": 14,
    #             "yend": 14.5,
    #         },
    #         {
    #             "xstart": x_label.index('4Hydroxy500ppb'),
    #             "xend": x_label.index('4Hydroxy1ppm'),
    #             "sig": "*",
    #             "ystart": 11,
    #             "yend": 11.5,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('4Hydroxy1ppm'),
    #             "sig": "*",
    #             "ystart": 15.5,
    #             "yend": 16,
    #         },
    #         {
    #             "xstart": x_label.index('6PPD50ppb'),
    #             "xend": x_label.index('6PPD500ppb'),
    #             "sig": "*",
    #             "ystart": 11,
    #             "yend": 11.5,
    #         },
    #         {
    #             "xstart": x_label.index('6PPD50ppb'),
    #             "xend": x_label.index('6PPD1ppm'),
    #             "sig": "*",
    #             "ystart": 14,
    #             "yend": 14.5,
    #         },
    #
    #         {
    #             "xstart": x_label.index('6PPD500ppb'),
    #             "xend": x_label.index('6PPD1ppm'),
    #             "sig": "*",
    #             "ystart": 12.5,
    #             "yend": 13,
    #         },
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPD500ppb'),
    #             "sig": "*",
    #             "ystart": 17.5,
    #             "yend": 18,
    #         }
    #         ,
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPD1ppm'),
    #             "sig": "*",
    #             "ystart": 19,
    #             "yend": 19.5,
    #         }
    #         ,
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPDQ50ppb'),
    #             "sig": "*",
    #             "ystart": 21,
    #             "yend": 21.5,
    #         },
    #         {
    #             "xstart": x_label.index('6PPDQ50ppb'),
    #             "xend": x_label.index('6PPDQ1ppm'),
    #             "sig": "*",
    #             "ystart": 14,
    #             "yend": 14.5,
    #         },
    #         {
    #             "xstart": x_label.index('6PPDQ500ppb'),
    #             "xend": x_label.index('6PPDQ1ppm'),
    #             "sig": "*",
    #             "ystart": 12.5,
    #             "yend": 13,
    #         },
    #          {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPDQ1ppm'),
    #             "sig": "*",
    #             "ystart": 22.5,
    #             "yend": 23,
    #         },
    #     ]
    # elif index_name == '45_91':
    #     x_label = list(data['drag_name'])
    #     error = [_ * 100 for _ in list(data[f"std_{index_name}"])]
    #     y_pos = [_ * 100 for _ in list(data[f"mean_{index_name}"])]
    #     y_label_name = 'Ratio of up down angle'
    #     unit = "%"
    #     annos = [
    #         {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPD50ppb'),
    #             "sig": "*",
    #             "ystart": 7,
    #             "yend": 7.25,
    #         }, {
    #             "xstart": x_label.index('6PPD50ppb'),
    #             "xend": x_label.index('6PPD500ppb'),
    #             "sig": "*",
    #             "ystart": 5.5,
    #             "yend": 5.75,
    #         }, {
    #             "xstart": x_label.index('6PPDQ500ppb'),
    #             "xend": x_label.index('6PPDQ1ppm'),
    #             "sig": "*",
    #             "ystart": 5.5,
    #             "yend": 5.75,
    #         },{
    #             "xstart": x_label.index('6PPD50ppb'),
    #             "xend": x_label.index('6PPD1ppm'),
    #             "sig": "*",
    #             "ystart": 6.25,
    #             "yend": 6.5,
    #         },
    #          {
    #             "xstart": x_label.index('CK'),
    #             "xend": x_label.index('6PPDQ1ppm'),
    #             "sig": "*",
    #             "ystart": 7.75,
    #             "yend": 8.0,
    #         },
    #     ]
    #
    # drawTransmiteeBar(
    #     x_label, error, y_pos,
    #     y_label_name=y_label_name,
    #     title=f"{index_name}",
    #     unit=unit,
    #     annos=annos
    # )
