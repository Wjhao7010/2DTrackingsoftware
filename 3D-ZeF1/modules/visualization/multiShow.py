import pyecharts.options as opts
from pyecharts.charts import Boxplot
from pyecharts.commons.utils import JsCode
import pandas as pd
import os, sys
import argparse
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
from scipy import stats
import pyecharts.options as opts
from pyecharts.faker import Faker
from pyecharts.charts import Grid, Boxplot, Scatter, Bar, Line
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(".")
sys.path.append('../')
sys.path.append('../../')
from common.utility import *

plt.rcParams['font.sans-serif'] = "Microsoft YaHei"
plt.rcParams['font.size'] = 15
# color = [
#     '#2e75b6',
#     '#c5e0b4', '#ffe699', '#f8cbad'
# ]

color = [
    '#2e75b6',
    '#548235', '#facb6c', '#cf6107'
]
# =================== echarts boxplot  ===========================#
def formatBoxData(data, index_name, exposureT_list, category_name):
    # 种类，重复试验，对应MultipleBox中的Series
    series_names = list(data[category_name].unique())
    drag_data = {}
    save_data = {}
    for drag in series_names:
        drag_data[drag] = {
            'inliers': [],
            'outliers': [],
        }
        save_data[drag] = {}
        for iexp_t in exposureT_list:
            exp_data = data[(data['exposure_time'] == int(iexp_t)) & (data['drag_name'] == drag)][[index_name]]
            std_3 = (exp_data[index_name] - exp_data[index_name].mean()) / exp_data[index_name].std()
            inliers = exp_data[std_3.abs() < 2]
            outliers = exp_data[~(std_3.abs() < 2)]

            # q1, q3 = exp_data[index_name].quantile([0.25, 0.75])
            # iqr = q3 - q1
            # inliers = exp_data[(exp_data[index_name] < iqr * 1.5 + q3) & (exp_data[index_name] > q1 - iqr * 1.5)]
            # outliers = exp_data[(exp_data[index_name] >= iqr * 1.5 + q3) | (exp_data[index_name] <= q1 - iqr * 1.5)]

            # df_Z = exp_data[(np.abs(stats.zscore(exp_data[index_name])) < 1.2)]
            # ix_keep = df_Z.index
            # inliers = exp_data.loc[ix_keep]
            #
            # df_Z = exp_data[(np.abs(stats.zscore(exp_data[index_name])) >= 1.2)]
            # ix_keep = df_Z.index
            # outliers = exp_data.loc[ix_keep]

            drag_data[drag]['inliers'].append(inliers[index_name].tolist())
            drag_data[drag]['outliers'].append(outliers[index_name].tolist())
            save_data[drag][iexp_t] = np.mean(inliers[index_name].tolist())

    return drag_data, save_data


def drawMultipleBox(exposure_T, drag_data, index_info):
    box_plot = Boxplot()

    box_plot.add_xaxis(xaxis_data=[f"{str(int(_) + 1)}" for _ in exposure_T])
    box_plot.set_global_opts(
        title_opts=opts.TitleOpts(
            pos_left="left", title='', item_gap=20
        ),
        toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    type_='jpeg',
                    background_color='white'
                )
            )
        ),

        legend_opts=opts.LegendOpts(
            textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=25,
                color='black'
                # font_weight='bold'
            ),
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item", axis_pointer_type="shadow"
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=True,
            splitarea_opts=opts.SplitAreaOpts(
                areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
            name_textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=25,
                color='black'
                # font_weight='bold'
            ),
            axislabel_opts=opts.LabelOpts(
                font_size=18,
                font_style='normal',
                font_family='Microsoft YaHei'
            ),
            name='Time (every 1 hour)',
            name_location='middle',
            name_gap=45,
            splitline_opts=opts.SplitLineOpts(is_show=False),

        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name=f"{index_info['name']} ({index_info['unit']})",
            name_location='middle',
            name_gap=70,
            name_rotate=90,
            name_textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=25,
                color='black'
                # font_weight='bold'
            ),
            axislabel_opts=opts.LabelOpts(
                font_size=18,
                font_style='normal',
                font_family='Microsoft YaHei',
                color='black'
            ),

        ),

    )
    for dname, y_data in drag_data.items():
        box_plot.add_yaxis(
            series_name=dname,
            y_axis=box_plot.prepare_data(y_data['inliers']),
            itemstyle_opts=opts.ItemStyleOpts(
                border_color0='black',
                opacity=0.8,
                border_width=3,
                color=box_plot.colors[list(drag_data.keys()).index(dname)]
            )
        )
    # scatter = Scatter()
    # scatter.set_global_opts(
    #     legend_opts=opts.LegendOpts(
    #         textstyle_opts=opts.TextStyleOpts(
    #             font_family='Microsoft YaHei',
    #             font_style='normal',
    #             font_size=15,
    #             font_weight='bold'
    #         )
    #     ),
    #     xaxis_opts=opts.AxisOpts(
    #         name_textstyle_opts=opts.TextStyleOpts(font_family='Microsoft YaHei'),
    #     ),
    # )
    # scatter.add_xaxis(
    #     [f"{str(int(_)+1)}" for _ in exposure_T]
    # )
    # for dname, y_data in drag_data.items():
    #     scatter.add_yaxis(
    #         series_name=dname, y_axis=y_data['outliers'], label_opts=opts.LabelOpts(is_show=False)
    #     )

    grid = (
        Grid(init_opts=opts.InitOpts(width="1560px", height="720px"))
            .add(
            box_plot,
            grid_opts=opts.GridOpts(pos_left="10%", pos_right="10%", pos_bottom="15%"),
        )
            # .add(
            # scatter,
            # grid_opts=opts.GridOpts(
            #     pos_left="10%", pos_right="10%", pos_bottom="15%"
            # ),
            # )
            .render(os.path.join(out_path, f"{'_'.join(drag_list)}_{index_info['name']}.html"))
    )
    print(os.path.join(out_path, f"{'_'.join(drag_list)}_{index_info['name']}.html"))

def drawSubMultipleLine(exposure_T, drag_data, index_info):
    # example data
    fig, ax_list = plt.subplots(
        nrows=4, ncols=1, sharex=True,
        figsize=(12, 6)
    )
    ls = 'dotted'

    x = np.array([int(_) + 1 for _ in exposure_T])
    for idx, (index_name, value_list) in enumerate(drag_data.items()):
        y_mean = [np.nanmean(_) for _ in value_list['inliers']]
        y_std = [np.nanstd(_) for _ in value_list['inliers']]
        xerr = 0.1
        # standard error bars
        ax_list[idx].errorbar(x, y_mean, xerr=xerr, yerr=y_std, linestyle=ls)
        ax_list[idx].set_yticks([_/10. for _ in range(0, 15, 3)])
        ax_list[idx].set_xticks([int(_) + 1 for _ in exposure_T])
    ax_list[idx].set_xlabel('Time (every 1 hour)')
    # tidy up the figure

    # ax.set_xlim((0, 5.5))
    # ax.set_title('Errorbar upper and lower limits')

    plt.show()


def drawMultipleLine(exposure_T, drag_data, index_info, annotations=None):
    # example data
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ls = 'dotted'
    markers = ['o', '^', 'v', 'D']
    index_names = []
    x = np.array([int(_) + 1 for _ in exposure_T])
    for idx, (index_name, value_list) in enumerate(drag_data.items()):
        y_mean = [np.nanmean(_+o) for _, o in zip(value_list['inliers'], value_list['outliers'])]
        y_std = [np.nanstd(_+o) for _, o in zip(value_list['inliers'], value_list['outliers'])]
        y_std = [_ if m-_ > 0 else m for _, m in zip(y_std, y_mean)]

        xerr = 0.1
        # standard error bars
        ax.errorbar(
            x, y_mean, xerr=xerr, yerr=y_std,
            linestyle=ls, marker=markers[idx],
            color=color[idx]
        )
        ax.set_yticks(annotations['y_ticks'])
        ax.set_xticks([int(_) + 1 for _ in exposure_T])
        index_names.append(index_name)

        if annotations is not None:
            if index_name in annotations:
                annos = annotations[index_name]
                for ianno in annos:
                    plt.annotate(
                        r'$%s$' % ianno['sig'], xy=(ianno['x0'], ianno['y0']),
                        xycoords='data', xytext=(0, 0),
                        textcoords='offset points',
                        fontsize=12,
                        color=color[idx]
                    )

    ax.legend(loc="best", ncol=2, labels=[_.replace("ppb", " μg/L").replace("ppm", "000 μg/L") for _ in index_names])
    ax.set_xlabel('Time (every 1 hour)')
    ax.set_ylabel(f'{index_info["name"]} ({index_info["unit"]})')
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_path, f"{'_'.join(drag_list)}_{index_info['name']}.png")
        , dpi=300)  # 保存图片
    print(os.path.join(out_path, f"{'_'.join(drag_list)}_{index_info['name']}.png"))
    # plt.show()


def drawLines(y_data, labels, drag_list, title, annotations=None):
    # example data
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ls = 'dotted'
    markers = ['o', '^', 'v', 'D']
    index_names = []
    x = np.arange(len(labels)) + 1
    for idx, drag_name in enumerate(drag_list):
        means_data = [_*100 for _ in y_data[drag_name]['mean']]
        stds_data = [_*100 for _ in y_data[drag_name]['std']]
        xerr = 0.1
        # standard error bars
        ax.errorbar(
            x, means_data, xerr=xerr, yerr=stds_data,
            linestyle=ls, marker=markers[idx],
            color=color[idx]
        )

        ax.set_yticks(annotations['y_ticks'])
        index_names.append(drag_name)

        if annotations is not None:
            if drag_name in annotations:
                annos = annotations[drag_name]
                for ianno in annos:
                    plt.annotate(
                        r'$%s$' % ianno['sig'], xy=(ianno['x0'], ianno['y0']),
                        xycoords='data', xytext=(0, 0),
                        textcoords='offset points',
                        fontsize=12,
                        color=color[idx]
                    )

    ax.legend(loc="best", ncol=2, labels=[_.replace("ppb", " μg/L").replace("ppm", "000 μg/L") for _ in index_names])
    ax.set_xlabel('Time (every 1 hour)')
    ax.set_xticks([_+1 for _ in range(12)])
    ax.set_xticklabels([_+1 for _ in range(12)])

    if title == 'top_time':
        ax.set_ylabel('Ratio of water surface retention (%)')
    else:
        ax.set_ylabel('meandering (%)')

    ax.set_yticks(annotations['y_ticks'])
    ax.set_yticklabels(annotations['yticklabels'])
    plt.tight_layout()
    filename = os.path.join(out_path, f"{title}_{'_'.join(drag_list)}.png")
    print(f"save data in {filename}")
    # plt.show()
    plt.savefig(filename, dpi=300)

# =================== echarts barplot  ===========================#
def formatBarData(data, index_name, exposureT_list, category_name):
    # 种类，重复试验，对应MultipleBox中的Series
    new_index_col = [
        # 'turning_angle_0_30',
        # 'turning_angle_30_60',
        # 'turning_angle_60_90'
        'turning_angle_30_90'
    ]
    # data['turning_angle_0_30'] = data['turning_angle_0_10'] + data['turning_angle_10_20'] + data['turning_angle_20_30']
    data['turning_angle_30_90'] = data['turning_angle_30_40'] + data['turning_angle_40_50'] + data[
        'turning_angle_50_60'] + data['turning_angle_60_70'] + data['turning_angle_70_80'] + data['turning_angle_80_91']
    # data['turning_angle_60_90'] = data['turning_angle_60_70'] + data['turning_angle_70_80'] + data['turning_angle_80_91']
    drag_names = list(data[category_name].unique())
    drag_data = {}
    mean_data = data.groupby(['exposure_time', 'drag_name'])[new_index_col].mean()
    mean_data.reset_index(inplace=True)
    for idrag in drag_names:
        drag_data_cond = (mean_data['drag_name'] == idrag) & (mean_data['exposure_time'].isin(exposureT_list))
        drag_data[idrag] = mean_data[drag_data_cond]
    return drag_data, new_index_col


def drawMultipleBar(exposure_T, drag_data, col_name, index_info, base_col='CK'):
    x_data = [str(int(_) + 1) for _ in exposure_T]

    bar = (
        Bar(
            init_opts=opts.InitOpts(width="1680px", height="600px"),
        ).add_xaxis(x_data)
    )

    for idrag, angle in drag_data.items():
        for col in col_name:
            bar.add_yaxis(f"{idrag}", [_ * 100 for _ in list(angle[col].values)], stack=idrag)
            if idrag == base_col:
                base_line = [_ * 100 for _ in list(angle[col].values)]

    bar.set_series_opts(
        label_opts=opts.LabelOpts(is_show=False),
        # markline_opts=opts.MarkLineOpts(
        #     data=[opts.MarkLineItem(y=_, name="CK") for _ in base_line]
        # ),
    )
    bar.set_global_opts(
        title_opts=opts.TitleOpts(
            pos_left="left", title='', item_gap=20
        ),
        toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    type_='jpeg',
                    background_color='white'
                )
            )
        ),

        legend_opts=opts.LegendOpts(
            textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=15,
                font_weight='bold',
            ),
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item", axis_pointer_type="shadow"
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=True,
            splitarea_opts=opts.SplitAreaOpts(
                areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
            name_textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=15,
                font_weight='bold'
            ),
            # axislabel_opts=opts.LabelOpts(formatter="expr {value}"),
            name='Time (every 1 hour)',
            name_location='middle',
            name_gap=45,
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name=f"{index_info['name']} (%)",
            name_location='middle',
            name_gap=45,
            name_rotate=90,
            name_textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=15,
                font_weight='bold',
                color='black'

            )
        ),
    )

    bar.render(os.path.join(out_path, f"{'_'.join(drag_list)}_{index_info['name']}.html"))


# =================== matplotlib barplot  ===========================#
def drawBars(y_data, labels, drag_list, title, annotations=None):
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(dpi=300)

    color = [
        '#2e75b6',
        '#ffcc66', '#ff9966', '#ff6600'
    ]


    for idx, drag_name in enumerate(drag_list):
        means_data = [_*100 for _ in y_data[drag_name]['mean']]
        stds_data = [_*100 for _ in y_data[drag_name]['std']]
        rects = ax.bar(
            x - width * len(drag_list) / 2 + width*(idx+1) - width /2,
            means_data, width,
            # yerr=stds_data,
            label=drag_name,
            color=color[idx%4]
        )
        # ax.bar_label(rects, padding=1)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Time (every 1 hour)')
    ax.set_xticks(x, labels)
    # ax.set_ylabel('meandering (%)')
    if title == 'top_time':
        ax.set_ylabel('Ratio of water surface retention (%)')
        ax.set_yticks([_*10 for _ in range(11)])
        ax.set_yticklabels([_*10 for _ in range(11)])
        # ax.legend(loc="best",ncol=2,fontsize="x-small")
    else:
        ax.set_ylabel('meandering (%)')
        ax.set_yticks([_ for _ in range(11)])
        ax.set_yticklabels([_ for _ in range(11)])
        # ax.legend(loc="best",fontsize="x-small")
    ax.legend()

    plt.tight_layout()
    filename = os.path.join(out_path, f"{title}_{'_'.join(drag_list)}.png")
    print(f"save data in {filename}")
    # plt.show()
    plt.savefig(filename, dpi=300)


# def drawGroupStackBars():
#     labels = ['G1', 'G2', 'G3', 'G4', 'G5']
#     men_means = [20, 34, 30, 35, 27]
#     men_means1 = [20, 35, 30, 35, 27]
#     women_means = [25, 32, 34, 20, 25]
#     women_means1 = [25, 32, 34, 20, 25]
#     men_std = [2, 3, 4, 1, 2]
#     women_std = [3, 5, 2, 3, 3]
#
#     x = np.arange(len(labels))  # the label locations
#     width = 0.35  # the width of the bars
#
#     fig, ax = plt.subplots()
#     ax.bar(x - width/2, men_means, width, yerr=men_std, label='Men')
#     ax.bar(x - width/2, men_means1, width, yerr=men_std, label='Men1', bottom=men_means)
#
#     ax.bar(x + width/2, women_means, width, yerr=women_std, label='Women')
#     ax.bar(x + width/2, women_means1, width, yerr=men_std, label='Women', bottom=women_means)
#
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('Scores')
#     ax.set_title('Scores by group and gender')
#     ax.set_xticks(x, labels)
#     ax.legend()
#
#     fig.tight_layout()
#
#     plt.show()

def calculateMeanData(data, index_col, group_col=('exposure_time', 'drag_name')):
    group_col = list(group_col)
    use_col = index_col + group_col
    mean_data = data[use_col].groupby(group_col).mean()
    std_data = data[use_col].groupby(group_col).std()
    mean_data.to_csv(os.path.join(out_path, 'mean_data.csv'))
    print("mean data done")
    return mean_data, std_data


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-rp", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("--drag_names",
                    default="CK,"
                    # "4Hydroxy50ppb,4Hydroxy500ppb,4Hydroxy1ppm",
                    # "6PPD50ppb,6PPD500ppb,6PPD1ppm",
                    "6PPDQ50ppb,6PPDQ500ppb,6PPDQ1ppm",
                    # "RJ",

                    # default="4Hydroxy50ppb,6PPDQ1ppm,6PPD50ppb,6PPD500ppb,6PPD1ppm,RJ",
                    type=str)
    ap.add_argument("--exposureT", default="0,1,2,3,4,5,6,7,8,9,10,11", type=str)

    args = vars(ap.parse_args())

    root_path = args['root_path']
    exposureT = args['exposureT']
    drag_names = args['drag_names']

    drag_list = drag_names.split(",")
    exposureT_list = exposureT.split(",")

    turning_angle_interval = (0, 10, 20, 30, 40, 50, 60, 70, 80, 91)
    up_down_angle_interval = (0, 15, 45, 91)

    info_col = ['exposure_time', 'DayTank', 'region_name']
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
    out_path = os.path.join(root_path, 'drag_result', 'multiple')

    topdata_dict = pd.DataFrame()
    sidedata_dict = pd.DataFrame()
    for drag_name in drag_list:
        index_path = os.path.join(root_path, 'drag_result', 'multiple', drag_name)
        side_data = pd.read_csv(os.path.join(index_path, "side_view.csv"))
        top_data = pd.read_csv(os.path.join(index_path, "top_view.csv"))
        top_data['drag_name'] = drag_name
        side_data['drag_name'] = drag_name
        topdata_dict = pd.concat([topdata_dict, top_data])
        sidedata_dict = pd.concat([sidedata_dict, side_data])

    # topdata_dict.to_csv(
    #     os.path.join(out_path, f"topdata_dict.csv")
    # )
    sidedata_dict.to_csv(
        os.path.join(out_path, f"sidedata_dict.csv")
    )
    print(f'save data in {out_path}')

    mean_data, std_data = calculateMeanData(topdata_dict, index_col=['velocity', 'dist_cal'])

    # ============================================================ #
    # velocity_info = {
    #     'name': 'velocity',
    #     'unit': "mm/s",
    # }
    # drag_data, save_data = formatBoxData(
    #     topdata_dict, index_name='velocity', exposureT_list=exposureT_list,
    #     category_name='drag_name'
    # )
    # # drawMultipleBox(exposureT_list, drag_data, velocity_info)
    # drawMultipleLine(exposureT_list, drag_data, velocity_info, annotations={
    #     'y_ticks': [_/10. for _ in range(0, 18, 3)],
    #     # 'CK': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 0.2,
    #     #     } for _ in [5,6,7,8,9,10,11,12]
    #     # ],
    #     # '4Hydroxy50ppb': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 0.16,
    #     #     } for _ in [4,5,6,7,8,9,10,11,12]
    #     # ],
    #     # '4Hydroxy500ppb': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 0.12,
    #     #     } for _ in [7,8,9,10,11,12]
    #     # ],
    #     # '4Hydroxy1ppm': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 0.08,
    #     #     } for _ in [4,5,6,7,8,9,10,11,12]
    #     # ],
    #     # 'CK': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 0.96,
    #     #     } for _ in [5,6,7,8,9,10,11,12]
    #     # ],
    #     # '6PPD50ppb': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 1.0,
    #     #     } for _ in [3,5,6,7,8,9,10,11,12]
    #     # ],
    #     # '6PPD500ppb': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 1.04,
    #     #     } for _ in [6,7,8,9,10,11,12]
    #     # ],
    #     # '6PPD1ppm': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 1.08,
    #     #     } for _ in [2,3,4,5,6,7,8,9,10,11,12]
    #     # ],
    #     'CK': [
    #         {
    #             "sig": "*",
    #             "x0": _,
    #             "y0": 0.96,
    #         } for _ in [5,6,7,8,9,10,11,12]
    #     ],
    #
    #     '6PPDQ500ppb': [
    #         {
    #             "sig": "*",
    #             "x0": _,
    #             "y0": 1.04,
    #         } for _ in [6,7,8,9,10,11,12]
    #     ],
    #     '6PPDQ1ppm': [
    #         {
    #             "sig": "*",
    #             "x0": _,
    #             "y0": 1.08,
    #         } for _ in [3,4,5,6,7,8,9,10,11,12]
    #     ]
    # })
    # pd.DataFrame(save_data).to_csv(
    #     os.path.join(out_path, f"{velocity_info['name']}.csv")
    # )
    # exit(3)

    # dist_info = {
    #     'name': 'dist',
    #     'unit': "mm",
    # }
    # drag_data = formatBoxData(
    #     topdata_dict, index_name='dist', exposureT_list=exposureT_list,
    #     category_name='drag_name'
    # )
    # drawMultipleBox(exposureT_list, drag_data, dist_info)

    # ============================================================ #
    # dist_cal_info = {
    #     'name': 'distance',
    #     'unit': "mm",
    # }
    # drag_data, save_data = formatBoxData(
    #     topdata_dict, index_name='dist_cal', exposureT_list=exposureT_list,
    #     category_name='drag_name'
    # )
    # # drawMultipleBox(exposureT_list, drag_data, dist_cal_info)
    # drawMultipleLine(exposureT_list, drag_data, dist_cal_info, annotations={
    #     'y_ticks': [_*5000. for _ in range(0, 9)],
    #     'CK': [
    #         {
    #             "sig": "*",
    #             "x0": _,
    #             "y0": 25000,
    #         } for _ in [4,5,6,7,8,9,10,11,12]
    #     ],
    #     '4Hydroxy50ppb': [
    #         {
    #             "sig": "*",
    #             "x0": _,
    #             "y0": 26000,
    #         } for _ in [4,5,6,7,8,9,10,11,12]
    #     ],
    #     '4Hydroxy500ppb': [
    #         {
    #             "sig": "*",
    #             "x0": _,
    #             "y0": 27000,
    #         } for _ in [7,8,9,10,11,12]
    #     ],
    #     '4Hydroxy1ppm': [
    #         {
    #             "sig": "*",
    #             "x0": _,
    #             "y0": 28000,
    #         } for _ in [3,4,5,6,7,8,9,10,11,12]
    #     ],
    #
    #     # 'CK': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 25000,
    #     #     } for _ in [4,5,6,7,8,9,10,11,12]
    #     # ],
    #     # '6PPD50ppb': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 26000,
    #     #     } for _ in [5,6,7,8,9,10,11,12]
    #     # ],
    #     # '6PPD500ppb': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 27000,
    #     #     } for _ in [6,7,8,9,10,11,12]
    #     # ],
    #     # '6PPD1ppm': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 28000,
    #     #     } for _ in [2,3,4,5,6,7,8,9,10,11,12]
    #     # ],
    #     #
    #     # 'CK': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 25000,
    #     #     } for _ in [4,5,6,7,8,9,10,11,12]
    #     # ],
    #     #
    #     # '6PPDQ500ppb': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 27000,
    #     #     } for _ in [6,7,8,9,10,11,12]
    #     # ],
    #     # '6PPDQ1ppm': [
    #     #     {
    #     #         "sig": "*",
    #     #         "x0": _,
    #     #         "y0": 28000,
    #     #     } for _ in [2,3,4,5,6,7,8,9,10,11,12]
    #     # ]
    # })
    # pd.DataFrame(save_data).to_csv(
    #     os.path.join(out_path, f"{dist_cal_info['name']}.csv")
    # )
    # print(out_path)

    # ============================================================ #
    # angle_cal_info = {
    #     'name': 'Turning Angle',
    #     'unit': "%",
    # }
    # drag_data, new_index_col = formatBarData(
    #     topdata_dict,
    #     index_name=top_angle_col,
    #     exposureT_list=exposureT_list,
    #     category_name='drag_name'
    # )
    # drawMultipleBar(exposureT_list, drag_data, new_index_col, angle_cal_info)

    # # ==========================top time================================== #
    # # top_time, up_down_angle_0_15,up_down_angle_15_45,up_down_angle_45_91
    #
    # use_index = 'top_time'
    # mean_topdata, std_topdata = calculateMeanData(sidedata_dict, index_col=[use_index])
    # mean_topdata.reset_index(inplace=True)
    # std_topdata.reset_index(inplace=True)
    # y_data = {}
    # save_data = {}
    # for drag_name in drag_list:
    #     mean_data = mean_topdata[mean_topdata['drag_name'] == drag_name][use_index].tolist()
    #     std_data = std_topdata[std_topdata['drag_name'] == drag_name][use_index].tolist()
    #     y_data[drag_name] = {
    #         'mean': mean_data,
    #         'std': std_data,
    #     }
    #     save_data[drag_name] = mean_data
    # labels = [_ + 1 for _ in list(mean_topdata['exposure_time'].unique())]
    # # drawBars(y_data, labels, drag_list, use_index)
    # drawLines(y_data, labels, drag_list, use_index, annotations={
    #     'y_ticks': [_*10 for _ in range(11)],
    #     'yticklabels': [_*10 for _ in range(11)],
    #     '6PPD500ppb': [
    #         {
    #             "sig": "*",
    #             "x0": _,
    #             "y0": 75,
    #         } for _ in [3,4,5,6,7,8,9,10,11,12]
    #     ],
    #     '6PPD1ppm': [
    #         {
    #             "sig": "*",
    #             "x0": _,
    #             "y0": 77,
    #         } for _ in [2,3,4,5,6,7,8,9,10,11,12]
    #     ],
    # })
    #
    #
    # pd.DataFrame(save_data).to_csv(
    #     os.path.join(out_path, f"index_{use_index}.csv")
    # )
    # print(out_path)
    # print("done")
    # exit(222)

    # ==========================meandering time================================== #
    topdata_dict['turning_angle_0_30'] = topdata_dict['turning_angle_0_10'] + topdata_dict['turning_angle_10_20'] + \
                                         topdata_dict['turning_angle_20_30']
    topdata_dict['turning_angle_30_90'] = 1 - topdata_dict['turning_angle_0_30']
    topdata_dict.to_csv(
        os.path.join(out_path, f"topdata_dict.csv")
    )
    mean_topdata, std_topdata = calculateMeanData(topdata_dict, index_col=['turning_angle_30_90'])
    mean_topdata.reset_index(inplace=True)
    std_topdata.reset_index(inplace=True)

    y_data = {}
    save_data = {}
    for drag_name in drag_list:
        mean_data = mean_topdata[mean_topdata['drag_name'] == drag_name]['turning_angle_30_90'].tolist()
        std_data = std_topdata[std_topdata['drag_name'] == drag_name]['turning_angle_30_90'].tolist()
        print(mean_data)
        print(std_data)
        print("===================")
        y_data[drag_name] = {
            'mean': mean_data,
            'std': std_data,
        }
        save_data[drag_name] = mean_data
    labels = [_ + 1 for _ in list(mean_topdata['exposure_time'].unique())]
    # drawBars(y_data, labels, drag_list, 'turning_angle_30_90')
    drawLines(y_data, labels, drag_list, 'turning_angle_30_90', annotations={
        'y_ticks': [_ for _ in range(11)],
        'yticklabels': [_ for _ in range(11)],
        'CK': [
            {
                "sig": "*",
                "x0": _,
                "y0": 6.5,
            } for _ in [3,4,5,6,7,8,9,10,11,12]
        ],
        '4Hydroxy50ppb': [
            {
                "sig": "*",
                "x0": _,
                "y0":6.8,
            } for _ in [3,4,5,6,7,8,9,10,11,12]
        ],
        '4Hydroxy500ppb': [
            {
                "sig": "*",
                "x0": _,
                "y0": 7.1,
            } for _ in [3,4,5,6,7,8,9,10,11,12]
        ],
        '4Hydroxy1ppm': [
            {
                "sig": "*",
                "x0": _,
                "y0": 7.4,
            } for _ in [3,4,5,6,7,8,9,10,11,12]
        ],
        '6PPD50ppb': [
            {
                "sig": "*",
                "x0": _,
                "y0": 6.8,
            } for _ in [2,3,4,5,6,7,8,9,10,11,12]
        ],'6PPD500ppb': [
            {
                "sig": "*",
                "x0": _,
                "y0": 7.1,
            } for _ in [3,4,5,6,7,8,9,10,11,12]
        ],
        '6PPD1ppm': [
            {
                "sig": "*",
                "x0": _,
                "y0": 7.4,
            } for _ in [2,4,5,6,7,8,9,10,11,12]
        ],'6PPDQ50ppb': [
            {
                "sig": "*",
                "x0": _,
                "y0": 6.8,
            } for _ in [3,4,5,6,7,8,9,10,11,12]
        ],'6PPDQ500ppb': [
            {
                "sig": "*",
                "x0": _,
                "y0": 7.1,
            } for _ in [3,4,5,6,7,8,9,10,11,12]
        ],
        '6PPDQ1ppm': [
            {
                "sig": "*",
                "x0": _,
                "y0": 7.4,
            } for _ in [2,3,4,5,6,7,8,9,10,11,12]
        ],
    })
    pd.DataFrame(save_data).to_csv(
        os.path.join(out_path, f"meandering_30_90.csv")
    )
    print(out_path)
    print("done")
