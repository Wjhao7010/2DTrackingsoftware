import pandas as pd
import numpy as np
import os, sys
import argparse
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt

class BasicStatistic(object):
    def __init__(self, index_data):
        self.index_data = index_data

    def groupbyExpData(self, stat_index=None, stat_method='mean'):
        stat_index = index_data.columns.tolist() if stat_index is None else stat_index
        if stat_method == 'mean':
            table = pd.pivot_table(
                self.index_data, values=stat_index, index=['exposure_time', 'region_name', 'DayTank'],
                aggfunc=np.nanmean
            )
        elif stat_method == 'max':
            table = pd.pivot_table(
                self.index_data, values=stat_index, index=['exposure_time', 'region_name', 'DayTank'],
                aggfunc=np.nanmax
            )
        elif stat_method == 'sum':
            table = pd.pivot_table(
                self.index_data, values=stat_index, index=['exposure_time', 'region_name', 'DayTank'],
                aggfunc=np.nansum
            )
        else:
            raise ValueError("unsupported statistic method")

        return table

    def drawTimeBoxes(self, box_data):
        box_data = pd.DataFrame(
            box_data,
            columns = [f'exposure_{_+1}' for _ in range(12)]
        )# 先生成0-1之间的5*4维度数据，再装入4列DataFrame中
        box_data.boxplot()  # 也可用plot.box()
        plt.show()

    def drawHillShading(self):
        # Load and format data
        dem = cbook.get_sample_data('jacksboro_fault_dem.npz', np_load=True)
        z = dem['elevation']
        nrows, ncols = z.shape
        x = np.linspace(dem['xmin'], dem['xmax'], ncols)
        y = np.linspace(dem['ymin'], dem['ymax'], nrows)
        x, y = np.meshgrid(x, y)

        region = np.s_[5:50, 5:50]
        x, y, z = x[region], y[region], z[region]

        # Set up plot
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        ls = LightSource(270, 45)
        # To use a custom hillshading mode, override the built-in shading and pass
        # in the rgb colors of the shaded surface calculated from "shade".
        rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=False, shade=False)

        plt.show()

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-rp", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("-drag", "--drag_name", default="6PPD1ppm", type=str)

    args = vars(ap.parse_args())

    root_path = args['root_path']
    drag_name = args['drag_name']

    index_path = os.path.join(root_path, 'drag_result', drag_name)
    # drag_result_data = pd.DataFrame()

    side_col = ['top_time', 'angle_15_45', 'angle_45_90', 'angle_0_15', 'exposure_time', 'DayTank', 'exp_time',
                'filename', 'region_name']
    top_col = [f"pos_distribution_block_{_ + 1}" for _ in range(25)] + ['dist', 'exposure_time', 'DayTank',
                                                                        'exp_time', 'filename', 'region_name']

    top_result_data = pd.DataFrame()
    side_result_data = pd.DataFrame()

    for ifile in os.listdir(index_path):
        index_data = pd.read_csv(os.path.join(index_path, ifile))
        if 'exp_info' in index_data.columns.tolist():
            index_data.drop(['exp_info'], axis=1, inplace=True)
            # index_data.drop(index_data[index_data['DayTank']=='D7_T5'].index, axis=0, inplace=True)
        top_index_data = index_data.copy()[top_col]
        top_index_data.dropna(axis=0, how='any', inplace=True)
        side_index_data = index_data.copy()[side_col]
        side_index_data.dropna(axis=0, how='any', inplace=True)
        top_index_data = index_data.copy()[top_col]
        side_index_data.dropna(axis=0, how='any', inplace=True)
        top_result_data = pd.concat([top_result_data, top_index_data])
        side_result_data = pd.concat([side_result_data, side_index_data])


    Tanalyzer = BasicStatistic(top_result_data)
    Sanalyzer = BasicStatistic(side_result_data)
    Tmean_data = Tanalyzer.groupbyExpData(stat_index=top_col)
    Smean_data = Sanalyzer.groupbyExpData(stat_index=side_col)

    Tmean_data.reset_index(inplace=True)

    # dist_data = Tmean_data[['dist']].values.reshape(Tmean_data.shape[0] // 12, -1)
    # print(dist_data)
    # Tanalyzer.drawTimeBoxes(dist_data)

    # v_data = Tmean_data[['velocity']].values.reshape(Tmean_data.shape[0] // 12, -1)
    # print(dist_data)
    # Tanalyzer.drawTimeBoxes(dist_data)

    pos_distribution = Tmean_data[[f"pos_distribution_block_{_ + 1}" for _ in range(25)]]
    print(pos_distribution)
    Tanalyzer.drawHillShading()
