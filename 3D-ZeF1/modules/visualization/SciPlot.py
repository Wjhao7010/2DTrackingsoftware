import os, sys
import argparse
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append('../../')
from common.utility import *


# ===================  boxplot  ===========================#
def formatBoxData(data, index_name, exposureT_list, category_name):
    # 种类，重复试验，对应MultipleBox中的Series
    series_names = list(data[category_name].unique())
    drag_data = {}
    for drag in series_names:
        drag_data[drag] = {
            'inliers': [],
            'outliers': [],
        }
        for iexp_t in exposureT_list:
            exp_data = data[(data['exposure_time'] == int(iexp_t)) & (data['drag_name'] == drag)][[index_name]]
            std_3 = (exp_data[index_name] - exp_data[index_name].mean()) / exp_data[index_name].std()
            inliers = exp_data[std_3.abs() < 3]
            outliers = exp_data[~(std_3.abs() < 3)]
            drag_data[drag]['inliers'].append(inliers[index_name].tolist())
            drag_data[drag]['outliers'].append(outliers[index_name].tolist())

    return drag_data

def drawCompareBox(exposureT_list, drag_data, velocity_info):


    # Random test data
    np.random.seed(19680801)
    all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
    labels = ['x1', 'x2', 'x3']

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

    # rectangular box plot
    bplot1 = ax1.boxplot(all_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    ax1.set_title('Rectangular box plot')


    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines
    ax1.yaxis.grid(True)
    ax1.set_xlabel('Three separate samples')
    ax1.set_ylabel('Observed values')

    plt.show()


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-rp", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("--drag_names",
                    default="4Hydroxy50ppb,4Hydroxy500ppb,4Hydroxy1ppm,"
                            # "6PPD50ppb,6PPD500ppb,6PPD1ppm,"
                            # "6PPDQ50ppb,6PPDQ500ppb,6PPDQ1ppm,"
                            # "RJ,"
                            "CK",
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

    velocity_info = {
        'name': 'velocity',
        'unit': "mm/s",
    }
    drag_data = formatBoxData(
        topdata_dict, index_name='velocity', exposureT_list=exposureT_list,
        category_name='drag_name'
    )
    drawCompareBox(exposureT_list, drag_data, velocity_info)
    #
    # dist_info = {
    #     'name': 'dist',
    #     'unit': "mm",
    # }
    # drag_data = formatBoxData(
    #     topdata_dict, index_name='dist', exposureT_list=exposureT_list,
    #     category_name='drag_name'
    # )
    # drawMultipleBox(exposureT_list, drag_data, dist_info)
    #
    # dist_cal_info = {
    #     'name': 'dist_cal',
    #     'unit': "mm",
    # }
    # drag_data = formatBoxData(
    #     topdata_dict, index_name='dist_cal', exposureT_list=exposureT_list,
    #     category_name='drag_name'
    # )
    # drawMultipleBox(exposureT_list, drag_data, dist_cal_info)

    # angle_cal_info = {
    #     'name': 'turning angle',
    #     'unit': "percent",
    # }
    # drag_data, new_index_col = formatBarData(
    #     topdata_dict,
    #     index_name=top_angle_col,
    #     exposureT_list=exposureT_list,
    #     category_name='drag_name'
    # )
    # drawMultipleBar(exposureT_list, drag_data, new_index_col, angle_cal_info)

    print("done")
