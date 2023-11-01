import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.colors import LightSource

sys.path.append("../../")
sys.path.append("../")
sys.path.append(".")
import argparse
from common.utility import *


def format_hmData(data, array_hm, tl_pos):
    for row in data.iterrows():
        x_p, y_p = tl_pos
        fish_area_tl_x = int(row[1].aa_tl_x) - x_p
        fish_area_tl_y = int(row[1].aa_tl_y) - y_p
        fish_area_br_x = int(row[1].aa_tl_x) + int(row[1].aa_w) - x_p
        fish_area_br_y = int(row[1].aa_tl_y) + int(row[1].aa_h) - y_p
        array_hm[fish_area_tl_y: fish_area_br_y, fish_area_tl_x: fish_area_br_x] += 1
        # if row[1].frame == 125:
        #     print(row[1].frame)
        #     break

    return array_hm


def drawHeatmap(array_hm, bg_img, save_filename):
    plt.imshow(bg_img)
    plt.imshow(
        array_hm, alpha=0.7, cmap='YlGnBu',
    )
    plt.xticks([])
    plt.yticks([])
    print(f"png file are saved in {save_filename}")
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300)  # results in 160x120 px image

def hillmap(array_hm, out_fig):
    nrows, ncols = array_hm.shape
    x = np.linspace(0, ncols, ncols)
    y = np.linspace(0, nrows, nrows)
    x, y = np.meshgrid(x, y)

    x, y, z = x, y, array_hm

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), dpi=300)

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # fig.show()  # results in 160x120 px image
    print(f"png file are saved in {out_fig}")
    plt.tight_layout()
    fig.savefig(out_fig, dpi=300)  # results in 160x120 px image

def getSavePath(drag_name, DayTank, region_no, expT, pic_type='hillmap'):
    out_path = os.path.join(root_path, 'drag_result/figure', drag_name, DayTank, f'region_{region_no}')
    out_fig = os.path.join(out_path, f"{pic_type}_{str(expT+1)}.png")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_fig

def run(bg_file, track_filelist, out_fig, pic_type='hillmap'):
    array_hm = None
    area_info = None
    bg_img = None
    tl_pos = None
    # track file
    all_data_num = 0

    trackfile_info = []
    for track_file in sorted(track_filelist):
        trackfile_info.extend(get_trackFileInfo(root_path, tracker, track_file))

    for itrackerInfo in trackfile_info:
        print(itrackerInfo)
        if (itrackerInfo['cam_view'] == 1) \
                and (itrackerInfo['region_name'] == region_name) \
                and (itrackerInfo['DayTank'] == DayTank):
            print(f"processing file: {DayTank}/{region_name}/{itrackerInfo['filepath']}")
            if area_info is None:
                tl_x, tl_y, br_x, br_y = load_EXP_region_pos_setting(path, itrackerInfo['camNO'])[region_name]
                w = br_x - tl_x
                h = br_y - tl_y
                tl_pos = (tl_x, tl_y)
                image_board = max(w, h)
                bg_img = plt.imread(bg_file)[tl_y:tl_y + image_board, tl_x:tl_x + image_board, :]
                array_hm = np.zeros((image_board, image_board), int)

            data = pd.read_csv(itrackerInfo['filepath'])
            all_data_num += data.shape[0]
            array_hm = format_hmData(data, array_hm, tl_pos)
            # break
    if pic_type == 'heatmap':
        drawHeatmap(array_hm/all_data_num, bg_img, out_fig)
    else:
        hillmap(array_hm/all_data_num, out_fig)

if __name__ == '__main__':
    # downsample = 1
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("-DT", "--DayTank", default="D1_T1", type=str)
    ap.add_argument("--RegionName", default='1_6PPD1ppm', help="region name of experience")
    ap.add_argument("--tracker", default='finalTrack', help="tracker name of experience")
    # ap.add_argument("--exposureT", default='0,1,2,3,4,5,6,7,8,9,10,11', type=str)
    ap.add_argument("--exposureT", default='0', type=str)
    ap.add_argument("--pic_type", default='hillmap', type=str)

    args = vars(ap.parse_args())

    # ARGUMENTS *************
    root_path = args["root_path"]
    DayTank = args["DayTank"]
    tracker = args["tracker"]
    pic_type = args["pic_type"]
    path = os.path.join(root_path, DayTank)

    bg_file = os.path.join(path, 'background/background_cam1.jpg')

    region_name = args["RegionName"]
    region_no, drag_name = region_name.split("_")
    exp_Tlist = [int(_) for _ in args["exposureT"].split(",")]

    for expT in exp_Tlist:
        out_file = getSavePath(drag_name, DayTank, region_no, expT, pic_type)
        track_filelist = get_filname(root_path, drag_name, expT)
        print("track_filelist")
        print(track_filelist)
        run(bg_file, track_filelist, out_file, pic_type)




