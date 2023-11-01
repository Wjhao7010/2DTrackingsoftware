import argparse
import sys

sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from common.utility import *
from modules.tracking.tools.utilits import *
from modules.quantify.quantifyIndex import get_block_area
#
# ROI_Areafuck = {
#     '1_6PPD1ppm': [513, 129, 913, 531],
#     '2_6PPD1ppm': [945, 129, 1337, 539],
#     '3_6PPD1ppm': [501, 557, 909, 581],
#     '4_6PPD1ppm': [931, 561, 1327, 949],
# }


def boarder_info(cfg_path, camNO, RegionName, it_tracker_data, board_coef=5):
    water_ImgPos = load_EXP_region_pos_setting(
        cfg_path, camNO
    )[RegionName]

    out_tl_x, out_tl_y, out_br_x, out_br_y = water_ImgPos
    w = out_br_x - out_tl_x
    h = out_br_y - out_tl_y

    w_iblock = w / board_coef
    h_iblock = h / board_coef

    for row in range(int(board_coef)):
        inner_tl_x = out_tl_x
        inner_tl_y = out_tl_y + row * h_iblock
        for col in range(int(board_coef)):
            tl_x = inner_tl_x + w_iblock * col
            br_x = inner_tl_x + w_iblock * (col + 1)
            tl_y = inner_tl_y
            br_y = inner_tl_y + h_iblock

            condition1 = (it_tracker_data['c_x'] >= tl_x)

            condition2 = (it_tracker_data['c_y'] >= tl_y)

            condition3 = (it_tracker_data['c_x'] <= br_x)

            condition4 = (it_tracker_data['c_y'] <= br_y)
            board_data = (condition1) & (condition2) & (condition3) & (condition4)
            if board_data:
                return row * board_coef + col + 1


def top_info(cfg_path, camNO, RegionName, y_pos):
    itl_x, itl_y, ibr_x, ibr_y = load_EXP_region_pos_setting(
        cfg_path, camNO
    )[RegionName]
    water_Depth = ibr_y - itl_y
    top_water_line = itl_y + water_Depth // 4
    if y_pos <= top_water_line:
        return True
    else:
        return False


def showFrame(frame, fish_row):
    if "process" in tracker:
        kps, bbs = readDetectionCSV(fish_row, camId, ROI_Area, downsample=downsample)
    else:
        kps, bbs = readTrackCSV(fish_row, downsample=downsample)
    # draw keypoint
    frame = cv2.drawKeypoints(frame, kps, None, (255, 0, 0), 4)

    cv2.rectangle(frame, (int(ROI_Area[0]), int(ROI_Area[1])),
                  (int(ROI_Area[2]), int(ROI_Area[3])), (0, 0, 0), 1)

    for bb in bbs:
        cv2.rectangle(frame, (int(bb["aa_tl_x"]), int(bb["aa_tl_y"])),
                      (int(bb["aa_tl_x"] + bb["aa_w"]), int(bb["aa_tl_y"] + bb["aa_h"])), (0, 0, 0), 1)

        if "process" in tracker:
            pass
        else:
            cv2.putText(frame, str(bb["id"]), (int(bb["aa_tl_x"]), int(bb["aa_tl_y"])), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 255),
                        1)

        cv2.circle(frame, (int(bb["l_x"]), int(bb["l_y"])), 2, (255, 0, 0), -1)
        cv2.circle(frame, (int(bb["c_x"]), int(bb["c_y"])), 2, (0, 255, 0), -1)
        cv2.circle(frame, (int(bb["r_x"]), int(bb["r_y"])), 2, (0, 0, 255), -1)

        try:
            if show_info:
                # 身体角度
                if round(bb["w"], 2) / round(bb["h"], 2) >= 2.75:
                    cv2.putText(
                        frame, f'body angle: w: {round(bb["theta"], 2)}',
                        (100, 130),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 1
                    )
                if camId == 1:
                    # 是否处于边界
                    blockid = boarder_info(path, camNO, region_name, bb)
                    cv2.putText(
                        frame, f'at block_{blockid}',
                        (100, 160),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 255), 1
                    )

                if camId != 1:
                    # 是否处于水箱上方
                    if top_info(path, camNO, region_name, int(bb["c_y"])):
                        cv2.putText(
                            frame, 'at top',
                            (100, 100),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 0, 255), 1
                        )
                    else:
                        cv2.putText(
                            frame, 'in water',
                            (100, 100),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 255, 0), 1
                        )
        except:
            print("unable to draw info")
            continue

    if saveVideo:
        out_frame = cv2.resize(frame, out_vid_size)
        videoWriter.write(out_frame)
    else:
        if (platform.system() == 'Windows'):
            cv2.imshow('test', frame)
            cv2.waitKey(100)
    # ******************************************************************* /


if __name__ == '__main__':
    # downsample = 1
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("-DT", "--DayTank", default="D2_T1", type=str)
    ap.add_argument("-pd", "--preDetector", default=True, action='store_true',
                    help="Use pre-computed detections from csv file")
    ap.add_argument("--RegionName", default='3_6PPD1ppm', help="region name of experience")
    ap.add_argument("--track_file", default='2021_10_13_02_32_13_ch03.csv', help="region name of experience")
    # ap.add_argument("--tracker", default='bg_processed', help="tracker name of experience")
    # ap.add_argument("--tracker", default='tracklets_2D', help="tracker name of experience")
    ap.add_argument("--tracker", default='finalTrack', help="tracker name of experience")
    # ap.add_argument("--tracker", default='sortTracker', help="tracker name of experience")
    # ap.add_argument("--tracker", default='sortTracker-refine', help="tracker name of experience")
    # ap.add_argument("--tracker", default='gapfill', help="tracker name of experience")

    ap.add_argument("--showTracklet", default=True, help="region name of experience")
    ap.add_argument("--show_info", default=True, help="region name of experience")
    ap.add_argument("--saveVideo", default=True, help="region name of experience")

    args = vars(ap.parse_args())

    # ARGUMENTS *************
    show_info = args["show_info"]

    root_path = args["root_path"]
    DayTank = args["DayTank"]
    tracker = args["tracker"]
    path = os.path.join(root_path, DayTank)

    region_name = args["RegionName"]
    # track file
    track_file = args["track_file"]
    saveVideo = args["saveVideo"]
    # 特殊处理


    tra_filename = os.path.join(path, tracker, region_name, track_file)
    track_data = pd.read_csv(tra_filename, sep=",")

    track_data.fillna(0, inplace=True)

    # video file config
    video_name = track_file.replace("csv", "avi")
    camNO = video_name.split(".")[0].split("_")[-1]
    video_nameT = '_'.join(video_name.split(".")[0].split("_")[: -1])
    # Configure settings for either reading images or the video file.
    video_path = os.path.join(path, 'cut_video')
    vid_file = os.path.join(video_path, video_name)
    print("processing video: ", vid_file)

    # 以视频的形式存储结果
    if saveVideo:
        vid_file = os.path.join(root_path, DayTank, 'cut_video', video_name)
        vid_out_file = os.path.join(path, tracker, region_name, video_name)
        cap = cv2.VideoCapture(vid_file)
        fps = 25  # 视频帧率
        out_vid_size = (640, 480)
        videoWriter = cv2.VideoWriter(vid_out_file, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, out_vid_size)


    camId = camera_id_map[camNO]

    # load deep learning detect result

    base_cfg_path = path
    ROI_Area = load_EXP_region_pos_setting(base_cfg_path, camNO)[region_name]

    # Close program if video file could not be opened
    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        print("Could not open video file {0}".format(vid_file))
        sys.exit()

    frameCount = 0

    # Analyse and detect fish using the video file
    while (cap.isOpened()):
        ret, frame = cap.read()



        frameCount += 1
        if frameCount % 500 == 0:
            print(f"frameCount: {frameCount}")

        if (ret) and (frame is not None):
            if track_file in [
                '2021_10_12_08_49_59_ch02.csv',
                '2021_10_12_09_09_59_ch02.csv',
                '2021_10_12_09_29_59_ch02.csv',
                '2021_10_12_09_49_59_ch02.csv'
            ]:
                downsample = 1
                frame = cv2.resize(frame, (1280, 720))
            else:
                downsample = 1

            fish_row = track_data.loc[(track_data['frame'] == frameCount)]
            if len(fish_row) == 0:
                continue

            dets = showFrame(frame, fish_row)

        else:
            if (frameCount > 7500):
                frameCount -= 1
                break
            else:
                continue
    cap.release()
    cv2.destroyAllWindows()
    print("done")
