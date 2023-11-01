# import easyocr
from paddleocr import PaddleOCR, draw_ocr


import sys

sys.path.append("../../")
sys.path.append(".")
import argparse
from common.utility import *
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from moviepy.editor import VideoFileClip


def camera_time_pos(camNO):
    return load_time_pos_setting(config_folder, camNO)


def manual_recorrect(time_str):
    #
    if video_name == 'ch12_20211012020154.mp4':
        if start_time in [
            "2021_10_12_08_11_59", "2021_10_12_08_23_59", "2021_10_12_08_35_59",
            "2021_10_12_08_47_59", "2021_10_12_08_59_59"
        ]:
            time_str = time_str.replace("2021-10-12 8", "2021 10 12 08")
        return time_str
    elif video_name == 'ch08_20211012200000.mp4':
        if start_time in ['2021_10_13_07_10_08']:
            time_str = time_str.replace("2021-10-13@7.10.08@", "2021_10_13_07_10_08")
        return time_str
    elif video_name == "ch12_20211012200000.mp4":
        # 出現年月日無法識別的情況
        time_str = time_str.strip()
        if time_str[:10] in ['2021-10-12', '2021-10-13']:
            timestep_str = time_str[10:].strip()
            if len(timestep_str) >= 8:
                hh = timestep_str[0:2]
                mm = timestep_str[3:5]
                ss = timestep_str[6:8]
                return f"{time_str[:10]}_{hh}_{mm}_{ss}"
    elif video_name == 'ch02_20211013183000.mp4':
        if '2021_10_13_19' in start_time:
            time_str = time_str.replace("2021-10-139", "2021-10-13 19")
    elif video_name == "ch12_20211016193000.mp4":
        if '2021_10_17_01_50' in start_time:
            time_str = time_str.replace("2021-10-1701250", "2021-10-17 01 50")
    elif video_name == "ch12_20211015190000.mp4":
        if '2021_10_16_06_01' in start_time:
            time_str = time_str.replace("2021-10-1606:01100", "2021_10_16_06_01_00")
        if '2021_10_16_06_21' in start_time:
            time_str = time_str.replace("2021-10-1606/21100", "2021_10_16_06_21_00")
        if '2021_10_16_07_47' in start_time:
            time_str = time_str.replace("2021-10-1607:47100", "2021_10_16_07_47_00")
    elif video_name == "ch12_20211014180000.mp4":
        time_str = time_str.replace("2021-10-1421450:00", "2021-10-14 21 49:59"). \
                replace("2021-10-1420449:59", "2021-10-14 20 49:59").\
                replace("2021-10-1505449:59", "2021-10-15 05 49:59"). \
                replace("2021-10-1421449:59", "2021_10_14_21_49_59")
                # replace("2021-10-1421449:59", "2021_10_14_21_49_59")

    # elif video_name == 'ch10_20211013183000.mp4':
    #     if '2021_10_13' in start_time:
    #         time_str = time_str
    # else:
    #     if CamNO == "ch02":
    #         # 出現年月日無法識別的情況
    #         time_str = time_str.strip()
    #         try:
    #             timesteps_str = time_str.split("@")[-2]
    #
    #             digital_timesteps_str = re.findall('\d+', timesteps_str)
    #             digital_timesteps_str = "".join(digital_timesteps_str)
    #             digital_timesteps_str = digital_timesteps_str[-6:]
    #             if video_name in ["ch02_20211012010327.mp4"]:
    #                 digital_timesteps_str = "0" + digital_timesteps_str
    #             if int(digital_timesteps_str[:2]) - int(hh) >= 0:
    #                 time_str = f"{year}_{month}_{day}_" \
    #                     f"{digital_timesteps_str[:2]}_" \
    #                     f"{digital_timesteps_str[2:4]}_" \
    #                     f"{digital_timesteps_str[4:6]}"
    #             else:
    #                 time_str = f"{year}_{month}_{str(int(day) + 1)}_" \
    #                     f"{digital_timesteps_str[:2]}_" \
    #                     f"{digital_timesteps_str[2:4]}_" \
    #                     f"{digital_timesteps_str[4:6]}"
    #         except:
    #             print(f"time_str: split @ in {time_str}")
    return time_str


def getImgTime(frame, postion, verbose=False):
    t_tl_y, t_br_y, t_tl_x, t_br_x = postion
    frame_area = frame[t_tl_y:t_br_y, t_tl_x:t_br_x]  # 裁剪时间

    if verbose:
        # print(result)
        cv2.imshow('frame area', frame)
        cv2.imshow('frame area1', frame_area)
        cv2.waitKey(10)

    # result = reader.readtext(frame_area.copy())
    result = reader.ocr(frame_area.copy(), det=False)

    if DEBUG:
        print(f"recognize ocr result is {result}")
    if len(result) == 0:
        time_str = str(int(time.time()))
        if DEBUG:
            print(f"current frame can not be recognized by OCR")
        return time_str
    else:
        # time_str = ''
        # for ires in result:
        #     time_str += ires[1] + '@'
        time_str = result[0][0]
        time_str = time_str.replace("Z", '2').replace("z", '2'). \
            replace("O", '0').replace("o", '0').replace("a", '0'). \
            replace("k", '4').replace("Q", '0').replace("S", '5'). \
            replace("12021", "2021").replace("B", "8").replace("J", "0"). \
            replace(")", "0").replace("T", "1").replace("202-10", "2021-10"). \
            replace("202-0", "2021-10").replace(":/", ":").replace("2021210-1", "2021-10-1").\
            replace("2029270-1", "2021-10-1").replace("2029210-1", "2021-10-1"). \
            replace("2021210-1", "2021-10-1").replace("2029290-1", "2021-10-1").\
            replace("2021290-1", "2021-10-1").replace("2029240-1", "2021-10-1").\
            replace("202929-1", "2021-10-1").replace("2021410-1", "2021-10-1").\
            replace("2024-10", "2021-10").replace("2029-10", "2021-10").\
            replace("2021240-1", "2021-10-1").replace("202140-1", "2021-10-1").\
            replace("2021540-1", "2021-10-1").replace("2021510-1", "2021-10-1"). \
            replace("2021340-1", "2021-10-1").replace("2021110-1", "2021-10-1"). \
            replace("2021010-1", "2021-10-1").replace("2021310-1", "2021-10-1"). \
            replace("2021810-1", "2021-10-1").replace("2021840-1", "2021-10-1"). \
            replace("2024010-1", "2021-10-1").replace("2021-0-1", "2021-10-1"). \
            replace("2024540-1", "2021-10-1").replace("20210-1", "2021-10-1"). \
            replace("2021040-1", "2021-10-1").replace("2024810-1", "2021-10-1"). \
            replace("2024210-1", "2021-10-1")
        # print(f"str format time_str is {time_str}")

        try:
            digital_time_str = re.findall('\d+', time_str)
            digital_str = "".join(digital_time_str)
            assert len(digital_str) == 14, 'orc result digital error!'
            # time_str = "_".join(digital_time_str)
            year = digital_str[0:4]
            month = digital_str[4:6]
            day = digital_str[6:8]
            hh = digital_str[8:10]
            mm = digital_str[10:12]
            ss = digital_str[12:14]
            time_str = f"{year}_{month}_{day}_{hh}_{mm}_{ss}"
            assert len(time_str) == 19, 'orc result length is smaller than true label!'
        except:
            # if DEBUG:
            print(f"extract date frome OCR failed with {time_str}")
            time_str = manual_recorrect(time_str)

            year = time_str[0:4]
            month = time_str[5:7]
            day = time_str[8:10]
            hh = time_str[11:13]
            mm = time_str[14:16]
            ss = time_str[17:19]
            time_str = f"{year}_{month}_{day}_{hh}_{mm}_{ss}"
            print(f"manual correct with {time_str}")
        return time_str.strip()


def isStartFrame(frame_time_str, start_time, current_unix_time):
    '''

    :param frame_time_str: 某一帧的识别时间
    :param start_time: 设计的开始时间
    :param current_unix_time: 目前的Unix时间，可能会出现该帧识别时间跳回很久之前的情况
    :return:
    '''
    try:
        frame_time_date = time.strptime(frame_time_str, "%Y_%m_%d_%H_%M_%S")
        frame_unix_time = time.mktime(frame_time_date)

        start_time_date = time.strptime(start_time, "%Y_%m_%d_%H_%M_%S")
        start_unix_time = time.mktime(start_time_date)

        if frame_unix_time > current_unix_time:
            if current_unix_time == -1:
                current_unix_time = frame_unix_time
                print(f"current_unix_time: {current_unix_time}")
            else:
                # 需要判断识别日期是否在合理范围内，合理才更新current unix time
                # print(f"frame_unix_time: {frame_unix_time}")
                # print(f"current_unix_time: {current_unix_time}")
                assert frame_unix_time - start_unix_time < preloading_time, f"{frame_unix_time} is too larger than {current_unix_time}"
                current_unix_time = frame_unix_time

        if current_unix_time < start_unix_time:
            if DEBUG:
                print(f"current time is slower than start time with {start_unix_time - current_unix_time} s")
            return False, 0, start_unix_time - current_unix_time, current_unix_time
        elif current_unix_time == start_unix_time:
            if DEBUG:
                print("current frame == start time")
            return True, 0, 0, current_unix_time
        else:
            return False, max_try, -1, current_unix_time
    except:
        if DEBUG:
            print(f"OCR Result can not be formated by function time.strptime")
        return False, 1, -1, current_unix_time


def getImgExp(frame, verbose=False):
    '''exp_info 存储的值：
    {
        1_1: {
            581, 93, 1020, 524
        }，
        2_CK: {
            1050, 79, 1514, 535
        }
    }
    :param frame:
    :param exp_info:
    :return:
    '''
    # 获取当前实验区域最长的边：宽或者高，后续补全padding 到640*640需使用
    # iarea = frame[0:h, int(3 * w / 16):int(3 * w / 4)]
    if frame.shape[1] == 1080:
        iarea = cv2.resize(frame, (1280, 720))
    else:
        iarea = frame
        # iarea = frame
    if verbose:
        cv2.imshow('frame area', iarea)
        cv2.waitKey(0)

    return iarea


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_floder", default="E:\\data\\3D_pre\\D7_T5", type=str, help="video shear dirpath")
    parser.add_argument("--save_path", default="E:\\data\\3D_pre\\D7_T5\\cut_video", type=str, help="video save path")
    parser.add_argument("--video_name",
                        # default="/home/data/HJZ/zef/0918-0919fish",
                        default="ch14_20211017190000.mp4",
                        type=str, help="video shear dirpath"
                        )
    parser.add_argument("--start_time",
                        # default="/home/data/HJZ/zef/0918-0919fish",
                        default="2021_10_18_08_15_00",
                        # default="2020_08_15_07_27_00",
                        type=str, help="vstart_timeideo start to be cut time"
                        )
    parser.add_argument("-spt", "--split_time",
                        default="2",
                        type=int, help="split time (s)"
                        )
    parser.add_argument("-gpuno", "--gpuno",
                        default="0",
                        type=str, help="gpu no"
                        )
    parser.add_argument("-pt", "--preloading_time",
                        default=50,
                        type=int, help="preloading_time"
                        )
    DEBUG = False
    args = parser.parse_args()
    gpuno = args.gpuno  # 遍历路径
    preloading_time = args.preloading_time  # 遍历路径
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuno
    exp_floder = args.exp_floder  # 遍历路径
    config_folder = args.exp_floder  # 遍历路径

    obj_path = args.save_path  # 目标路径
    if not os.path.exists(obj_path):
        os.mkdir(obj_path)
    video_name = args.video_name  # 视频名称
    CamNO, time_str = video_name.split("_")
    year = time_str[0:4]
    month = time_str[4:6]
    day = time_str[6:8]
    hh = time_str[8:10]
    mm = time_str[10:12]
    ss = time_str[12:14]
    video_start_time_str = f"{year}_{month}_{day}_{hh}_{mm}_{ss}"
    start_time = args.start_time  # 开始时间
    spt = args.split_time  # 切割间隔
    # reader = easyocr.Reader(['en'], gpu=True)  # this needs to run only once to load the model into memory
    # reader = easyocr.Reader(['en'], gpu=True, recog_network='latin_g2')  # this needs to run only once to load the model into memory
    reader = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory

    DayTank_setting_path = os.path.join(exp_floder, 'settings.ini')
    # failed_time_list = load_Video_start_time(exp_floder, video_name)
    failed_time_list = load_Video_start_time(exp_floder, video_name, time_part='VideoStartTime_Failed')

    # camera_id_list = ['D01', 'D02', 'D04']
    camera_id_list = load_Cam_list(config_folder)

    camNO, VideoTime = video_name.split('_')
    # 打开文件，准备数据
    # cap = cv2.VideoCapture(os.path.join(exp_floder, video_name))

    # using moviepy
    cap = FFMPEG_VideoReader(os.path.join(exp_floder, video_name), True)
    cap.initialize()
    # using moviepy

    # 如果当前帧时间距离设定开始时间过远,time_gap为秒，则需要跳转到 距离开始帧3秒内
    _, _, time_gap, current_unix_time = isStartFrame(
        video_start_time_str, start_time, -1
    )
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = 25
    # preloading_time = 40
    if time_gap > 11:
        frame_gap = (time_gap - preloading_time) * fps
        # cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_gap))
        # cap.set(cv2.CAP_PROP_POS_MSEC, int(time_gap)*1000.0)
        # print(f"jumping {frame_gap}")
        # success, image = cap.read()
        # print(f"jumping {success}")

        # using moviepy
        image = cap.get_frame(time_gap - preloading_time)
        # using moviepy

        if DEBUG:
            cv2.imshow('jump frame area', image)
            cv2.waitKey(100)

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
    # total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cut_name = None

    # starting_flag 用来判断第一帧图像的时间是否小于预计要求的开始时间 start_time
    starting_flag = True
    init_videoWriter = False
    starting_flag_try_times = 0
    max_try = preloading_time * fps
    time_flag = 0

    while True:
        # success, frame = cap.read()
        # print(f"current frame no: {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
        # frame = img_as_float(frame)
        # frame = img_as_ubyte(frame)
        # ====================== using moviepy ==========================#
        # cap.skip_frames(10)
        frame = cap.read_frame()
        # ====================== using moviepy ==========================#
        time_pos = camera_time_pos(camNO)
        frame_time_str = getImgTime(frame, time_pos)
        # cv2.imshow('frame area', frame)
        # cv2.waitKey(100)
        # if time_flag == 0:
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, 45000)
        #     time_flag += 1
        # continue
        # ================= 检查是否是初始帧 ===================== #
        if starting_flag:
            # isStartFrame
            # OCR识别失败，try_times=1,
            # 当前帧小于开始帧，try_times=-1，
            # 当前帧大于开始帧，try_times=max_try，

            isStartFrame_flag, try_times, time_gap, current_unix_time = isStartFrame(frame_time_str, start_time,
                                                                                     current_unix_time)
            if DEBUG:
                print(f"time gap is {time_gap}")
            # 如果是开始帧，则之后的starting flag都置为False，不在检查帧时间
            if isStartFrame_flag:
                starting_flag = False
                # 第一个视频
                # ================= 正式开始截取视频 ===================== #
                old_time_str = start_time
                cut_name = start_time

                exp_frame = getImgExp(frame)
                # exp_frame: 2_CK: 截取的frame 画面
                if DEBUG:
                    print(f"first time to cutting new video {frame_time_str}-{camNO}")
                save_video_name = os.path.join(
                    obj_path, f"{frame_time_str}_{camNO}.avi"
                )
                videoWriter = cv2.VideoWriter(
                    save_video_name, fourcc, fps, (exp_frame.shape[1], exp_frame.shape[0]), True
                )
                init_videoWriter = True

            # 如果不是开始帧，则查看是否小于最大尝试次数，否则退出
            else:
                if starting_flag_try_times < max_try:
                    starting_flag_try_times += try_times
                    print(
                        f"{frame_time_str} not starting frame {start_time} with {starting_flag_try_times}/{max_try}")
                    continue
                else:
                    # time_local = time.localtime(current_unix_time)
                    # dt = time.strftime("%Y_%m_%d_%H_%M_%S", time_local)
                    if frame_time_str not in failed_time_list:
                        failed_time_list.append(frame_time_str)
                        writeConfig(exp_floder, [('VideoStartTime_Failed', video_name, '\n'.join(failed_time_list))])
                    print(f"{start_time} finding starting frame in {video_name} failed")
                    exit(987)
        #  ================ 检查该帧是否已经处理过 ================ #
        if init_videoWriter:
            if time_flag < spt * fps:
                if time_flag % 100 == 0:
                    print(f"progress: {video_name} / {start_time}: {time_flag}/{spt * fps}")
                time_flag += 1
                exp_frame = getImgExp(frame)
                videoWriter.write(exp_frame)
            else:
                print(f"{frame_time_str}_{camNO} have finished, saved in {save_video_name}")
                break
    exit(987654)
