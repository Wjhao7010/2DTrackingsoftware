import easyocr
import sys
sys.path.append("../../")
sys.path.append(".")
import argparse
from common.utility import *
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from moviepy.editor import VideoFileClip


def camera_time_pos(camNO):
    return load_time_pos_setting(config_folder, camNO)

def getImgTime(frame, postion, verbose=True):
    t_tl_y, t_br_y, t_tl_x, t_br_x = postion
    frame_area = frame[t_tl_y:t_br_y, t_tl_x:t_br_x]  # 裁剪时间

    if verbose:
        # print(result)
        cv2.imshow('frame', frame)
        cv2.imshow('frame area', frame_area)
        cv2.waitKey(100)

    result = reader.readtext(frame_area.copy())
    if DEBUG:
        print(f"recognize ocr result is {result}")
    if len(result) == 0:
        time_str = str(int(time.time()))
        if DEBUG:
            print(f"current frame can not be recognized by OCR")
        return time_str
    else:
        time_str = ""
        for ires in result:
            time_str += ires[1] + ' '
        print(f"str format time_str is {time_str}")

        try:
            time_str = time_str.replace("Z", '2').replace("z", '2').\
                replace("O", '0').replace("o", '0').replace("a",'0').\
                replace("k", '4').replace("Q", '0').replace("S", '5').\
                replace("12021", "2021").replace("B", "8")

            digital_time_str = re.findall('\d+', time_str)
            digital_str = "".join(digital_time_str)
            assert len(digital_str) == 14, 'orc result digital error!'
            time_str = "_".join(digital_time_str)
            assert len(time_str) == 19, 'orc result length is smaller than true label!'
        except:
            if DEBUG:
                print("extract date frome OCR failed")
            year = time_str[0:4]
            month = time_str[5:7]
            day = time_str[8:10]
            hh = time_str[11:13]
            mm = time_str[14:16]
            ss = time_str[17:19]
            time_str = f"{year}_{month}_{day}_{hh}_{mm}_{ss}"
        return time_str.strip()


def isStartFrame(frame_time_str, start_time, current_unix_time):
    '''

    :param frame_time_str: 某一帧的识别时间
    :param start_time: 设计的开始时间
    :param current_unix_time: 目前的Unix时间，可能会出现该帧识别时间跳回很久之前的情况,
    load video 的时候设置为-1
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
                print(f"frame_unix_time: {frame_unix_time}")
                print(f"current_unix_time: {current_unix_time}")
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
    if frame.shape[0] == 1080:
        iarea = cv2.resize(frame, (720, 1280))
    else:
        iarea = frame
    if verbose:
        cv2.imshow('frame area', iarea)
        cv2.waitKey(0)

    return iarea


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_floder", default="E:\\data\\3D_pre\\D2_T1", type=str, help="video shear dirpath")
    parser.add_argument("--save_path", default="E:\\data\\3D_pre\\D2_T1\\cut_video", type=str, help="video save path")
    parser.add_argument("--video_name",
                        # default="/home/data/HJZ/zef/0918-0919fish",
                        default="ch13_20211012022645.mp4",
                        type=str, help="video shear dirpath"
                        )
    parser.add_argument("--start_time",
                        # default="/home/data/HJZ/zef/0918-0919fish",
                        default="2021_10_12_02_36_00",
                        # default="2020_08_15_07_27_00",
                        type=str, help="vstart_timeideo start to be cut time"
                        )
    parser.add_argument("-spt", "--split_time",
                        default="2",
                        type=int, help="split time (s)"
                        )
    parser.add_argument("-gpuno", "--gpuno",
                        default="0",
                        type=int, help="gpu no"
                        )
    DEBUG = True
    args = parser.parse_args()
    exp_floder = args.exp_floder  # 遍历路径
    config_folder = args.exp_floder  # 遍历路径

    obj_path = args.save_path  # 目标路径
    if not os.path.exists(obj_path):
        os.mkdir(obj_path)
    video_name = args.video_name  # 视频名称
    time_str = video_name.split("_")[-1]
    year = time_str[0:4]
    month = time_str[4:6]
    day = time_str[6:8]
    hh = time_str[8:10]
    mm = time_str[10:12]
    ss = time_str[12:14]
    video_start_time_str = f"{year}_{month}_{day}_{hh}_{mm}_{ss}"
    start_time = args.start_time  # 开始时间
    spt = args.split_time  # 切割间隔

    reader = easyocr.Reader(['en'], gpu=True)  # this needs to run only once to load the model into memory

    processed_list = [
        files
        for files in os.listdir(obj_path)
    ]

    DayTank_setting_path = os.path.join(exp_floder, 'settings.ini')
    failed_time_list = load_Video_start_time(exp_floder, video_name, time_part="VideoStartTime_Failed")

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
    preloading_time = 30

    if time_gap > 11:
        frame_gap = (time_gap - preloading_time) * fps
        # cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_gap))
        # cap.set(cv2.CAP_PROP_POS_MSEC, int(time_gap)*1000.0)
        print(f"jumping {time_gap}")
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
    #     cap.skip_frames(10)
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

            isStartFrame_flag, try_times, time_gap, current_unix_time = isStartFrame(frame_time_str, start_time, current_unix_time)
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
                    if DEBUG:
                        print(
                            f"{frame_time_str} not starting frame {start_time} with {starting_flag_try_times}/{max_try}")
                    continue
                else:
                    time_local = time.localtime(current_unix_time)
                    dt = time.strftime("%Y_%m_%d_%H_%M_%S", time_local)
                    failed_time_list.append(dt)
                    writeConfig(DayTank_setting_path, [('VideoStartTime_Failed', video_name, '\n'.join(failed_time_list))])
                    print(f"{start_time} finding starting frame in {video_name} failed")
                    exit(987)
        #  ================ 检查该帧是否已经处理过 ================ #
        if f"{frame_time_str}_{camNO}.avi" in processed_list:
            print(f"{frame_time_str}_{camNO} has been processed")
            continue

        if init_videoWriter:
            if time_flag < spt * fps:
                if DEBUG:
                    print(f"progress: {time_flag}/{spt * fps}")
                time_flag += 1
                exp_frame = getImgExp(frame)
                videoWriter.write(exp_frame)
            else:
                print(f"{frame_time_str}_{camNO} have finished, saved in {save_video_name}")
                break
    exit(987654)