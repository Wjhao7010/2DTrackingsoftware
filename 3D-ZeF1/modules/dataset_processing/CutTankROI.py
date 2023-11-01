import os, sys
sys.path.append("../../")
sys.path.append(".")
import argparse
from common.utility import *
# python中的多线程无法利用多核优势，
# 如果想要充分地使用多核CPU的资源，
# 在python中大部分情况需要使用多进程。
# Python提供了multiprocessing。
import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

def getImgTime(frame, postion, verbose=False):
    t_tl_y, t_br_y, t_tl_x, t_br_x = postion
    frame_area = frame[t_tl_y:t_br_y, t_tl_x:t_br_x]  # 裁剪时间

    result = reader.readtext(frame_area.copy())
    if verbose:
        print(result)
        cv2.imshow('frame area', frame)
        cv2.waitKey(0)
    if len(result) == 0:
        time_str = str(int(time.time()))
        print(f"current frame can not be recognized by OCR")
        return time_str
    else:
        try:
            time_str = re.findall('\d+', result[0][1])
            time_str = "_".join(time_str)
        except:
            time_str = result[0][1]
            year = time_str[0:4]
            month = time_str[5:7]
            day = time_str[8:10]
            hh = time_str[11:13]
            mm = time_str[14:16]
            ss = time_str[17:19]
            time_str = f"{year}_{month}_{day}_{hh}_{mm}_{ss}"
        return time_str

def getImgExp(frame, exp_info, verbose=False):
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
    frame_area = {}
    for region_name, position in exp_info.items():
        tl_x, tl_y, br_x, br_y = position

        iarea = frame[tl_y:br_y, tl_x:br_x]
        # 获取当前实验区域最长的边：宽或者高，后续补全padding 到640*640需使用
        v = max(iarea.shape)

        iarea = np.pad(iarea, ((v-iarea.shape[0], 0), (0, v-iarea.shape[1]), (0, 0)), 'constant', constant_values=0)
        iarea = cv2.resize(iarea, (640, 640))
        # frame = cv2.resize(frame, (640, 640))
        frame_area[region_name] = iarea
        if verbose:
            cv2.imshow('frame area', frame_area[region_name])
            cv2.waitKey(0)
    return frame_area

def isStartFrame(current_time_str, start_time):
    try:
        current_time_date = time.strptime(current_time_str, "%Y_%m_%d_%H_%M_%S")
        current_unix_time = time.mktime(current_time_date)

        start_time_date = time.strptime(start_time, "%Y_%m_%d_%H_%M_%S")
        start_unix_time = time.mktime(start_time_date)

        if current_unix_time < start_unix_time:
            print(f"current time is slower than start time with {start_unix_time - current_unix_time} s")
            return False, -1, start_unix_time - current_unix_time
        elif current_unix_time == start_unix_time:
            print("current frame == start time")
            return True, -1, 0
        else:
            return False, max_try, -1
    except:
        print(f"OCR Result can not be formated by function time.strptime")
        return False, 0, -1


if __name__ == '__main__':
    # 是否需要并行运行
    if_parallel = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_floder",
                        # default="/home/data/HJZ/zef/exp_pre",
                        default="E:\\data\\3D_pre\\0918-0919fish",
                        type=str, help="video shear dirpath"
                        )
    parser.add_argument("--video_name",
                        # default="/home/data/HJZ/zef/0918-0919fish",
                        default="D01_20210918195000.mp4",
                        type=str, help="video shear dirpath"
                        )
    parser.add_argument("--start_time",
                        # default="/home/data/HJZ/zef/0918-0919fish",
                        # default="2021_09_18_19_51_20",
                        default="2021_09_18_20_31_20",
                        type=str, help="vstart_timeideo start to be cut time"
                        )
    parser.add_argument("-spt", "--split_time",
                        default="60",
                        type=int, help="split time (s)"
                        )
    parser.add_argument("-ln", "--lasting_no",
                        default="60",
                        type=int, help="从当前视频对应的开始时间开始，以split_time为间隔，往后切多少个视频"
                        )
    args = parser.parse_args()
    exp_floder = args.exp_floder  # 遍历路径
    config_folder = args.exp_floder  # 遍历路径
    video_name = args.video_name  # 视频名称
    start_time = args.start_time  # 开始时间
    spt = args.split_time  # 切割间隔
    lasting_no = args.lasting_no  # 往后切的数量

    camera_time_pos = {
        "D01": load_time_pos_setting(config_folder, 'D01'),
        "D02": load_time_pos_setting(config_folder, 'D02'),
        "D04": load_time_pos_setting(config_folder, 'D04'),
    }
    Exp_region_pos = {
        "D01": load_EXP_region_pos_setting(config_folder, 'D01'),
        "D02": load_EXP_region_pos_setting(config_folder, 'D02'),
        "D04": load_EXP_region_pos_setting(config_folder, 'D04'),
    }

    camera_id_list = load_Cam_list(config_folder)
    region_names = load_EXP_region_name(config_folder)

    for region_name in region_names:
        # data_root + {CK_1} : /home/data/HJZ/zef/0918-0919fish/1_1
        output_floder = os.path.join(exp_floder, 'cut_video', region_name)
        if not os.path.isdir(output_floder):
            os.mkdir(output_floder)
        processed_list = [
            files
            for files in os.listdir(output_floder)
        ]

    # camera_id_list = ['D01', 'D02', 'D04']
    camNO, VideoTime = video_name.split('_')

    # 打开文件，准备数据
    cap = cv2.VideoCapture(os.path.join(exp_floder, video_name))
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cut_name = None
    old_time_str = None
    # 视频刚开始的时候
    video_frame = {
        'videoWriter': {},
        'cut_name': {}
    }

    # starting_flag 用来判断第一帧图像的时间是否小于预计要求的开始时间 start_time
    starting_flag = True
    starting_flag_try_times = 0
    max_try = 50

    # 上一个视频与当前视频的时间间隔
    time_interval = 0
    current_video_nums = 1

    while True:
        success, frame = cap.read()
        time_pos = camera_time_pos[camNO]
        exp_info = Exp_region_pos[camNO]

        current_time_str = getImgTime(frame, time_pos)

        # ================= 检查是否是初始帧 ===================== #
        if starting_flag:
            # isStartFrame
            # OCR识别失败，count_=0,
            # 当前帧小于开始帧，count_=-1，
            # 当前帧大于开始帧，count_=max_try，
            isStartFrame_flag, try_times, time_gap = isStartFrame(current_time_str, start_time)
            # 如果是开始帧，则之后的starting flag都置为False，不在检查帧时间
            if isStartFrame_flag:
                starting_flag = False
                # 第一个视频
                current_video_nums += 1

                # ================= 正式开始截取视频 ===================== #
                old_time_str = start_time
                cut_name = start_time

                exp_frame = getImgExp(frame, exp_info)
                # exp_frame: 2_CK: 截取的frame 画面

                for exp_name, irframe in exp_frame.items():
                    print(f"first time to cutting new video {current_time_str}-{camNO}-{exp_name}")
                    video_frame['cut_name'][exp_name] = os.path.join(
                        exp_floder, 'cut_video', exp_name,
                        f"{current_time_str}_{camNO}.avi"
                    )
                    video_frame['videoWriter'][exp_name] = cv2.VideoWriter(
                        video_frame['cut_name'][exp_name], fourcc, fps, (irframe.shape[0], irframe.shape[1]), True
                    )

            # 如果不是开始帧，则查看是否小于最大尝试次数，否则退出
            else:
                # 如果当前帧时间距离设定开始时间过远,time_gap为秒，则需要跳转到 距离开始帧3秒内
                if time_gap > 10:
                    frame_gap = (time_gap - 8) * fps
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_gap + starting_flag_try_times))
                    print(f"jumping {frame_gap}")

                starting_flag_try_times += try_times
                if starting_flag_try_times <= max_try:
                    print(f"{current_time_str} not starting frame {start_time} with {starting_flag_try_times}/{max_try}")
                    starting_flag_try_times += 1
                    continue
                else:
                    print(f"finding starting frame failed")
                    exit(987)


        #  ================ 检查该帧是否已经处理过 ================ #
        if f"{current_time_str}_{camNO}.avi" in processed_list:
            print(f"{current_time_str}_{camNO} has been processed")
            continue

        # ================= 检查当前帧对应的时间是否与之前的时间是否相同 ===================== #

        try:
            current_time_date = time.strptime(current_time_str, "%Y_%m_%d_%H_%M_%S")
            current_unix_time = time.mktime(current_time_date)

            old_time_date = time.strptime(old_time_str, "%Y_%m_%d_%H_%M_%S")
            old_unix_time = time.mktime(old_time_date)

            # 比较上一个视频的时间和当前帧的时间间隔
            time_interval = current_unix_time - old_unix_time
            if time_interval <= spt:
                # 不需要新的videoWriter
                init_videoWriter = False
            else:
                # 需要新的videoWriter以及 cut video name
                old_time_str = current_time_str
                cut_name = current_time_str
                init_videoWriter = True
        except:
            # 如果到某一秒无法识别时间了，则判断上一轮的time_interval的值。
            # 如果小于spt，则按照 fps帧/秒 计算时间
            if time_interval < spt:
                time_interval += 1.0 / fps
                init_videoWriter = False
            # 如果大于或者等于spt，则按照按照当前“错误时间str”开始新的video 片段
            # 注意,此时 old_time_str 并未更新,当遍历过程中成功识别后 time_interval 的值大于spt
            # 创建新的video片段。
            else:
                cut_name = current_time_str
                time_interval = 0
                init_videoWriter = True

        exp_frame = getImgExp(frame, exp_info)
        # exp_frame: 2_CK: 截取的frame 画面

        if init_videoWriter:
            # 一个大video截取成不同的小video
            # 如果视频数量大于 lasting_no
            if current_video_nums > lasting_no:
                break
            else:
                current_video_nums += 1

            for exp_name, irframe in exp_frame.items():
                video_frame['videoWriter'][exp_name].release()
                cv2.destroyAllWindows()

            for exp_name, irframe in exp_frame.items():
                print(f"cutting new video {camNO}-{current_time_str}-{exp_name}")
                video_frame['cut_name'][exp_name] = os.path.join(
                    exp_floder, 'cut_video', exp_name,
                    f"{current_time_str}_{camNO}.avi"
                )
                video_frame['videoWriter'][exp_name] = cv2.VideoWriter(
                    video_frame['cut_name'][exp_name], fourcc, fps, (irframe.shape[0], irframe.shape[1]), True
                )

        else:
            for exp_name, irframe in exp_frame.items():
                assert video_frame['videoWriter'][exp_name] is not None, 'please init {0}'.format(video_frame["videoWriter"][exp_name])
                video_frame['videoWriter'][exp_name].write(irframe)
