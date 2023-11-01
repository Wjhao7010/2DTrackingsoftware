import nsq
import os
from functools import partial

from tornado import gen


@gen.coroutine
def write_message(topic, data, writer):  # data=video_name_without_extension
    response = writer.pub(topic, data.encode('utf8'))
    if isinstance(response, nsq.Error):
        print("Error with Message: {}:{}".format(data, response))
    else:
        print("Published Message: ", data)


# 存储了每一段视频的背景图像 threading_extractBG.py
def ExtractBackground(filename):
    cmd_background = (
        'python E:/Project/PyQt5/ZeF/3D-ZeF1/modules/detection/ExtractBackground.py --path E:/data/3D_pre/D1_T1 --video_name ')
    cmd_background += filename
    os.system(cmd_background)


def meanBackground():
    cmd_mean_background = (
        'python E:/Project/PyQt5/ZeF/3D-ZeF1/modules/detection/meanBackground.py ')
    os.system(cmd_mean_background)


# 定义消息处理函数
def detect_message(message, writer):
    # 解析消息内容
    message.enable_async()
    os.chdir('E:/Project/PyQt5/ZeF/3D-ZeF1/')  # 修改工程路径
    video_info = message.body.decode('utf-8')
    # time.sleep(60)
    print("接收到的信息：", video_info)
    video_name = video_info.split("/")[-1]
    video_name_without_extension = video_name.split('.')[0]
    area = video_name[-8:-4]
    print("area:", area)
    print("video_name:", video_name)

    topic = "back_names"

    write_message(topic, video_name_without_extension, writer)
    message.finish()

    # if area == 'ch02':  # ch01=3 ch03=2 ch02=1  ch02对应的是顶部视图
    #     # 对每段视频执行目标检测
    #     cmd1 = (
    #         'python modules/yolo5/detect.py --weights E:/Project/PyQt5/ZeF/3D-ZeF/best.pt --config_path E:/data/3D_pre/D1_T1 --region_name 1_6PPD1ppm --device 1 --project E:/data/3D_pre/D1_T1/realprocessed/1_6PPD1ppm --img 640 --source ')
    #     cmd1 += video_info
    #     os.system(cmd1)
    #     cmd2 = (
    #         'python modules/yolo5/detect.py --weights E:/Project/PyQt5/ZeF/3D-ZeF/best.pt --config_path E:/data/3D_pre/D1_T1 --region_name 2_6PPD1ppm --device 1 --project E:/data/3D_pre/D1_T1/realprocessed/2_6PPD1ppm --img 640 --source ')
    #     cmd2 += video_info
    #     os.system(cmd2)
    #     cmd3 = (
    #         'python modules/yolo5/detect.py --weights E:/Project/PyQt5/ZeF/3D-ZeF/best.pt --config_path E:/data/3D_pre/D1_T1 --region_name 3_6PPD1ppm --device 1 --project E:/data/3D_pre/D1_T1/realprocessed/3_6PPD1ppm --img 640 --source ')
    #     cmd3 += video_info
    #     os.system(cmd3)
    #     cmd4 = (
    #         'python modules/yolo5/detect.py --weights E:/Project/PyQt5/ZeF/3D-ZeF/best.pt --config_path E:/data/3D_pre/D1_T1 --region_name 4_6PPD1ppm --device 1 --project E:/data/3D_pre/D1_T1/realprocessed/4_6PPD1ppm --img 640 --source ')
    #     cmd4 += video_info
    #     os.system(cmd4)
    #     # 存储了每一段视频的背景图像 threading_extractBG.py
    #     ExtractBackground(video_name)
    #
    # elif area == 'ch01':
    #     # 对每段视频执行目标检测
    #     cmd2 = (
    #         'python modules/yolo5/detect_side.py --weights E:/Project/PyQt5/ZeF/3D-ZeF/best.pt --config_path E:/data/3D_pre/D1_T1 --region_name 2_6PPD1ppm --device 1 --project E:/data/3D_pre/D1_T1/realprocessed/2_6PPD1ppm --img 640 --source ')
    #     cmd2 += video_info
    #     os.system(cmd2)
    #     cmd4 = (
    #         'python modules/yolo5/detect_side.py --weights E:/Project/PyQt5/ZeF/3D-ZeF/best.pt --config_path E:/data/3D_pre/D1_T1 --region_name 4_6PPD1ppm --device 1 --project E:/data/3D_pre/D1_T1/realprocessed/4_6PPD1ppm --img 640 --source ')
    #     cmd4 += video_info
    #     os.system(cmd4)
    #     # 存储了每一段视频的背景图像 threading_extractBG.py
    #     ExtractBackground(video_name)
    #
    # elif area == 'ch03':
    #     # 对每段视频执行目标检测
    #     cmd1 = (
    #         'python modules/yolo5/detect_side.py --weights E:/Project/PyQt5/ZeF/3D-ZeF/best.pt --config_path E:/data/3D_pre/D1_T1 --region_name 1_6PPD1ppm --device 1 --project E:/data/3D_pre/D1_T1/realprocessed/1_6PPD1ppm --img 640 --source ')
    #     cmd1 += video_info
    #     os.system(cmd1)
    #     cmd3 = (
    #         'python modules/yolo5/detect_side.py --weights E:/Project/PyQt5/ZeF/3D-ZeF/best.pt --config_path E:/data/3D_pre/D1_T1 --region_name 3_6PPD1ppm --device 1 --project E:/data/3D_pre/D1_T1/realprocessed/3_6PPD1ppm --img 640 --source ')
    #     cmd3 += video_info
    #     os.system(cmd3)
    #     # 存储了每一段视频的背景图像 threading_extractBG.py
    #     ExtractBackground(video_name)
    #
    # # 存储了每一段视频的背景图像 threading_extractBG.py
    # ExtractBackground(video_name)
    # # 生成背景图像的均值 meanBackground.py
    # meanBackground()
    #
    # if area == 'ch02':  # ch01=3 ch03=2 ch02=1  ch02对应的是顶部视图
    #     cmd_bg_detect1 = (
    #         'python modules/detection/BgDetector.py --path E:/data/3D_pre/D1_T1 --region_name 1_6PPD1ppm --video_name ')
    #     cmd_bg_detect1 += video_name
    #     os.system(cmd_bg_detect1)
    #     cmd_bg_detect2 = (
    #         'python modules/detection/BgDetector.py --path E:/data/3D_pre/D1_T1 --region_name 2_6PPD1ppm --video_name ')
    #     cmd_bg_detect2 += video_name
    #     os.system(cmd_bg_detect2)
    #     cmd_bg_detect3 = (
    #         'python modules/detection/BgDetector.py --path E:/data/3D_pre/D1_T1 --region_name 3_6PPD1ppm --video_name ')
    #     cmd_bg_detect3 += video_name
    #     os.system(cmd_bg_detect3)
    #     cmd_bg_detect4 = (
    #         'python modules/detection/BgDetector.py --path E:/data/3D_pre/D1_T1 --region_name 4_6PPD1ppm --video_name ')
    #     cmd_bg_detect4 += video_name
    #     os.system(cmd_bg_detect4)
    # elif area == 'ch01':
    #     cmd_bg_detect2 = (
    #         'python modules/detection/BgDetector.py --path E:/data/3D_pre/D1_T1 --region_name 2_6PPD1ppm --video_name ')
    #     cmd_bg_detect2 += video_name
    #     os.system(cmd_bg_detect2)
    #     cmd_bg_detect4 = (
    #         'python modules/detection/BgDetector.py --path E:/data/3D_pre/D1_T1 --region_name 4_6PPD1ppm --video_name ')
    #     cmd_bg_detect4 += video_name
    #     os.system(cmd_bg_detect4)
    #
    # elif area == 'ch03':
    #     cmd_bg_detect1 = (
    #         'python modules/detection/BgDetector.py --path E:/data/3D_pre/D1_T1 --region_name 1_6PPD1ppm --video_name ')
    #     cmd_bg_detect1 += video_name
    #     os.system(cmd_bg_detect1)
    #     cmd_bg_detect3 = (
    #         'python modules/detection/BgDetector.py --path E:/data/3D_pre/D1_T1 --region_name 3_6PPD1ppm --video_name ')
    #     cmd_bg_detect3 += video_name
    #     os.system(cmd_bg_detect3)
    #
    # # 生成对应的轨迹信息 SortTracker.py
    # if area == 'ch02':  # ch02=1.2.3.4  ch02对应的是顶部视图
    #     cmd_tracker1 = (
    #         'python E:/Project/PyQt5/ZeF/3D-ZeF1/modules/tracking/sortTracker.py --root_path E:/data/3D_pre --DayTank D1_T1 --RegionName 1_6PPD1ppm --detection_filename ')
    #     detection_filename = video_name_without_extension + '.csv'
    #     cmd_tracker1 += detection_filename
    #     os.system(cmd_tracker1)
    #
    #     cmd_tracker2 = (
    #         'python E:/Project/PyQt5/ZeF/3D-ZeF1/modules/tracking/sortTracker.py --root_path E:/data/3D_pre --DayTank D1_T1 --RegionName 2_6PPD1ppm --detection_filename ')
    #     detection_filename = video_name_without_extension + '.csv'
    #     cmd_tracker2 += detection_filename
    #     os.system(cmd_tracker2)
    #
    #     cmd_tracker3 = (
    #         'python E:/Project/PyQt5/ZeF/3D-ZeF1/modules/tracking/sortTracker.py --root_path E:/data/3D_pre --DayTank D1_T1 --RegionName 3_6PPD1ppm --detection_filename ')
    #     detection_filename = video_name_without_extension + '.csv'
    #     cmd_tracker3 += detection_filename
    #     os.system(cmd_tracker3)
    #
    #     cmd_tracker4 = (
    #         'python E:/Project/PyQt5/ZeF/3D-ZeF1/modules/tracking/sortTracker.py --root_path E:/data/3D_pre --DayTank D1_T1 --RegionName 4_6PPD1ppm --detection_filename ')
    #     detection_filename = video_name_without_extension + '.csv'
    #     cmd_tracker4 += detection_filename
    #     os.system(cmd_tracker4)
    # elif area == 'ch01':  # ch01=2.4
    #     cmd_tracker2 = (
    #         'python E:/Project/PyQt5/ZeF/3D-ZeF1/modules/tracking/sortTracker.py --root_path E:/data/3D_pre --DayTank D1_T1 --RegionName 2_6PPD1ppm --detection_filename ')
    #     detection_filename = video_name_without_extension + '.csv'
    #     cmd_tracker2 += detection_filename
    #     os.system(cmd_tracker2)
    #
    #     cmd_tracker4 = (
    #         'python E:/Project/PyQt5/ZeF/3D-ZeF1/modules/tracking/sortTracker.py --root_path E:/data/3D_pre --DayTank D1_T1 --RegionName 4_6PPD1ppm --detection_filename ')
    #     detection_filename = video_name_without_extension + '.csv'
    #     cmd_tracker4 += detection_filename
    #     os.system(cmd_tracker4)
    #
    # elif area == 'ch03':  # ch03=1.3
    #     cmd_tracker1 = (
    #         'python E:/Project/PyQt5/ZeF/3D-ZeF1/modules/tracking/sortTracker.py --root_path E:/data/3D_pre --DayTank D1_T1 --RegionName 1_6PPD1ppm --detection_filename ')
    #     detection_filename = video_name_without_extension + '.csv'
    #     cmd_tracker1 += detection_filename
    #     os.system(cmd_tracker1)
    #
    #     cmd_tracker3 = (
    #         'python E:/Project/PyQt5/ZeF/3D-ZeF1/modules/tracking/sortTracker.py --root_path E:/data/3D_pre --DayTank D1_T1 --RegionName 3_6PPD1ppm --detection_filename ')
    #     detection_filename = video_name_without_extension + '.csv'
    #     cmd_tracker3 += detection_filename
    #     os.system(cmd_tracker3)

    return True


# class HeartbeatReader(nsq.Reader):
#     def __init__(self):
#         super().__init__()
#         self.last_heartbeat_time = time.time()
#
#     def process_response(self, command, data):
#         if command == nsq.FRAME_TYPE_HEARTBEAT:
#             self.last_heartbeat_time = time.time()
#         else:
#             super().process_response(command, data)
#
#     def is_connection_alive(self):
#         return time.time() - self.last_heartbeat_time < 5  # 5秒内收到过心跳响应


# nsq连接部分
if __name__ == '__main__':
    writer = nsq.Writer(['127.0.0.1:4150'])
    handler = partial(detect_message, writer=writer)
    reader = nsq.Reader(message_handler=handler,
                        nsqd_tcp_addresses=['127.0.0.1:4150'],
                        topic='test1',
                        channel='work_group_t1')

    nsq.run()
