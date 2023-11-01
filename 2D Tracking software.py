import sys
import os
import nsq
import tornado.ioloop
import functools
import shutil
import asyncio
from functools import partial
from multiprocessing import Process
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog


def setup_nsq_reader():
    print("wdwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
    # nsq.Reader(
    #     message_handler=fish.export_metrics,
    #     nsqd_tcp_addresses=['127.0.0.1:4150'],
    #     topic='back_names',
    #     channel='work_group_t1')
    r = nsq.Reader(message_handler=fish.export_metrics,
                   lookupd_http_addresses=['http://127.0.0.1:4161'],
                   topic='test1', channel='work_group_t1', lookupd_poll_interval=15)
    print("wdwwwww12312312312312321wwwwwwwwwwwwwwwww")


class TwoDTrack(QWidget):
    def __init__(self):
        super().__init__()
        self.video_name_without_extension = None
        self.writer = None
        # self.callback = None
        self.nsq_reader = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('2DTrack')

        # 创建鱼类运动视频播放窗口
        video_window = QLabel('鱼类运动视频播放窗口')
        video_window.setStyleSheet('background-color: lightyellow;')
        video_window.setMinimumSize(800, 600)

        # 创建鱼类运动指标展示窗口
        metrics_window = QLabel('鱼类运动指标展示窗口')
        metrics_window.setStyleSheet('background-color: lightblue;')
        metrics_window.setMinimumSize(500, 400)

        # 创建上传视频并分析按钮
        upload_button = QPushButton('上传视频并分析')
        upload_button.clicked.connect(self.upload_video)

        # 创建数据指标文件导出按钮
        export_button = QPushButton('数据指标文件导出')
        export_button.clicked.connect(setup_nsq_reader)

        # 创建视频保存按钮
        save_button = QPushButton('视频保存')
        save_button.clicked.connect(self.save_video)

        # 创建按钮布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(export_button)
        button_layout.addWidget(save_button)

        # 创建主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(video_window, 1)
        main_layout.addWidget(metrics_window, 2)

        # 创建整体布局
        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addWidget(upload_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def pub_message(self, video_path):
        def finish_pub(conn, data):
            print("上传完成")

        self.writer.pub('test1', video_path.encode('utf-8'), finish_pub)
        print("pub success")
        # self.callback.stop()
        # tornado.ioloop.IOLoop.current().stop()
        # self.writer.pub('test', os.path.basename(video_path).encode('utf-8'), finish_pub)

    def upload_video(self):
        # self.enable_async()
        # 上传视频并进行分析的功能实现
        button_name = self.sender().text()
        print("点击的按钮名称：", button_name)
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, '上传视频', '', 'Video Files (*.mp4 *.avi)')
        video_name = os.path.basename(video_path)
        print("上传的视频名称：", video_name)

        self.writer = nsq.Writer(['127.0.0.1:4150'])  # todo 每次点击只上传一次内容，不是循环发送

        # if self.writer is not None:
        #     await self.upload_file(video_path)
        self.pub_message(video_path)
        # tornado.ioloop.PeriodicCallback(self.pub_message, 5000).start()
        # self.callback = tornado.ioloop.PeriodicCallback(
        #     functools.partial(self.pub_message, video_path), 1000)
        # self.callback.start()
        # tornado.ioloop.IOLoop.current().start()
        # self.callback.stop()
        # tornado.ioloop.IOLoop.current().stop()
        # print("yes")
        # nsq.run()

    def export_metrics(self, message):
        print("dwaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        message.enable_async()
        # if message.decode('utf8') == "OK":
        #     print("success!")
        video_name_without_extension = message.body.decode('utf-8')
        print(video_name_without_extension)
        button_name = self.sender().text()
        print("点击的按钮名称：", button_name)
        file_dialog = QFileDialog()  # 用于显示文件对话框
        # 显示保存文件的对话框，并返回用户选择的文件路径和名称
        metrics_path, _ = file_dialog.getSaveFileName(self, '导出数据指标文件', '', 'CSV Files (*.csv)')
        # csv_file_path = os.path.join(metrics_path, f"{video_name_without_extension}.csv")
        file_dir = r'E:\data\3D_pre\D1_T1\sortTracker\1_6PPD1ppm'
        file_name = video_name_without_extension + '.csv'
        # file_name = '2021_10_11_21_49_59_ch03.csv'
        # csv_file = r'E:/data/3D_pre/D1_T1/sortTracker/1_6PPD1ppm/2021_10_11_21_49_59_ch03.csv'
        csv_file = os.path.join(file_dir, file_name)
        # if metrics_path:
        #     csv_file = os.path.join(file_dir, file_name)
        # csv_file = ['E:/data/3D_pre/D1_T1/sortTracker/1_6PPD1ppm/2021_10_11_21_49_59_ch03.csv']
        shutil.copyfile(csv_file, metrics_path)
        print('CSV 文件导出完成:', metrics_path)

    def save_video(self):
        # 视频保存的功能实现
        button_name = self.sender().text()
        print("点击的按钮名称：", button_name)
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(self, '保存视频', '', 'Video Files (*.mp4)')
        # nsq_url = '127.0.0.1:4150'  # NSQ的URL

        # writer = nsq.Writer([nsq_url])
        # tornado.ioloop.PeriodicCallback(
        #     functools.partial(self.pub_message, writer),
        #     # pub_message,
        #     1000).start()
        # nsq.run()

        # 在此处添加视频保存的代码逻辑


def nsq_process():
    pid = os.getpid()  # 获取当前进程的进程ID
    print(f'{pid}开始执行...')
    nsq.run()
    print(f'{pid}执行完成')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    fish = TwoDTrack()
    fish.show()
    # fish.setup_nsq_reader()
    p = Process(target=nsq_process)
    p.daemon = True  # 设置为守护进程
    p.start()
    sys.exit(app.exec_())

