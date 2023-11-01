import pandas as pd
import os, sys
import argparse
import cv2

tank_drag_map = {
    'D1_T1': '6PPD1ppm', 'D1_T2': '6PPD500ppb', 'D1_T3': '6PPD50ppb',
    'D2_T1': '6PPD1ppm', 'D2_T2': '6PPD500ppb', 'D2_T3': '6PPD50ppb',
    'D3_T1': '6PPD1ppm', 'D3_T2': '6PPD500ppb', 'D3_T3': '6PPD50ppb',
    'D1_T4': 'RJ', 'D1_T5': 'CK', 'D4_T1': '4Hydroxy500ppb',
    'D2_T4': 'RJ', 'D2_T5': 'CK', 'D5_T1': '4Hydroxy500ppb',
    'D3_T4': 'RJ', 'D3_T5': 'CK', 'D6_T1': '4Hydroxy500ppb',
    'D4_T2': '4Hydroxy50ppb', 'D4_T4': '6PPDQ500ppb', 'D4_T5': '6PPDQ50ppb',
    'D5_T2': '4Hydroxy50ppb', 'D5_T4': '6PPDQ500ppb', 'D5_T5': '6PPDQ50ppb',
    'D6_T2': '4Hydroxy50ppb', 'D6_T4': '6PPDQ500ppb', 'D6_T5': '6PPDQ50ppb',
    'D7_T1': '4Hydroxy1ppm', 'D4_T3': '6PPDQ1ppm',
    'D7_T2': '4Hydroxy1ppm', 'D7_T5': '6PPDQ1ppm',
    'D7_T4': '4Hydroxy1ppm', 'D8_T2': '6PPDQ1ppm',
    'D8_T4': '6PPDQ1ppm'

}

class ChcekData(object):
    def __init__(self, video_foldername):

        self.load_settings()

    def load_settings(self):
        self.detection_missing_num = 500
        self.track_missing_num = 50
        self.final_missing_num = 500

    def get_frame_info(self, video_folder, video_name):
        video_data = cv2.VideoCapture(os.path.join(video_folder, video_name))
        video_info = {
            'FrameNum': video_data.get(cv2.CAP_PROP_FRAME_COUNT),
            'Width': video_data.get(cv2.CAP_PROP_FRAME_WIDTH),
            'Height': video_data.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'FPS': video_data.get(cv2.CAP_PROP_FPS)
        }
        return video_info

    def check_bgprocess(self, DayTank, DRegion_rag_Name):
        print(f"checking {DayTank} with {DRegion_rag_Name}")
        video_folder = os.path.join(root_path, DayTank, video_foldername)
        video_list = os.listdir(video_folder)
        video_num = len(video_list)

        error_log = os.path.join(root_path, DayTank, detector, "detector_error.txt")
        error_f = open(error_log, "w+")
        error_f.write(f"=========================++++++++++++++++++++++++=========================\n")
        error_f.write(f"========================= In the detection stage =========================\n")
        error_f.write(f"=========================++++++++++++++++++++++++=========================\n")

        detector_folder = detector_template_folder.replace('DayTank', DayTank).replace('Drag_Name', DRegion_rag_Name)
        if not os.path.exists(detector_folder):
            print("=" * 25)
            print(f"folder:={detector_folder}= is not exist!")
            error_f.write(f"folder:={detector_folder}= is not exist! \n")
            print("=" * 25)
            return

        detect_lists = os.listdir(detector_folder)

        print("============================ Ignore Video ===============================")
        error_f.write("============================ Ignore Video =============================== \n")
        if len(detect_lists) != video_num * 2 / 3:
            print(f"looks like some videos aren't processed!")
            error_f.write(
                f"looks like some videos are not processed by detector! "
                f"[{len(detect_lists)}/ {video_num * 2 / 3}] in {DRegion_rag_Name} \n"
            )
        print("=" * 25)

        print("============================ Ignore Detection frames ===============================")
        error_f.write("============================ Ignore Detection frames =============================== \n")
        for idet_filename in detect_lists:
            print(f"checking {DayTank} with {DRegion_rag_Name} in detection {idet_filename} file")

            idet_file = os.path.join(detector_folder, idet_filename)
            data = pd.read_csv(idet_file)
            video_info = self.get_frame_info(video_folder, idet_filename.replace(".csv", ".avi"))
            if video_info['FrameNum'] - data.shape[0] >= self.detection_missing_num:
                print("=" * 25)
                print(f"detector doesn't work in some videos!")
                print(f"{idet_file}")
                error_f.write(
                    f"{idet_file}: "
                    f"[{data.shape[0]}/{video_info['FrameNum']}] \n"
                )
                print("=" * 25)
        return

    def check_tracker(self, DayTank, DRegion_rag_Name):
        print(f"checking {DayTank} with {DRegion_rag_Name}")

        error_log = os.path.join(root_path, DayTank, detector, "tracker_error.txt")
        error_f = open(error_log, "w+")
        error_f.write(f"=========================++++++++++++++++++++++++=========================\n")
        error_f.write(f"========================= In the tracker stage =========================\n")
        error_f.write(f"=========================++++++++++++++++++++++++=========================\n")

        tracker_folder = tracker_template_folder.replace('DayTank', DayTank).replace('Drag_Name', DRegion_rag_Name)
        detector_folder = detector_template_folder.replace('DayTank', DayTank).replace('Drag_Name', DRegion_rag_Name)

        if not os.path.exists(tracker_folder):
            print("=" * 25)
            print(f"folder:={tracker_folder}= is not exist!")
            error_f.write(f"folder:={tracker_folder}= is not exist! \n")
            print("=" * 25)
            return

        track_lists = os.listdir(tracker_folder)
        det_lists = os.listdir(detector_folder)

        print("============================ Ignore Tracker files ===============================")
        error_f.write("============================ Ignore Tracker files =============================== \n")
        if len(track_lists) != len(det_lists):
            print("=" * 25)
            print(f"looks like some files are missing!")
            error_f.write(
                f"looks like some detection file are not processed or ignored by tracker! "
                f"[{len(track_lists)}/ {len(det_lists)}] \n"
            )
            print("=" * 25)

        print("============================ Ignore Tracker frames ===============================")
        error_f.write("============================ Ignore Tracker frames =============================== \n")
        for idet_filename, itrk_filename in zip(det_lists, track_lists):
            print(f"checking {DayTank} with {DRegion_rag_Name} in tracker file")

            idet_file = os.path.join(detector_folder, idet_filename)
            itrk_file = os.path.join(tracker_folder, itrk_filename)
            detect_data = pd.read_csv(idet_file)
            track_data = pd.read_csv(itrk_file)

            if detect_data.shape[0] - track_data.shape[0] >= self.track_missing_num:
                print("=" * 25)
                print(f"tracker doesn't work in some detection frames!")
                error_f.write(
                    f"{idet_file}: "
                    f"[{track_data.shape[0]}/{detect_data.shape[0]}] \n"
                )
                print("=" * 25)
        return

    def check_finaltrack(self, DayTank, DRegion_rag_Name):
        print(f"checking {DayTank} with {DRegion_rag_Name}")
        video_folder = os.path.join(root_path, DayTank, video_foldername)

        error_log = os.path.join(root_path, DayTank, detector, "trajectory_error.txt")
        error_f = open(error_log, "w+")
        error_f.write(f"=========================++++++++++++++++++++++++=========================\n")
        error_f.write(f"========================= In the trajectory stage =========================\n")
        error_f.write(f"=========================++++++++++++++++++++++++=========================\n")

        tracker_folder = tracker_template_folder.replace('DayTank', DayTank).replace('Drag_Name', DRegion_rag_Name)
        traectory_folder = trajectory_template_folder.replace('DayTank', DayTank).replace('Drag_Name', DRegion_rag_Name)

        if not os.path.exists(tracker_folder):
            print("=" * 25)
            print(f"folder:={tracker_folder}= is not exist!")
            error_f.write(f"folder:={tracker_folder}= is not exist! \n")
            print("=" * 25)
            return

        track_lists = os.listdir(tracker_folder)
        traectory_lists = os.listdir(traectory_folder)

        print("============================ Ignore Tracker frames ===============================")
        error_f.write("============================ Ignore Tracker frames =============================== \n")
        if len(track_lists) != len(traectory_lists):
            print("=" * 25)
            print(f"looks like some integrated trajectory files are missing!")
            error_f.write(
                f"looks like some integrated trajectory are not processed or ignored by interpolator! "
                f"[{len(track_lists)}/ {len(traectory_lists)}] \n"
            )
            print("=" * 25)

        for itrk_filename, itra_filename in zip(track_lists, traectory_lists):
            print(f"checking {DayTank} with {DRegion_rag_Name} in trajectory file")

            itra_file = os.path.join(traectory_folder, itra_filename)

            final_data = pd.read_csv(itra_file)

            video_info = self.get_frame_info(video_folder, itra_file.replace(".csv", ".avi"))

            if abs(final_data.shape[0] - video_info['FrameNum']) >= self.final_missing_num:
                print("=" * 25)
                print(f"final process doesn't work in some tracker files!")
                error_f.write(
                    f"{itra_file}: "
                    f"[{final_data.shape[0]}/{video_info['FrameNo']}] \n"
                )
                print("=" * 25)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-rp", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("-vid", "--video_foldername", default="cut_video", type=str)
    ap.add_argument("-deter", "--detector", default="bg_processed", type=str)
    ap.add_argument("-trker", "--tracker", default="sortTracker", type=str)

    args = vars(ap.parse_args())

    root_path = args['root_path']
    detector = args['detector']
    tracker = args['tracker']
    video_foldername = args['video_foldername']

    detector_template_folder = os.path.join(root_path, 'DayTank', detector, 'Drag_Name')
    tracker_template_folder = os.path.join(root_path, 'DayTank', tracker, 'Drag_Name')
    trajectory_template_folder = os.path.join(root_path, 'DayTank', 'gapfill', 'Drag_Name')

    analyzer = ChcekData(video_foldername)
    for iDayTank, drag_name in tank_drag_map.items():
        for iregion in ['1_', '2_', '3_', '4_']:
            analyzer.check_bgprocess(iDayTank, iregion+drag_name)
            # analyzer.check_tracker(iDayTank, iregion+drag_name)
            # analyzer.check_finaltrack(iDayTank, iregion+drag_name)