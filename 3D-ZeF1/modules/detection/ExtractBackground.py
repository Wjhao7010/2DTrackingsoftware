import cv2
import argparse
import os.path
import sys

sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')
from common.utility import *


class BackgroundExtractor():
    """
    Class implementation for background subtraction
    """

    def __init__(self, bg_file, cfg_path, media_file, video=True):
        """
        Initialize object

        Input:
            vidPath: Path to the video files
        """

        print("BackgroundExtractor initialized.")

        self.bg_file = bg_file
        self.video = video

        if self.video:
            self.vid_file = media_file

            # Load video
            cap = cv2.VideoCapture(self.vid_file)

            # Close program if video file could not be opened
            if not cap.isOpened():
                print("Could not open file: {0}".format(self.vid_file))
                sys.exit()
            self.cap = cap
        else:
            self.imgPath = media_file

            if not os.path.isdir(self.imgPath):
                print("Could not find image folder {0}".format(self.imgPath))
                sys.exit()

        if (os.path.isfile(self.bg_file)):
            print("Background file already exists. Exiting...")
            return

        self.loadSettings(cfg_path)

    def loadSettings(self, path):
        """
        Load settings from config file in the provided path.

        Config file includes information on the following, which is set in the object:
            n_median: Number of images to use for calculating the median image

        Input:
            path: String path to the folder where the settings.ini file is located
        """

        config = readConfig(path)
        c = config['BackgroundExtractor']
        self.n_median = c.getint("n_median")

    def _collectSamplesVideo(self, verbose=False):
        """
        Sample frames for the background calculation by extracting them from the defined video.

        Input:
            verbose: Whether to print information regarding which frames are used
        """

        cap = self.cap
        # Collect sample images from the video
        imgs = []
        numSamples = self.n_median
        maxFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        sampleFrames = np.linspace(0, maxFrames, numSamples + 30)[1:-1]

        for f in sampleFrames:
            if verbose:
                print("Extracting image from frame: {0}".format(int(f)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ret, frame = cap.read()
            if (ret):
                imgs.append(frame)
        self.imgs = imgs

    def _collectSamplesImg(self, verbose=False):
        """
        Sample frames for the background calculation by extracting them from the prior extracted image files.

        Input:
            verbose: Whether to print information regarding which frames are used
        """

        imgs = []
        frames = [f for f in sorted(os.listdir(self.imgPath)) if os.path.splitext(f)[-1] in [".png", ".jpg"]]
        numSamples = self.n_median
        maxFrames = len(frames)
        sampleFrames = np.linspace(0, maxFrames, numSamples, endpoint=False, dtype=np.int)

        for f in sampleFrames:
            if verbose:
                print("Extracting image from frame: {0}".format(int(f)))
            frame = cv2.imread(os.path.join(self.imgPath, frames[f]))
            if frame is not None:
                imgs.append(frame)
        self.imgs = imgs

    def collectSamples(self, verbose=False):
        """
        Collect samples to be used when calculating background image.
        The images are sampled at uniform intervals throughout the provided video or img folder
        """

        if self.video:
            self._collectSamplesVideo()
        else:
            self._collectSamplesImg()

    def createBackground(self):
        """
        Compute the median image of the collected samples.

        Output:
            bg: The computed background image
        """

        try:
            # Merge samples into background and save
            print("Generating background from {0} samples".format(len(self.imgs)))
            stack = np.stack(self.imgs)
            bg = np.median(stack, axis=0)
            return bg
        except:
            print("No images given. Run BackgroundExtractor.collectSamples before trying to create a background")
            sys.exit()


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--path", default='E:\\data\\3D_pre\\D1_T1',
                    help="Path to the directory containing the video file")
    ap.add_argument("--video_name", default='2021_10_11_21_49_59_ch01.avi', help="camerid and recording datetime")
    ap.add_argument("-v", "--video", default=True, action='store_true', help="input media is video")

    args = vars(ap.parse_args())

    if args.get("path", None) is None:
        print('No path was provided. Try again!')
        sys.exit()
    else:
        path = args["path"]

    video_name = args["video_name"]
    video = args["video"]

    video_nameT = '_'.join(video_name.split(".")[0].split("_")[: -1])

    # Configure settings for either reading images or the video file.
    video_path = os.path.join(path, 'cut_video')
    vid_file = os.path.join(video_path, video_name)

    camNO = video_name.split(".")[0].split("_")[-1]
    base_cfg_path = path
    camId = camera_id_map[camNO]
    # Prepare path
    bg_path = os.path.join(base_cfg_path, 'background')
    if not os.path.exists(bg_path):
        os.makedirs(bg_path)
    bg_file = os.path.join(bg_path, f'{video_nameT}_cam{camId}.jpg')

    bgExt = BackgroundExtractor(bg_file, path, vid_file, video=video)
    bgExt.collectSamples()
    bg = bgExt.createBackground()
    print(bg_file)
    cv2.imwrite(bg_file, bg)
