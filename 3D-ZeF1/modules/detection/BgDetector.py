import cv2, sys
import argparse
import shutil
import matplotlib.pyplot as plt
import skimage.morphology
### Module imports ###
import sys

sys.path.append(".")
sys.path.append('../../')
from common.utility import *


class BgDetector:
    """
    Class implementation for detecting fish keypoints.
    Utilizes an extracted background image (From ExtractBackground.py)
    Image is thresholded using either Entropy split (front) or Intermodal split (Top)
    """

    def __init__(self, camNO, base_cfg_path, bg_file, region_name, video_settings_file):
        """
        Initialize object
        
        Input:
            camId: Camera view of the video to be analysed. 1 = Top, 2 = Front
            dataPath: Path to the video files
        """

        self.timer = False
        self.region_name = region_name
        self.camNO = camNO
        self.camId = camera_id_map[camNO]
        self.onlyHeads = (self.camId == 1)
        self.loadSettings(base_cfg_path, video_settings_file)

        # Load static background and downsample it
        bg = cv2.imread(bg_file)
        self.bg = bg[::self.downsample, ::self.downsample]

        # Frame at different stages
        self.frame = None  # Original frame
        self.dif = None  # After background subtraction
        self.blur = None  # After applying blur
        self.thresh = None  # After thresholding (i.e. binary)
        self.thin = None  # After skeletonization

    def loadSettings(self, path, video_settings_file):
        """
        Load settings from config file in the provided path.
        
        Config file includes information on the following, which is set in the object:
            downsample_factor: How much the images should be downsampled during processing
            blur_size: Size of the the median blur filter
            min_blob_size: MInimum size of the detected blobs
            
        Input:
            path: String path to the folder where the settings.ini file is located
        """

        config = readConfig(path, configFile=video_settings_file)
        c = config['Detector']
        self.n_fish = c.getint('n_fish')
        self.detectorType = c.get('cam{}_type'.format(self.camId))
        self.downsample = c.getint('downsample_factor')  # How much to downsample the image by
        self.blurSize = c.getint('blur_size')  # Size of median blur
        # self.blurSize = 1  # Size of median blur
        self.minBlobSize = c.getint('min_blob_size')  # used to filter BLOBs in the "blob" function
        self.maxBlobSize = c.getint('max_blob_size')  # used to filter BLOBs in the "blob" function
        self.minPatchArea = c.getint("min_patch_area")  # used to filter BLOBs in calceig
        self.minSkeletonSize = c.getint(
            "min_skeleton_length")  # minimum length between two keypoint in the skeleton (cam1), for the distance to be considered when finding best keypoint
        self.maxSkeletonSize = c.getint(
            "max_skeleton_length")  # minimum length between two keypoint in the skeleton (cam1), for the distance to be considered when finding best keypoint
        self.winSize = c.getint("window_size")  # Size of window around keypoint in calcEig
        self.nms_thresh = c.getfloat(
            "nms_threshold")  # Threshold for how large an overlap there can be before applying NMS
        self.mean_threshold = c.getfloat(
            "mean_threshold")  # Threshold for how large an overlap there can be before applying NMS
        if self.mean_threshold is None:
            self.mean_threshold = 3
        if self.camId == 1:
            self.max_frame = c.getint("cam1_maxframe")
            self.min_frame = c.getint("cam1_minframe")
        elif self.camId == 2:
            self.max_frame = c.getint("cam2_maxframe")
            self.min_frame = c.getint("cam2_minframe")
        else:
            self.max_frame = c.getint("cam3_maxframe")
            self.min_frame = c.getint("cam3_minframe")

        region_info = load_EXP_region_pos_setting(path, self.camNO)
        if self.region_name not in region_info.keys():
            # print(f"{self.region_name} not in") # don't need this region name,so #
            exit(0)
        tl_x, tl_y, br_x, br_y = load_EXP_region_pos_setting(path, self.camNO)[self.region_name]
        tl, br = np.array([tl_x, tl_y]), np.array([br_x, br_y])
        self.tl = tl // self.downsample
        self.br = br // self.downsample
        print(self.tl, self.br)

    def detect(self, frame, bboxes):
        """
        Performs the detection step
        
        Input:
            frame: The current frame
            camId: Which camera view the fram eis from (1 = Top, 2 = Front)
        
        Output:
            filtered: The detected keypoints after filtering. List of cv2.KeyPoints
            bbs: The rotated bounding boxes of the filtered keypoints. List of dicts, with the following keys, containing floats:
                tl_x: Top left x coordinate of rotated bounding box
                tl_y: Top left y coordinate of rotated bounding box
                c_x: x center coordiante of origianl bounding box
                c_y: y center coordiante of original bounding box
                w: Width ofthe rotated bounding box
                h: Height of the rotated bounding box
                theta: The angle of the rotated bounding box
        """

        ## Downsample video
        _start = time.time()
        self.frame = frame[::self.downsample, ::self.downsample]
        _end = time.time()
        if (self.timer):
            print("Downsample time: {0}".format(_end - _start))

        if 'D4_T5' in path and 'ch16' in video_name:
            self.frame = self.imgBrightness(self.frame, 1.5, 3)
            self.bg = self.imgBrightness(self.bg, 2.5, 3)
        ## Subtract background
        self.diff = self.bgSubtract(self.frame)
        self.diff = self.applyROIMat(self.diff)
        if verbose:
            plt.imshow(self.diff)
            plt.show()

        ## Blur image
        _start = time.time()
        # self.blur = cv2.medianBlur(self.diff, self.blurSize)
        self.blur = self.diff
        if verbose:
            plt.imshow(self.blur)
            plt.show()
        _end = time.time()

        # ## remove shadow
        # self.noshodow = self.removeShade(self.blur)
        # if verbose:
        #     plt.imshow(self.noshodow)
        #     plt.show()
        # if (self.timer):
        #     print("Blur time: {0}".format(_end - _start))

        ## Threshold image. Method is dependent on camera view
        if (self.camId == 1):
            # Threshold image using intermodes algorithm
            th_inter = self.intermodesSplit(self.blur)
            th_entropy = self.entropySplit(self.blur)
            th = (th_inter + th_entropy) / self.mean_threshold
            if np.isnan(th):
                th = th_entropy
            self.thresh = self.blur > th
            if verbose:
                plt.imshow(self.thresh)
                plt.show()
        else:
            # Threshold image using max entropy
            th = self.entropySplit(self.blur)
            self.thresh = self.blur > th
        # Remove everything outside of the ROI

        if verbose:
            plt.imshow(self.thresh)
            plt.show()
        self.bboxes = applyROIBBs(bboxes, self.tl, self.br)

        # Find keypoints and boundingbox of the objects based on the detector method
        if (self.detectorType == 'blob'):
            filtered, bbs = self.blob()
        elif (self.detectorType == 'skeleton'):
            filtered, bbs = self.skeleton()
        else:
            print(
                "Error. Unknown detector type. Check settings.ini to see whether detector type is [blob] or [skeleton].")
            sys.exit()

        return filtered, bbs

    def imgBrightness(self, img1, c, b):
        rows, cols, channels = img1.shape
        blank = np.zeros([rows, cols, channels], img1.dtype)
        rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
        return rst

    def applyROIMat(self, mat):
        """
        Sets everything outside of the ROI to 0

        Input:
            Mat: Input image

        Output: 
            mat: ROI output image
        """

        if mat.ndim == 2:
            mat[:, :self.tl[0]] = 0
            mat[:, self.br[0] + 1:] = 0

            mat[:self.tl[1]] = 0
            mat[self.br[1] + 1:] = 0

        elif mat.ndim == 3:
            mat[:, :self.tl[0], :] = 0
            mat[:, self.br[0] + 1:, :] = 0

            mat[:self.tl[1], :] = 0
            mat[self.br[1] + 1:, :] = 0

        return mat

    def blob(self):
        """
        Detection method that finds the BLOBs consisting of the most pixels
        
        Input:
        
        Output:
            filtered: The detected keypoints after filtering. List of cv2.KeyPoints
            bbs: The rotated bounding boxes of the filtered keypoints. List of dicts, with the following keys, containing floats:
                tl_x: Top left x coordinate of rotated bounding box
                tl_y: Top left y coordinate of rotated bounding box
                c_x: x center coordiante of origianl bounding box
                c_y: y center coordiante of original bounding box
                w: Width ofthe rotated bounding box
                h: Height of the rotated bounding box
                theta: The angle of the rotated bounding box
        """

        ## Find BLOBs
        img = self.thresh.astype(np.uint8) * 255
        ret, self.labels = cv2.connectedComponents(img)

        ## Sort BLOBs based on their pixel count. Assuming the background (label = 0) is the largest
        unq_labels, counts = np.unique(self.labels, return_counts=True)
        unq_labels = unq_labels[1:]
        counts = counts[1:]

        sorted_indecies = np.argsort(counts)[::-1]
        unq_labels = unq_labels[sorted_indecies]
        counts = counts[sorted_indecies]
        counts = counts[counts > self.minBlobSize]  # Ignore tiny BLOBs
        # counts = counts[counts < self.maxBlobSize]  # Ignore tiny BLOBs

        # Find the largest BLOBs 
        numBlobs = self.n_fish * 2
        if len(counts) < numBlobs:
            numBlobs = len(counts)

        unq_labels = unq_labels[:numBlobs]

        ## Find rotated bounding boxes of the detected keypoints
        bbs = self.findRotatedBB(unq_labels)

        ## Keypoints are determined by the center-point
        filtered = []
        for b in bbs:
            filtered.append(cv2.KeyPoint(x=b["c_x"], y=b["c_y"], _size=1))

        return filtered, bbs

    def skeleton(self):
        """
        Detection method that find keypoints in the skeleton of the BLOBs
        
        Input:
        
        Output:
            filtered: The detected keypoints after filtering. List of cv2.KeyPoints
            bbs: The rotated bounding boxes of the filtered keypoints. List of dicts, with the following keys, containing floats:
                tl_x: Top left x coordinate of rotated bounding box
                tl_y: Top left y coordinate of rotated bounding box
                c_x: x center coordiante of origianl bounding box
                c_y: y center coordiante of original bounding box
                w: Width ofthe rotated bounding box
                h: Height of the rotated bounding box
                theta: The angle of the rotated bounding box
        """

        ## Fill holdes in the thresholded BLOBs
        self.thresh = self.fillHoles(self.thresh)
        if verbose:
            plt.imshow(self.thresh)
            plt.show()

        ## Extract skeletons of BLOBs
        self.thin = skimage.morphology.skeletonize(self.thresh)
        if verbose:
            plt.imshow(self.thin)
            plt.show()
        ## Detect potential keypoints
        detections = self.interestPoints(findJunctions=True)

        filtered = []

        for label in detections:
            kps = detections[label]

            kps.sort(key=lambda x: x[0].size)

            # Remove small detections 
            kps = [x for x in kps if x[0].size > 1]

            # Find the largest of the two keypoints placed furthest from each other
            bestkp = self.filterKeypoints(kps)

            # Remove the smallest half of the keypoints (in order to remove tail-points etc)
            if (self.onlyHeads and len(kps) > 1):
                numPts = len(kps) // 2
                kps = kps[-numPts:]

                # If the bestkp has been removed, add it again (largest of the two keypoints placed furthest from each other)
            if bestkp and (not bestkp[0] in kps):
                kps.extend(bestkp)

            # kps.sort(key=lambda x: x.size)
            # filtered += [kps[-1]]

            filtered += kps

        ## Find rotated bounding boxes of the detected keypoints
        bbs = self.findRotatedBB(filtered)

        filtered = [x[0] for x in filtered]

        return filtered, bbs

    def findRotation(self, img):
        """
        Calculates the rotation of the foreground pixels in the binary image
        
        Input:
            img: Binary input image 
            
        Output:
            theta : The orientation in degrees from [-pi/2 : pi/2]
        """

        _, cov = self.estimateGaussian(img)

        ## Get the eigenvalues/vectors and sort them by descending eigenvalues
        U, S, _ = np.linalg.svd(cov)
        x_v1, y_v1 = U[:, 0]

        theta = np.arctan((y_v1) / (x_v1))  # arctan vs arctan2. Use arctan2 to handled x_v1 = 0?

        return np.rad2deg(theta)

    def closestPos(self, img, target):
        """
        Finds the closest non-zero pixel position in the image to the given target position
        
        Input:
            img: Input image
            target: Tuple with the x, y coordinates of the target position
            
        Output:
            pos: The position of the closest non-zero pixel
            dist: The distance between the target and pos pixels
        """

        y, x = np.nonzero(img)

        distances = np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)
        nearest_index = np.argmin(distances)
        dist = np.min(distances)

        pos = (x[nearest_index], y[nearest_index])

        return pos, dist

    def getBB(self, img):
        """
        Computes the Axis-aligned bounding box for the provide image.
        It is assumed that the input image is a binary image with a single BLOB in it
        
        Input:
            img: Input binary image
            
        Output:
            tl: Tuple containing the coordinates of the top left point of the BB
            br: Tuple containing the coordinates of the bottom right point of the BB
            center: Tuple containing the coordinates of the center of the BB
        """

        y, x = np.nonzero(img)

        if len(y) == 0:
            return (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1)

        tl = (np.min(x), np.min(y))
        br = (np.max(x), np.max(y))
        center = (np.mean(x), np.mean(y))

        if br[0] - tl[0] > br[1] - tl[1]:  # If width > height
            left = (np.min(x), np.mean(y[x == np.min(x)]))  # Middle of the LEFT edge of the BLOB
            right = (np.max(x), np.mean(y[x == np.max(x)]))  # Middle of the RIGHT edge of the BLOB
        else:
            left = (np.mean(x[y == np.min(y)]), np.min(y))  # Middle of the TOP edge of the BLOB
            right = (np.mean(x[y == np.max(y)]), np.max(y))  # Middle of the BOTTOM edge of the BLOB

        return tl, br, center, left, right

    def estimateGaussian(self, img):
        """
        Computes the mean and covariance of a gaussian approximated to the foreground pixels in the provided image
        
        Input:
            img: Input binary image
            
        Output:
            params: tuple containing the mean and covariance of a unimodal multivariate gaussian
        """

        y, x = np.nonzero(img)

        mean = (np.mean(x), np.mean(y))
        cov = np.cov(np.vstack([x, y]))

        return (mean, cov)

    def findRotatedBB(self, keypoints):
        """
        Computes the rotated bounding box that fits the best to the BLOB that each keypoint is assocaited with
        
        Input:
            keypoints: List of opencv Keypoints or list of BLOB labels
            
        Output:
            bb: List of dictionaries each contatining the top left coordiantes, width, height and angle of the rotated bounding box as well as the center coordiantes of the origianl bounding box
        """

        # if self.bboxes and (self.camId == 2 or self.camId == 3):
        #     keypoints = [x for x in range(len(self.bboxes))]

        bbs = []
        for key in keypoints:
            bbox = None
            if self.bboxes:  # If BBOXs have been detected, utilize these
                if self.camId == 2 or self.camId == 3:
                    try:
                        bbox = self.bboxes[key]
                    except:
                        bbox = None
                else:
                    bbox = key[-1]
            if bbox is not None:
                predInfo = True
            else:
                predInfo = False

            if predInfo:
                blob = self.labels.copy()
                blob[:bbox[1]] = 0
                blob[bbox[3] + 1:] = 0
                blob[:, :bbox[0]] = 0
                blob[:, bbox[2] + 1:] = 0

                if self.camId == 2 or self.camId == 3:
                    # Get the largest blob within the ROI
                    unique_labels, label_count = np.unique(blob, return_counts=True)

                    bg_index = unique_labels == 0
                    unique_labels = unique_labels[~bg_index]
                    label_count = label_count[~bg_index]
                    if len(label_count) == 0:
                        continue
                    label_idx = np.argmax(label_count)
                    label = unique_labels[label_idx]
                else:
                    # Get the closest foreground pixel compared to the detected keypoint
                    label = self.labels[int(key[0].pt[1]), int(key[0].pt[0])]
                    if label == 0:
                        label_pos, dist = self.closestPos(self.thresh, (int(key[0].pt[0]), int(key[0].pt[1])))
                        label = self.labels[label_pos[1], label_pos[0]]
                blob = (blob == label).astype(np.bool).astype(np.uint8)

            else:  # If no bbox, then just find the closest foreground pixel to the detected keypoint
                if np.issubdtype(type(key), np.integer):  # Cam2
                    label = key
                else:  # Cam1
                    label = self.labels[int(key[0].pt[1]), int(key[0].pt[0])]

                    ## If the label at the keypoint position is the background label, find the closest foreground pixel and take its label instead
                    if label == 0:
                        label_pos, dist = self.closestPos(self.thresh, (int(key[0].pt[0]), int(key[0].pt[1])))
                        label = self.labels[label_pos[1], label_pos[0]]

                blob = (self.labels == np.ones(self.labels.shape) * label).astype(np.uint8)

            if np.sum(blob) == 0:  # If there are no foreground pixels, just skip the keypoint
                continue

            theta = self.findRotation(blob)
            tl, br, center, left, right = self.getBB(blob)

            if predInfo:
                tl = (bbox[0], bbox[1])
                br = (bbox[2], bbox[3])

            rot_blob = rotate(blob, theta, center)
            r_tl, r_br, _, _, _ = self.getBB(rot_blob)

            mean, cov = self.estimateGaussian(blob)

            bb = {}
            # Rotated Bounding box
            bb["tl_x"] = r_tl[0]
            bb["tl_y"] = r_tl[1]
            bb["w"] = r_br[0] - r_tl[0]
            bb["h"] = r_br[1] - r_tl[1]
            bb["theta"] = theta

            # Standard/Axis-aligned Bounding box
            bb["aa_tl_x"] = tl[0]
            bb["aa_tl_y"] = tl[1]
            bb["aa_w"] = br[0] - tl[0]
            bb["aa_h"] = br[1] - tl[1]

            # Box points of the BBOX
            bb["c_x"] = center[0]
            bb["c_y"] = center[1]
            bb["l_x"] = left[0]
            bb["l_y"] = left[1]
            bb["r_x"] = right[0]
            bb["r_y"] = right[1]

            # Label
            bb["label"] = label

            # Estimated Gaussian
            bb["mean"] = mean
            if self.camId == 1:
                bb["cov"] = np.eye(2)
            else:
                bb["cov"] = cov

            # Confidence:
            if predInfo:
                bb["conf"] = bbox[-1]
            else:
                bb["conf"] = 1.0

            bbs.append(bb)

        return bbs

    def fillHoles(self, img):
        """
        Fill the holes in the provided binary image.
        This is done by applying the floodFill function on all the parent contours
        
        Input:
            img: Binary image
        
        Output:
            img: Binary image
        """

        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1:]
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if hierarchy is not None:
            # 上一个轮廓序号、下一个轮廓序号、子轮廓序号、父轮廓序号
            for idx, h in enumerate(hierarchy[0]):
                try:
                    if h[3] > -1:
                        cv2.floodFill(
                            img, None,
                            (contours[idx][0][0][0] + 1, contours[idx][1][0][1] + 1),
                            255,
                        )
                except:
                    print(f"floodFill failed in {video_name}/{region_name}")
                    # exit(-1)
        return img.astype(bool)

    def filterKeypoints(self, kps):
        """
        Filter the provided keypoints based on their euclidean distance between two keypoints. The further away the better
        The largest of the two keypoints found are then returned.
        
        Input:
            kps: List of cv2.KeyPoints
            
        Output:
            bestKps: cv2.KeyPoint in a list or an empty list
        
        """

        maxDist = 0
        bestKps = []
        for row, rowVal in enumerate(kps):
            for col, colVal in enumerate(kps):
                dist = np.linalg.norm(np.asarray(rowVal[0].pt) - np.asarray(colVal[0].pt))
                if (dist > maxDist):
                    if dist < self.minSkeletonSize:  # Ignore short skeletons
                        continue
                    bestKps = [rowVal, colVal]
                    maxDist = dist
        # Only return the largest of the two keypoints
        if bestKps:
            bestKps.sort(key=lambda x: x[0].size, reverse=True)
            return [bestKps[0]]
        return []

    def entropySplit(self, img):
        """
        Find threshold resulting in maximum entropy
        Adapted from ImageJ's plugin:    
            https://imagej.nih.gov/ij/plugins/download/Entropy_Threshold.java
            
        Input: 
            img: Grayscale image
            
        Output:
            res: Threshold value
        """

        _start = time.time()
        # Ensure correct range
        img[0][0] = 0.0
        img[1][0] = 255.0

        # Calc normed histogram
        hist = np.bincount(img.flatten().astype(int))
        hist = hist / np.sum(hist)
        histSum = np.cumsum(hist)  # Cummulative sum of histogram
        notZero = np.nonzero(hist)[0]

        # Calc black and white entropy
        eps = 0.000000000000000000000000001
        hist += eps  # NOTE: Added instead of if check as to allow vectorization

        hB = np.zeros(hist.size)
        hW = np.zeros(hist.size)
        for t in notZero:
            # Black entropy
            ids = np.arange(0, t + 1)
            res = abs(np.sum(hist[ids] / histSum[t] * np.log(hist[ids] / histSum[t])))
            hB[t] = res

            # White entropy
            pTW = 1.0 - histSum[t]
            ids = np.arange(t + 1, hist.size)
            res = abs(np.sum(hist[ids] / pTW * np.log(hist[ids] / pTW)))
            hW[t] = res

        # Return histogram index resulting in max entropy
        res = np.argmax((np.array(hB) + np.array(hW)))
        _end = time.time()
        if (self.timer):
            print("Entropy time: {0}".format(_end - _start))
        return res

    def intermodesSplit(self, img, verbose=False):
        """
        Find threshold by finding middle point between two modes in the image histogram.
        If the histogram is not bimodal by default, the image will be mean filtered untill it is
            
        Input: 
            img: Grayscale image
            verbose: Boolean stating whether information should be printed
            
        Output:
            thresh: Threshold value  
        """

        _start = time.time()
        # Ensure correct range
        img[0][0] = 0.0
        img[1][0] = 255.0

        # Calc histogram
        hist = np.bincount(img.flatten().astype(int))

        # Iterative mean filtering (window size = 3) until hist is bimodal
        iterations = 0
        f = np.ones((24)) / 3
        while (not (self.isBimodal(hist))):
            hist = np.convolve(hist, f, 'same')
            iterations += 1
            if (iterations > 10000):
                tt = 0
                break

        # Find mean of the two peaks in the histogram
        tt = []
        for i in np.arange(1, len(hist) - 1):
            if (hist[i - 1] < hist[i] and hist[i + 1] < hist[i]):
                tt.append(i)
        thres = np.floor(np.mean(tt))

        _end = time.time()
        if (verbose):
            print("Intermodes thresholding:")
            print(" - found threshold: {0}".format(thres))
            print(" - after {0} iterations".format(iterations))
            print(" - took {0} seconds".format(_end - _start))
        return thres

    def isBimodal(self, hist):
        """
        Check if the provided histogram is bimodal, by counting how many local maximum bins there are
            
        Input: 
            hist: Histogram contained in numpy array
            
        Output:
            Boolean
        """

        modes = 0

        for i in np.arange(1, len(hist) - 1):
            if (hist[i - 1] < hist[i] and hist[i + 1] < hist[i]):
                modes += 1
                if (modes > 2):
                    return False

        if (modes == 2):
            return True
        return False

    def nms(self, kp, overlap_thresh=0.5):
        """
        Non maximum suppression of the supplied keypoints, based on the overlap of the keypoints' bounding boxes.
        
        Input:
            kp: Dict of lists of found cv2.KeyPoints
            overlap_thresh: Threhsold for how much the keypoitns bounding boxes can overlap
                
        Output:
            keys: List of fitlered keypoints
        """

        # Return empty list if no positions are given in the keypoint input argument
        if len(kp) == 0:
            return []

        # List of picked indexes
        pick = []

        # Get coordinates of the keypoints
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for i in kp:
            # Radius of bounding box
            radius = max(i[0].size // 2,
                         2)  # Enforces minimum radius of 2, to insure overlap between direct neighbour pixels, and force removal of one of them (using NMS)
            # Upper left corner of bounding box
            x1.append(i[0].pt[0] - radius)
            y1.append(i[0].pt[1] - radius)
            # Lower right corner of bounding box
            x2.append(i[0].pt[0] + radius)
            y2.append(i[0].pt[1] + radius)

        # Convert to float numpy arrays
        x1 = np.asarray(x1, dtype=np.float32)
        y1 = np.asarray(y1, dtype=np.float32)
        x2 = np.asarray(x2, dtype=np.float32)
        y2 = np.asarray(y2, dtype=np.float32)

        # Calculate the area of the bounding boxes
        # and get the sorted index values based on
        # the area (sorted in ascending order).
        area = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(area)

        while len(idxs) > 0:
            # Get the highest area point of the ones remaining
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # Calculate the bounding boxes which contain the current point, and each of the other points per box.
            # Finds the upper left and lower right corners of the box that will be there if the two points' bounding boxes overlap
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Get the height/width of the new bounding boxes, used to calculate area
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            # Determine the ratio between the new bounding box areas and the area of the current point in focus
            overlap = (h * w) / area[i]

            # Handle cases of total overlap 
            ## Get indecies for when the min x/min y/max x/max y are inside the bb of the point in focus.
            x1max = np.where(x1[i] > x1[idxs[:last]])[0]
            y1max = np.where(y1[i] > y1[idxs[:last]])[0]
            x2max = np.where(x2[i] < x2[idxs[:last]])[0]
            y2max = np.where(y2[i] < y2[idxs[:last]])[0]

            max1 = list(set(x1max).intersection(y1max))  ## Find the cases where both x1max/y1max are true
            max2 = list(set(x2max).intersection(y2max))  ## Find the cases where both x2max/y2max are true
            for k in max2:
                for i in max1:
                    if k == i:  # The case where all corners of the box are within in the bb of the point in focus
                        overlap[k] = 1.0

            # Delete points that overlap too much with the current point 
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        keys = []
        for i in pick:
            keys.append(kp[i])

        return keys

    def bgSubtract(self, image):
        """
        Simple background subtraction method.
        Finds the max value across channels per pixel, and then normalizes the image to the range [0; 255]
        
        Input:
            image: Color image
        
        Output: 
            res: Resulting grayscale image
        """

        _start = time.time()
        image = image.astype(np.int16)
        diff = np.abs(image - self.bg)
        diff = np.max(diff, axis=2)
        diff = 255 * (diff - np.min(diff)) / (np.max(diff) - np.min(diff))

        res = diff.astype(np.uint8)
        _end = time.time()
        if (self.timer):
            print("BG sub time: {0}".format(_end - _start))
        return res

    # ======================= remove shade function ======================== #
    def max_filtering(self, N, I_temp):
        wall = np.full((I_temp.shape[0] + (N // 2) * 2, I_temp.shape[1] + (N // 2) * 2), -1)
        wall[(N // 2):wall.shape[0] - (N // 2), (N // 2):wall.shape[1] - (N // 2)] = I_temp.copy()
        temp = np.full((I_temp.shape[0] + (N // 2) * 2, I_temp.shape[1] + (N // 2) * 2), -1)
        for y in range(0, wall.shape[0]):
            for x in range(0, wall.shape[1]):
                if wall[y, x] != -1:
                    window = wall[y - (N // 2):y + (N // 2) + 1, x - (N // 2):x + (N // 2) + 1]
                    num = np.amax(window)
                    temp[y, x] = num
        A = temp[(N // 2):wall.shape[0] - (N // 2), (N // 2):wall.shape[1] - (N // 2)].copy()
        return A

    def min_filtering(self, N, A):
        wall_min = np.full((A.shape[0] + (N // 2) * 2, A.shape[1] + (N // 2) * 2), 300)
        wall_min[(N // 2):wall_min.shape[0] - (N // 2), (N // 2):wall_min.shape[1] - (N // 2)] = A.copy()
        temp_min = np.full((A.shape[0] + (N // 2) * 2, A.shape[1] + (N // 2) * 2), 300)
        for y in range(0, wall_min.shape[0]):
            for x in range(0, wall_min.shape[1]):
                if wall_min[y, x] != 300:
                    window_min = wall_min[y - (N // 2):y + (N // 2) + 1, x - (N // 2):x + (N // 2) + 1]
                    num_min = np.amin(window_min)
                    temp_min[y, x] = num_min
        B = temp_min[(N // 2):wall_min.shape[0] - (N // 2), (N // 2):wall_min.shape[1] - (N // 2)].copy()
        return B

    def background_subtraction(self, I, B):
        O = I - B
        norm_img = cv2.normalize(O, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        return norm_img

    def removeShade(self, image, M=0, N=20):
        if M == 0:
            # max_filtering
            A = self.max_filtering(N, image)
            # min_filtering
            B = self.min_filtering(N, A)
            # subtraction
        elif M == 1:
            # min_filtering
            A = self.min_filtering(N, image)
            # max_filtering
            B = self.max_filtering(N, A)
            # subtraction
        normalised_img = self.background_subtraction(image, B)
        return normalised_img

    # ======================= remove shade function ======================== #

    def interestPoints(self, findJunctions=False):
        """
        Identify interest points (i.e. end points) from skeletonized image
        
        Input:
            findJunctions: Whether keypoints at junctions (Where a line splits into two or more) should be counted
            
        Output:
            endpoints: Dictionary of cv2.KeyPoints
        
        """

        _start = time.time()

        kernel = np.uint8([[1, 1, 1, 1, 1],
                           [1, 15, 15, 15, 1],
                           [1, 15, 100, 15, 1],
                           [1, 15, 15, 15, 1],
                           [1, 1, 1, 1, 1]])

        ret, self.labels = cv2.connectedComponents(self.thresh.astype(np.uint8))

        endpoints = {}
        if self.bboxes:  # If BBOXs have been detected, utilize this information

            for idx, bbox in enumerate(self.bboxes):

                # Look at the thinned image within the current BBOX, and do the same for the labels mat
                reduced_thin = self.thin.copy()
                reduced_thin[:bbox[1]] = 0
                reduced_thin[bbox[3] + 1:] = 0
                reduced_thin[:, :bbox[0]] = 0
                reduced_thin[:, bbox[2] + 1:] = 0

                reduced_labels = self.labels.copy()
                reduced_labels[:bbox[1]] = 0
                reduced_labels[bbox[3] + 1:] = 0
                reduced_labels[:, :bbox[0]] = 0
                reduced_labels[:, bbox[2] + 1:] = 0

                # Get unique labels in the bbox, all except background (0)
                unique_labels, label_count = np.unique(reduced_labels, return_counts=True)

                bg_index = unique_labels == 0
                unique_labels = unique_labels[~bg_index]
                label_count = label_count[~bg_index]
                if len(label_count) == 0:
                    continue

                # Get the most occuring label, and filter the thinned and label mats
                label_idx = np.argmax(label_count)
                label = unique_labels[label_idx]

                reduced_labels = (reduced_labels == label).astype(np.bool).astype(np.uint8)
                reduced_thin = np.multiply(reduced_thin, reduced_labels)

                # Get the interest point of the most prominent thinned skeleton in the BBOX
                points = self.getInterestPoints(reduced_thin, self.labels, kernel, findJunctions, bbox)
                for key in points:
                    if key in endpoints:
                        endpoints[key].extend(points[key])
                    else:
                        endpoints[key] = points[key]

        # else:
        endpoints = self.getInterestPoints(self.thin, self.labels, kernel, findJunctions, None)

        # Remove any empty lists in the dict
        endpoints = {key: endpoints[key] for key in endpoints if endpoints[key]}

        # Apply Non-maximum suppresion in order to remove redundant points close to each other
        for label in endpoints:
            endpoints[label] = self.nms(endpoints[label], overlap_thresh=self.nms_thresh)

        _end = time.time()

        if (self.timer):
            print("Interest points time: {0}".format(_end - _start))
        return endpoints

    def getInterestPoints(self, thinned_image, labels, kernel, findJunctions, bbox):
        """
        Takes a thinned input image and finds "interest points" by applying the provided kernel

        Input:
            thinned_image: An image which has been thinend to a single pixels width for each blob
            labels: A companion image for the thinned_image with labels
            kernel: Matrix/kenrel which should be applied to the thinned image in order to find interest points
            findJunctions: Boolean parameter indicating whether junctions in the thinned blobs should be detected
            bbox: The associated bbox if any

        Output:
            encpoints: Dictionary of detected interest points

        """
        filtered = cv2.filter2D(thinned_image.astype(np.uint8), -1, kernel)

        # Find endpoints
        positions = np.where((filtered == 116) | (filtered == 117) | (filtered == 118) | (filtered == 131))
        points = list(zip(list(positions[0]), list(positions[1])))
        endpoints = self.points2CvKeypoints(points, labels, bbox)

        # Find junctions
        if (findJunctions):
            positions = np.where((filtered == 148) | (filtered == 149) | (filtered == 150) | (filtered == 151))
            points = list(zip(list(positions[0]), list(positions[1])))
            junctions = self.points2CvKeypoints(points, labels, bbox)

            for label in junctions:
                if (label in endpoints):
                    # Decrease the significance (size) of the junctions-points. (As they are usually larger than the other keypoints)
                    for x in junctions[label]:
                        x[0].size = x[0].size / 2.5
                    endpoints[label] += junctions[label]

        return endpoints

    def points2CvKeypoints(self, points, labels, bbox):
        """
        Converts the provided points into cv2.Keypoint objects.

        Input:
            points: List of points
            labels: Grayscale image where each BLOB has pixel value equal to its label
            
        Output:
            keypoints: Dict of cv2.KeyPoints obejcts, where the keys are the BLOB labels.
        """

        keypoints = {}
        for p in points:
            currLabel = labels[p]
            if (currLabel not in keypoints):
                keypoints[currLabel] = []

            size, rot = self.calcEig(*p, self.thresh, labels, self.winSize)
            if (size == -1):
                continue
            kp = cv2.KeyPoint(float(p[1]), float(p[0]), size, rot, -1, -1, currLabel)
            keypoints[currLabel].append([kp, bbox])
        return keypoints

    def calcEig(self, y, x, image, labels, winSize=20, calcAngle=False):
        """
        Determine the size (Importance) of the keypoint, and the orientation of the keypoint (if set to).
        This is done by determining the eigenvalues of the covariance matrix of the coordinate values of the detected head, in some window defined by winSize x winSize
        Input:
            x: x coordinate of the keypoint
            y: y coordinate of the keypoint
            image: Binary image
            labels: Grayscale image where each BLOB has pixel value equal to its label 
            winSize: Size of the window extract around the keypoint
            calcAngle: Whether to calculate the angle of the extracted patch
        
        Output:
            size: Size of the keypoint, based on the smallest eigenvalue
            angle: Angle of the keypoint
        """
        ## Find the BLOB label at the keypoint
        labelId = labels[y, x]

        ## extract patch around keypoint
        roi = extractRoi(labels, (x, y), winSize)
        mask = roi == labelId

        y, x = np.nonzero(mask)
        x_ = x - winSize / 2
        y_ = y - winSize / 2
        coords = np.vstack([x_, y_])

        # Skip BLOBs with size smaller than minimum BLOB size
        if (coords.shape[-1] <= self.minPatchArea):
            return -1, -1

        ## Calculate eigenvectors of the covariance matrix of the patch
        cov = np.cov(coords)
        eVal, eVec = np.linalg.eig(cov)

        # Take the smallest eigenvalue, as it will most likely be the variance along the width of the head
        size = min(eVal)

        angle = 0.0
        if (calcAngle):
            sort_id = np.argmax(eVal)
            eVec1 = eVec[:, sort_id]
            x, y = eVec1
            angle = np.degrees(np.arctan2(x, y))

        return size, angle


def _analyseFrame(frame, frameCount, det, predet_df, showVideo, frame_time_str=None):
    """
    Performs the main loop of detection fish in a frame

    Input:
        frame: The frame to be analyzed
        frameCount: The frame number 
        det: The initalized detection object, which performs the detections
        predet_df: A Pandas dataframe containing prior detected bounding boxes, if any
        showVideo: A boolean indicating whetehr to show the frame that is being processed
    
    Output:
        detLst: A list of dicts, one per detection, containing all the necessary information
    """

    bboxes = None
    if predet_df is not None:
        predet_series = predet_df[predet_df["Frame"] == frameCount]
        bboxes = []

        # if len(predet_series) == 0:
        #     return []

        for index, value in predet_series.iterrows():
            bboxes.append((int(value["Upper left corner X"] // det.downsample),
                           int(value["Upper left corner Y"] // det.downsample),
                           int(value["Lower right corner X"] // det.downsample),
                           int(value["Lower right corner Y"] // det.downsample),
                           value["Confidence"]))

    ## Detect keypoints in the frame, and draw them
    kps, bbs = det.detect(frame, bboxes)
    detections = []

    for i in range(len(kps)):
        detections.append((kps[i], bbs[i]))

    if showVideo:
        # draw keypoint
        frame = cv2.drawKeypoints(det.frame, kps, None, (255, 0, 0), 4)
        cv2.rectangle(frame, (det.tl[0], det.tl[1]),
                      (det.br[0], det.br[1]), (0, 0, 0), 1)
        for bb in bbs:
            cv2.rectangle(frame, (bb["aa_tl_x"], bb["aa_tl_y"]),
                          (bb["aa_tl_x"] + bb["aa_w"], bb["aa_tl_y"] + bb["aa_h"]), (0, 0, 0), 1)
            cv2.circle(frame, (int(bb["l_x"]), int(bb["l_y"])), 2, (255, 0, 0), -1)
            cv2.circle(frame, (int(bb["c_x"]), int(bb["c_y"])), 2, (0, 255, 0), -1)
            cv2.circle(frame, (int(bb["r_x"]), int(bb["r_y"])), 2, (0, 0, 255), -1)

        ## Draw the skeletons
        frame[(det.thin), 0] = 0
        frame[(det.thin), 1] = 0
        frame[(det.thin), 2] = 255
        if (platform.system() == 'Windows'):
            if DEBUG == True:
                np_box = []
                for ib in bbs:
                    tl_x = ib['tl_x']
                    tl_y = ib['tl_y']
                    br_x = ib['tl_x'] + ib['w']
                    br_y = ib['tl_y'] + ib['h']

                    np_box.append([tl_x, tl_y, br_x, br_y])
                np_box = np.array(np_box)
                if len(np_box) != 0:
                    track_result = tracker.update(np_box)
                    if track_result.shape[0] != 0:
                        bbox_xyxy = track_result[:, :4]
                        identities = track_result[:, -1]
                        print(identities)
                        for i in range(bbox_xyxy.shape[0]):
                            try:
                                cv2.putText(frame, str(identities[i]), (int(bbox_xyxy[i][0]), int(bbox_xyxy[i][1])),
                                            cv2.FONT_HERSHEY_COMPLEX, 1,
                                            (0, 0, 255),
                                            1)
                            except:
                                continue

                cv2.imshow('test', frame)
                cv2.waitKey(50)
            else:
                # print(f"results are saved in {vid_out_file}")
                out_frame = cv2.resize(frame, out_vid_size)
                videoWriter.write(out_frame)
                # cv2.imshow('test', out_frame)
                # cv2.waitKey(100)
            # print("test")
        elif (platform.system() == 'Linux'):
            print(f"results are saved in {vid_out_file}")
            out_frame = cv2.resize(frame, out_vid_size)
            videoWriter.write(out_frame)
        # ******************************************************************* /
        # plt.show()

    # Save data into CSV file
    detLst = []
    for detection in detections:
        kps, bb = detection
        bb["frame"] = frameCount
        bb["frame_time_str"] = frame_time_str
        bb["x"] = kps.pt[0]
        bb["y"] = kps.pt[1]
        detLst.append((bb))

    return detLst


def load_SingleDetectRes(region_name):
    preDet_file = os.path.join(path, 'processed', region_name, video_name.replace("avi", "csv"))

    if os.path.isfile(preDet_file):
        predet_df = pd.read_csv(preDet_file, sep=",")
        try:
            predet_df["Frame"] = predet_df["Filename"].str.replace(".jpg", "").astype('int32')
        except:
            predet_df["Frame"] = predet_df["Frame"].astype('int32')
    else:
        predet_df = None
    return predet_df


def generate_SingleBgDetect(region_name):
    bgDet_out_path = os.path.join(path, 'bg_processed', region_name)

    bgDet_file = os.path.join(bgDet_out_path,
                              video_name.replace("mp4", "csv").replace("avi", "csv").replace("MOV", "csv"))
    video_out_path = os.path.join(bgDet_out_path, video_name)
    video_settings_file = os.path.join(bgDet_out_path, video_name.replace(".avi", "_settings.ini"))

    # Check if /processed/ folder exists
    if not os.path.isdir(bgDet_out_path):
        os.makedirs(bgDet_out_path)

    # 在经过精细化提取的过程中，需要复制settings.ini并重命名为 video_name_settings.ini
    if not os.path.isfile(video_settings_file):
        shutil.copy(
            os.path.join(path, 'settings.ini'),
            video_settings_file
        )
    return bgDet_file, video_settings_file, video_out_path


if __name__ == '__main__':
    import pandas as pd
    from paddleocr import PaddleOCR
    from modules.detection.ExtractBackground import BackgroundExtractor

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    if (platform.system() == 'Windows'):
        ap.add_argument("-f", "--path", default='E:\\data\\3D_pre\\D1_T1', help="Path to folder")

    elif (platform.system() == 'Linux'):
        ap.add_argument("-f", "--path", default='/home/HJZ/data/3D_pre/D4_T5/', help="Path to folder")
    ap.add_argument("--region_name",
                     default='1_6PPD1ppm', # D1_T1
                    # default='2_6PPD1ppm', # D1_T1
                    # default='2_6PPDQ50ppb', # D1_T1
                    # default='3_6PPD50ppb',  # D1_T1

                    help="region name of experience")
    ap.add_argument("--video_name", default='2021_10_11_21_49_59_ch02.avi', help="camerid and recording datetime")
    ap.add_argument("-v", "--video", default=True, action='store_true', help="input media is video")
    ap.add_argument("-i", "--images", action='store_true', help="Use extracted images instead of mp4 file")
    ap.add_argument("-pd", "--preDet", default=False, action='store_true', help="Use predetected bounding boxes")
    ap.add_argument("-verbose", "--verbose", default=False, action='store_true', help="show process")
    ap.add_argument("-Debug", "--Debug", default=False, action='store_true', help="show process")

    args = vars(ap.parse_args())

    reader = PaddleOCR(use_angle_cls=True, lang='en')
    # ARGUMENTS *************
    DEBUG = args["Debug"]
    # DEBUG = True
    path = args["path"]
    video = args["video"]
    useImages = args["images"]
    preDet = args["preDet"]
    verbose = args["verbose"]

    region_name = args["region_name"]
    video_name = args["video_name"]

    if (platform.system() == 'Windows'):
        showVideo = True
    elif (platform.system() == 'Linux'):
        showVideo = True

    camNO = video_name.split(".")[0].split("_")[-1]
    video_nameT = '_'.join(video_name.split(".")[0].split("_")[: -1])
    base_cfg_path = path
    camId = camera_id_map[camNO]

    # background 相关信息
    bg_path = os.path.join(path, 'background')
    bg_file = os.path.join(bg_path, f'{video_nameT}_cam{camId}.jpg')
    if not os.path.isfile(bg_file):
        print("using general background")
        bg_file = os.path.join(bg_path, f'background_cam{camId}.jpg')

    # 处理结果相关信息
    bgDet_file, video_settings_file, vid_out_file = generate_SingleBgDetect(region_name)
    if showVideo and (not DEBUG):
        fps = 25  # 视频帧率
        out_vid_size = (640, 480)
        videoWriter = cv2.VideoWriter(vid_out_file, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, out_vid_size)

    if DEBUG:
        from modules.tracking.SortTracker import Sort

        tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.15)
    # load deep learning detect result
    predet_df = None
    # predet_df = pd.DataFrame()
    # for iregion in region_name:
    #     df = load_SingleDetectRes(iregion)
    #     predet_df = pd.concat([predet_df, df])
    # if predet_df.shape[0] == 0:
    #     predet_df = None

    # Configure settings for either reading images or the video file.
    video_path = os.path.join(path, 'cut_video')
    vid_file = os.path.join(video_path, video_name)
    print("processing video: ", vid_file)
    cap = cv2.VideoCapture(vid_file)

    # Close program if video file could not be opened
    if not cap.isOpened():
        print("Could not open video file {0}".format(vid_file))
        sys.exit()
    if not os.path.isfile(bg_file):
        print("No background image present")
        print("... creating one.")
        # background的参数属于基础设置，在path目录下
        bgExt = BackgroundExtractor(bg_file, path, vid_file, video=video)
        bgExt.collectSamples()
        bg = bgExt.createBackground()
        cv2.imwrite(bg_file, bg)

    # Prepare detector
    if os.path.isfile(video_settings_file):
        region_settings = video_settings_file
    else:
        region_settings = os.path.join(path, 'settings.ini')
    det = BgDetector(camNO, base_cfg_path, bg_file, region_name, region_settings)

    # Prepare output dataframe
    df = pd.DataFrame()
    detLst = []

    frameCount = 0

    _start = time.time()

    # Analyse and detect fish using the video file
    while (cap.isOpened()):
        ret, frame = cap.read()

        frameCount += 1

        if frameCount < det.min_frame:
            continue

        if (frameCount > det.max_frame) and (det.max_frame > 0):
            frameCount -= 1
            if DEBUG:
                break

        if video:
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

        if (ret) and (frame is not None):
            if video_name in [
                '2021_10_12_08_49_59_ch02.avi',
                '2021_10_12_09_09_59_ch02.avi',
                '2021_10_12_09_29_59_ch02.avi',
                '2021_10_12_09_49_59_ch02.avi'
            ]:
                downsample = 1080 / 720
                frame = cv2.resize(frame, (1280, 720))

            time_pos = load_time_pos_setting(base_cfg_path, camNO)
            # frame_time_str = getImgTime(frame, time_pos, video_nameT, video_name, verbose=verbose, reader=reader)
            frame_time_str = None
            if frameCount % 500 == 0:
                print(f"Frame: {frameCount} in {frame_time_str}---{path}/{video_name}/{region_name}")
                print(f"costing time {time.time() - _start}")
                # break
            dets = _analyseFrame(frame, frameCount, det, predet_df, showVideo, frame_time_str)
            detLst.extend(dets)

        else:
            if (frameCount > 7750):
                frameCount -= 1
                break
            else:
                continue
    cap.release()
    cv2.destroyAllWindows()

    for bb in detLst:
        df_tmp = pd.DataFrame({
            'frame': [bb["frame"]],
            'cam': [camId],
            'x': [bb["x"] * det.downsample],
            'y': [bb["y"] * det.downsample],
            'tl_x': [bb['tl_x'] * det.downsample],
            'tl_y': [bb["tl_y"] * det.downsample],
            'c_x': [bb["c_x"] * det.downsample],
            'c_y': [bb["c_y"] * det.downsample],
            'w': [bb["w"] * det.downsample],
            'h': [bb["h"] * det.downsample],
            "theta": [bb["theta"]],
            "l_x": [bb["l_x"] * det.downsample],
            "l_y": [bb["l_y"] * det.downsample],
            "r_x": [bb["r_x"] * det.downsample],
            "r_y": [bb["r_y"] * det.downsample],
            "aa_tl_x": [bb["aa_tl_x"] * det.downsample],
            "aa_tl_y": [bb["aa_tl_y"] * det.downsample],
            "aa_w": [bb["aa_w"] * det.downsample],
            "aa_h": [bb["aa_h"] * det.downsample],
            "var_x": [bb["cov"][0, 0] * (det.downsample ** 2)],
            "var_y": [bb["cov"][1, 1] * (det.downsample ** 2)],
            "covar": [bb["cov"][0, 1] * (det.downsample ** 2)],
            "confidence": [bb["conf"]],
            "time_str": bb["frame_time_str"]
        })
        df = df.append(df_tmp)

    df = df.sort_values(by=['frame'], ascending=[True])

    if DEBUG:
        exit(123)
    writeConfig(path, [("CameraSynchronization", "cam{}_length".format(camId),
                        str(frameCount) + " #Automatically set in BgDetector.py")],
                cfgFile=video_settings_file
                )
    print("Saving data to: {0}".format(bgDet_file))
    df.to_csv(bgDet_file)
    _end = time.time()
    print(f"cost time:{_end - _start}")
