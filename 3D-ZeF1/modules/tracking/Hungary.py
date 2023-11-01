import cv2
import os
import argparse
import numpy as np
import pandas as pd
#from munkres import munkres
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.metrics.pairwise import pairwise_distances
import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')

### Module imports ###
from modules.tracking.tools.utilits import *
from common.utility import *

class Tracker:
    """
    Class implementation for associating detections into 2D tracklets.        
    """
    
    def __init__(self, dataPath, camId):
        """
        Initialize object
        
        Input:
            dataPath: String path to the video files
            camId: Camera view of the video to be analysed. 1 = Top, 2 = Front
        """
        
        self.cam = camId
        self.loadSettings(dataPath)
        self.tracks = []
        self.trackCount = 0
        self.oldTracks = []


    def loadSettings(self,path):
        """
        Load settings from config file in the provided path.
        
        Config file includes information on the following, which is set in the object:
            ghost_threshold: Threshold for how large the distance cost can be
            
        Input:
            path: String path to the folder where the settings.ini file is located
            camId: Indicates the camera view
        """
        
        config = readConfig(path)
        c = config['Tracker']
        self.downsample = config['Detector'].getint('downsample_factor')

        if self.cam == 1:
            # self.ghostThreshold = c.getfloat('cam1_ghost_threshold')
            self.ghostThreshold = 50
        else:
            self.ghostThreshold = c.getfloat('cam2_ghost_threshold')
        # else:
        #     print("Supplied camera id, {}, is not supported".format(self.cam))
        #     sys.exit()

        # self.maxKillCount = c.getint("max_kill_count")
        self.maxKillCount = 2
        self.minConfidence = c.getfloat("min_confidence")
    
    

    def matrixInverse(self, X, lambd = 0.001, verbose=False):
        """
        Tries to calculate the inverse of the supplied matrix.
        If the supplied matrix is singular it is regularized by the supplied lambda value, which is added to the diagonal of the matrix
        
        Input:
            X: Matrix which has to be inverted
        Output:
            X_inv: Inverted X matrix
        """
        
        try:
            if verbose:
                print("Condition number of {}".format(np.linalg.cond(X)))        
            X_inv = np.linalg.inv(X)
        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e):
                if verbose:
                    print("Adding regularizer of {}".format(lambd))
                    print("Condition number of {}".format(np.linalg.cond(X)))
                X = X + np.diag(np.repeat(lambd,X.shape[1]))
                
                if verbose:
                    print("Condition number of {}".format(np.linalg.cond(X)))
                X_inv = np.linalg.inv(X)
            else:
                raise
        
        return X_inv

    def mahalanobisDistance(self, Xp, Xg, M, psd=False):
        """
        Calcualtes the squared Mahalanobis distance between the supplied probe and gallery images
        
        Input:
            Xp: Numpy matrix containg the probe image features, with dimesnions [n_probe, n_features]
            Xg: Numpy matrix containg the gallery image features, with dimesnions [n_gallery, n_features]
            M: The Mahalanobis matrix to be used, with dimensions [n_features, n_features]
            psd: Describes whether M is a PSD matrix or not. If True the sklearn pairwise_distances function will be used, while if False a manual implementation is used
            
        Output:
            dm: Outputs a distance matrix of size [n_probe, n_gallery]
        """
        
        if psd:
            return pairwise_distances(Xp, Xg, metric="mahalanobis", VI=M)
        else:    
            mA = Xp.shape[0]
            mB = Xg.shape[0]
            dm = np.empty((mA, mB), dtype=np.double)
            for i in range(0, mA):
                for j in range(0, mB):  
                    difference = Xp[i] - Xg[j]
                    dm[i,j] = difference.dot(M.dot(difference.T)) 
            return dm

    def pairwiseDistance(self, detections, tracks):
        """
        Calculate pairwise distance between new detections and the old tracks.
        The matrix is of size: [s, s], where s = max(len(detections), len(tracks))*2.
        This is done to allow for potential ghost tracks, so a tracklet is not gone if just a signle detection is missed
        All values are set to a default weight defiend by the ghostThreshold from the config file.
        
        Input:
            detections: List of cv2.Keypoints
            tracks: List of Track objects
        
        Output:
            pDist: Matrix of size [s,s], containing the pairwise distances
        """
        
        maxLen = int(max(len(self.detections),len(self.tracks)))

        # Init matrix (including ghost tracks)
        pDist = np.ones((maxLen,maxLen), np.float32)*self.ghostThreshold

        # Update matrix
        for detIndex in range(len(self.detections)):
            for trackIndex in range(len(self.tracks)):
                pDist[detIndex][trackIndex] = self.distanceFunc(detIndex,trackIndex)

        return pDist

    
    def distanceFunc(self, detIndex, trackIndex):
        """
        Calculates a cost values between the provided detection and track
        
        The Euclidean distance is calculated, and turned into a norm by dividing with the max allwoed distance, from the config file.
        
        Input:
            detIndex: Int index of the detection
            trackIndex: Int index of the track
            
        Output:
            cost: The floating point values cost will either be the distCost or the dissimilarity
        
        """
        
        # Distance cost

        if self.cam == 1: 
            # L2 distance
            distCost = np.linalg.norm(
                self.tracks[trackIndex].pos[-1] - np.array(self.detections[detIndex].pt)
            )
        else:  # Make sure the coordinates are in order x,y and not y,x (as the cov matrix will be incorrect then)
            # Mahalanobis distance
            detPt = np.asarray(self.detections[detIndex].pt).reshape(1,2)
            trackPt = np.asarray(self.tracks[trackIndex].pos[-1]).reshape(1,2)
            mdist = self.mahalanobisDistance(detPt, trackPt, self.tracks[trackIndex].M)  # Square mahalanobis distances

            distCost = np.sqrt(mdist)
        
        return distCost

    def findMatches(self, assignM):
        """
        Find matches in the matrix computed using Hungarian
        
        Input:
            assignM: A binary matrix, where 1 indicates an assignment of row n to column m, and otherwise 0
                
        Output: 
            matches: List of tuples containing the row and column indecies for the matches
        """

        matches = []
        for mRow in range(0, len(assignM)):
            for pCol in range(0, len(assignM[mRow])):
                if(assignM[mRow][pCol]):
                    matches.append((mRow,pCol))

        matches.sort(key=lambda x: x[1],reverse=True)
        return matches


    def convertBBoxtoList(self, BBdict):
        
        return    [BBdict["tl_x"],
                   BBdict["tl_y"],
                   BBdict["c_x"],                   
                   BBdict["c_y"],                   
                   BBdict["w"],                   
                   BBdict["h"],                   
                   BBdict["theta"],                   
                   BBdict["l_x"],                   
                   BBdict["l_y"],                   
                   BBdict["r_x"],                   
                   BBdict["r_y"],                   
                   BBdict["aa_tl_x"],                   
                   BBdict["aa_tl_y"],                   
                   BBdict["aa_w"],                   
                   BBdict["aa_h"]]

    
    def recognise(self, frameNumber, detections, bbox, verbose=True):
        """
        Update tracker with new measurements.
        This is done by calculating a pairwise distance matrix and finding the optimal solution through the Hungarian algorithm.
        
        Input: 
            frameNumber: The current frame number (Int)
            detections: List of cv2.keyPoints (the detections) found in the current frame.
            bbox: List of dicts containing bounding boxes associated with the detected keypoints. 
            frame: The current frame as numpy array
            labels: Grayscale image where each BLOB has pixel value equal to its label 
            verbose: Whether to print information or not.
            
        Output:
            tracks: List of Track objects
        """

        for idx in reversed(range(len(bbox))):
            if bbox[idx]["confidence"] < self.minConfidence:
                del bbox[idx]
                del detections[idx]

        self.detections = detections

        # Update tracking according to matches
        numNew = len(self.detections)
        numOld = len(self.tracks)
        if(verbose):
            print("New detections: ", numNew)
            print("Existing tracks: ", numOld)

        for t in self.tracks:
            if verbose:
                print("ID {} - Kill Count {}".format(t.id, t.killCount))
            t.killCount += 1
        
        # Construct cost matrix
        costM = self.pairwiseDistance(detections, self.tracks)


        row_ind, col_ind = hungarian(costM)
        matches = [(row_ind[i], col_ind[i]) for i in range(row_ind.shape[0])]
        
        killedTracks = []
        for (mRow, pCol) in matches:
            '''
            matches 行代表检测的值，列代表轨迹的值，如 matches[1][3]代表第一个检测点和第三条轨迹之间的损失
            在匹配过程中，可能会遇到下列情况：
            1. 构造损失矩阵时：检测数量numNew>=轨迹数量numOld:
                检测索引mRow >= 检测数量numNew(因为是从0开始计算索引的，所以等号属于超纲的）:
                    不可能发生
                检测索引mRow < 检测数量numNew:
                    跟踪索引 pCol < 轨迹数量 numOld（说明pCol in tracker_list）:
                        损失阈值 matches[mRow][pCol] < ghostThreshold:
                            Operation: tracks[pCol].append(mRow)
                            
                        损失阈值 matches[mRow][pCol] >= ghostThreshold:
                            # 这个点可能有问题或者这个轨迹可能有问题，有两种可能：新增轨迹 或者 kill计数
                            Operation:  
                            
                    跟踪索引 pCol >= 轨迹数量 numOld（说明pCol not in tracker_list）:
                        损失阈值 matches[mRow][pCol] < ghostThreshold:
                            Operation:  不可能出现，因为在初始化的时候，matches为ghostThreshold的值
                            
                        损失阈值 matches[mRow][pCol] >= ghostThreshold:
                            Operation:  newTrack = Track(mRow)
                                        tracks.append(newTrack)
                                        轨迹数量增加了
                                        
            2. 构造损失矩阵时：检测数量numNew<轨迹数量numOld:
                检测索引mRow >= 检测数量numNew(因为是从0开始计算索引的，所以等号属于超纲的）:
                    可能发生，当轨迹数量大于检测数量时，但是这样的情况无意义
                    
                检测索引mRow <  检测数量numNew:
                    跟踪索引 pCol < 轨迹数量 numOld（说明pCol in tracker_list）:
                        损失阈值 matches[mRow][pCol] < ghostThreshold:
                            Operation: tracks[pCol].append(mRow)
                            
                        损失阈值 matches[mRow][pCol] >= ghostThreshold:
                            # 这个点可能有问题或者这个轨迹可能有问题，需要kill计数
                            Operation:  
                    跟踪索引 pCol >= 轨迹数量 numOld（说明pCol in tracker_list）:
                        不可能发生
            '''
            ## If the assignment cost is below the Ghost threshold, then update the existing tracklet
            if(costM[mRow][pCol] < self.ghostThreshold):
                # Update existing track with measurement
                p = np.array(detections[mRow].pt)
                self.tracks[pCol].pos.append(p)
                self.tracks[pCol].bbox.append(self.convertBBoxtoList(bbox[mRow]))
                self.tracks[pCol].M = self.matrixInverse(bbox[mRow]["cov"])
                self.tracks[pCol].mean = bbox[mRow]["mean"]
                self.tracks[pCol].frame.append(frameNumber)
                self.tracks[pCol].killCount = 0
                
            ## If the cost assignment is higher than the ghost threshold, then either create a new track or kill an old one
            else:
                # A new track is created if the following is true:
                # 1) The cost (L2 distance) is higher than the ghost threshold
                # 2) It is an actual detection (mRow < numNew)
                if(mRow < numNew):
                    # Create new track
                    newTrack = Track()
                    p = np.array(detections[mRow].pt)
                    newTrack.pos.append(p)
                    newTrack.bbox.append(self.convertBBoxtoList(bbox[mRow]))
                    newTrack.M = self.matrixInverse(bbox[mRow]["cov"])
                    newTrack.mean = bbox[mRow]["mean"]
                    newTrack.frame.append(frameNumber)
                    newTrack.id = self.trackCount
                    self.trackCount += 1
                    self.tracks.append(newTrack)

                    if verbose:
                        print("Num tracks: {}".format(len(self.tracks)))

                # The track is deleted if the following is true:
                # 1) The assigned detection is a dummy detection (mRow >= numNew),
                # 2) There are more tracks than detections (numOld > numNew)
                # 3) The assigned track is a real track (pCol < numOld)
                elif(numOld > numNew and pCol < numOld):
                    if(self.tracks[pCol].killCount > self.maxKillCount):
                        killedTracks.append(pCol)

                        if verbose:
                            print("Num tracks: {}".format(len(self.tracks)))       
        
        for pCol in sorted(killedTracks, reverse=True):
            self.oldTracks.append(self.tracks.pop(pCol))

        del(costM)     
        if verbose:
            print()   
    

###########################
###### MAIN START!!! ######
###########################
def drawline(track, frame):
    """
    Draw the last 50 points of the provided track on the provided frame
    
    Input: 
        track: Track object
        frame: 3D numpy array of the current frame
        
    Output:
        frame: Input frame with the line drawn onto it
    """
    
    colors = [(255,0,0),
              (255,255,0),
              (255,255,255),
              (0,255,0),
              (0,0,255),
              (255,0,255),
              (0,255,255),
              (100,100,100),
              (100,100,0),
              (0,100,0),
              (0,0,100),
              (100,0,100),
              (0,100,100),
              (150,150,150),
              (150,150,0),
              (150,0,0),
              (0,150,150),
              (0,0,150),
              (150,0,150)]
    # Loop through the positions of the given track
    for idx, i in enumerate(track.pos): 
        # Loop through the 50 newest positions.
        if idx < len(track.pos)-1 and idx < 50:
            line_pos = (int(track.pos[-idx-1][0]),int(track.pos[-idx-1][1]))
            line_pos2 = (int(track.pos[-idx-2][0]),int(track.pos[-idx-2][1]))
            c = colors[track.id%len(colors)]
            cv2.line(frame,line_pos,line_pos2,c,2)
        else:
            break
    return frame

def saveTrackCSV(allTracks, folder, csvFilename, downsample):
    df = tracks2Dataframe(allTracks)
    df['x'] *= downsample
    df['y'] *= downsample
    df['tl_x'] *= downsample
    df['tl_y'] *= downsample
    df['c_x'] *= downsample
    df['c_y'] *= downsample
    df['w'] *= downsample
    df['h'] *= downsample
    df["aa_tl_x"] *= downsample
    df["aa_tl_y"] *= downsample
    df["aa_w"] *= downsample
    df["aa_h"] *= downsample
    df["l_x"] *= downsample
    df["l_y"] *= downsample
    df["r_x"] *= downsample
    df["r_y"] *= downsample
    df['cam'] = camId

    outputPath = os.path.join(folder, csvFilename)
    print("Saving data to: {0}".format(outputPath))
    df.to_csv(outputPath)

def csvTracking(path, camId, df_fish):
    df_counter = 0
    # Prepare tracker
    tra = Tracker(path, camId)       
    
    frameList = np.linspace(
        df_fish["frame"].min(), df_fish["frame"].max(), df_fish["frame"].max() - df_fish["frame"].min() + 1,
        True, dtype=np.int32
    )

    for frameCount in frameList:
        #print("Frame: {0}".format(frameCount))        
        
        if df_counter >= len(df_fish):
            break
        
        fish_row = df_fish.loc[(df_fish['frame'] == frameCount)]
        if len(fish_row) == 0:
            continue
        kps, bbs = readDetectionCSV(fish_row, camId, ROI_Area, downsample=tra.downsample)
        df_counter += len(fish_row)

        tra.recognise(frameCount, kps, bbs)


    # Save CSV file
    allTracks = tra.tracks+tra.oldTracks

    csvFilename = detect_file
    saveTrackCSV(allTracks, track_folder, csvFilename, tra.downsample)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("-DT", "--DayTank", default="D1_T5", type=str)
    ap.add_argument("-pd", "--preDetector", default=True, action='store_true', help="Use pre-computed detections from csv file")
    ap.add_argument("--RegionName", default='3_CK', help="region name of experience")
    ap.add_argument("--detection_filename", default='2021_10_12_04_12_00_ch16.csv', help="region name of experience")
    ap.add_argument("--showTracklet", default=False, help="region name of experience")

    args = vars(ap.parse_args())
    
    # ARGUMENTS *************
    root_path = args["root_path"]
    DayTank = args["DayTank"]
    showTracklet = args["showTracklet"]
    preDet = args["preDetector"]
    region_name = args["RegionName"]
    detect_file = args["detection_filename"]
    video_file = detect_file.replace("csv", "avi")


    camNO = detect_file.split(".")[0].split("_")[-1]
    video_nameT = '_'.join(detect_file.split(".")[0].split("_")[: -1])

    path = os.path.join(root_path, DayTank)
    # Check if /processed/ folder exists
    track_folder = os.path.join(path, 'Hungary', region_name)
    if not os.path.isdir(track_folder):
        os.makedirs(track_folder)

    base_cfg_path = os.path.join(root_path, DayTank)
    camId = camera_id_map[camNO]
    ROI_Area = load_EXP_region_pos_setting(base_cfg_path, camNO)[region_name]

    if preDet:
        detPath = os.path.join(path, 'bg_processed', f'{region_name}/{detect_file}')
        if os.path.isfile(detPath):
            df_fish = pd.read_csv(detPath, sep=",") 
        else:
            print("Detections file found '{}' not found. Ending program".format(detPath))
            sys.exit()

    if preDet:
        csvTracking(path, camId, df_fish)
