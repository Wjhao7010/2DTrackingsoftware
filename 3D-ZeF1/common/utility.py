import cv2
import sys
import configparser
import re
import json
import os.path
import pickle as pkl
import pandas as pd
import numpy as np
import networkx as nx
from subprocess import Popen, PIPE
import sys, platform

if platform.system() != 'Windows':
    import fcntl
import socket

from common.Track import Track
# from sklearn.externals import joblib
import joblib
import math
import time
import datetime

camera_id_map = {
    "ch02": 1,
    "ch03": 2,
    "ch01": 3,
    "ch04": 3,
    "ch05": 2,
    "ch06": 1,
    "ch07": 2,
    "ch08": 3,
    "ch09": 1,
    "ch10": 2,
    "ch11": 3,
    "ch12": 1,
    "ch13": 3,
    "ch14": 2,
    "ch16": 1,
}

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

    'D7_T1': '4Hydroxy1ppm',
    'D7_T2': '4Hydroxy1ppm',
    'D7_T4': '4Hydroxy1ppm',

    'D4_T3': '6PPDQ1ppm',
    'D7_T5': '6PPDQ1ppm',
    'D8_T2': '6PPDQ1ppm',
    'D8_T4': '6PPDQ1ppm',
}
drag_tank_map = {
    '6PPD1ppm': {
        'D1_T1': {},
        'D2_T1': {},
        'D3_T1': {}
    },
    '6PPD500ppb': {
        'D1_T2': {},
        'D2_T2': {},
        'D3_T2': {}
    },
    '6PPD50ppb': {
        'D1_T3': {},
        'D2_T3': {},
        'D3_T3': {}
    },
    'RJ': {
        'D1_T4': {},
        'D2_T4': {},
        'D3_T4': {}
    },
    'CK': {
        'D1_T5': {},
        'D2_T5': {},
        'D3_T5': {}
    },
    '4Hydroxy500ppb': {
        'D4_T1': {},
        'D5_T1': {},
        'D6_T1': {}
    },
    '4Hydroxy50ppb': {
        'D4_T2': {},
        'D5_T2': {},
        'D6_T2': {}
    },
    '6PPDQ500ppb': {
        'D4_T4': {},
        'D5_T4': {},
        'D6_T4': {}
    },
    '6PPDQ50ppb': {
        'D4_T5': {},
        'D5_T5': {},
        'D6_T5': {}
    },
    '4Hydroxy1ppm': {
        'D7_T1': {},
        'D7_T2': {},
        'D7_T4': {}
    },
    '6PPDQ1ppm': {
        'D4_T3': {},
        'D7_T5': {},
        'D8_T2': {},
        'D8_T4': {}
    },
}


def manual_recorrect(time_str, start_time, video_name):
    #
    if 'ch08' in video_name:
        if start_time in ['2021_10_13_07_10_08']:
            time_str = time_str.replace("2021-10-13@7.10.08@", "2021_10_13_07_10_08")
        return time_str
    elif 'ch03' in video_name or 'ch02' in video_name or 'ch01' in video_name:
        if '2021_10_13_19' in start_time:
            time_str = time_str.replace("2021-10-13u9", "2021-10-13 19").replace("2021-10-139", "2021-10-13 19")
    elif "ch12" in video_name:
        if start_time in [
            "2021_10_12_08_11_59", "2021_10_12_08_23_59", "2021_10_12_08_35_59",
            "2021_10_12_08_47_59", "2021_10_12_08_59_59"
        ]:
            time_str = time_str.replace("2021-10-12 8", "2021 10 12 08")
        if '2021_10_17_01_50' in start_time:
            time_str = time_str.replace("2021-10-1701250", "2021-10-17 01 50")
        if '2021_10_16_06_01' in start_time:
            time_str = time_str.replace("2021-10-1606:01100", "2021_10_16_06_01_00")
        if '2021_10_16_06_21' in start_time:
            time_str = time_str.replace("2021-10-1606/21100", "2021_10_16_06_21_00")
        if '2021_10_16_07_47' in start_time:
            time_str = time_str.replace("2021-10-1607:47100", "2021_10_16_07_47_00")
        time_str = time_str.replace("2021-10-1421450:00", "2021-10-14 21 49:59"). \
            replace("2021-10-1420449:59", "2021-10-14 20 49:59"). \
            replace("2021-10-1505449:59", "2021-10-15 05 49:59"). \
            replace("2021-10-1421449:59", "2021_10_14_21_49_59")
        # replace("2021-10-1421449:59", "2021_10_14_21_49_59")

    return time_str


def getImgTime(frame, postion, start_time, video_name, verbose=False, reader=None):
    t_tl_y, t_br_y, t_tl_x, t_br_x = postion
    frame_area = frame[t_tl_y:t_br_y, t_tl_x:t_br_x]  # 裁剪时间

    if verbose:
        # print(result)
        cv2.imshow('frame area', frame)
        cv2.imshow('frame area1', frame_area)
        cv2.waitKey(10)

    # result = reader.readtext(frame_area.copy())
    result = reader.ocr(frame_area.copy(), det=False)

    if len(result) == 0:
        time_str = str(int(time.time()))

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
            replace("202-0", "2021-10").replace(":/", ":").replace("2021210-1", "2021-10-1"). \
            replace("2029270-1", "2021-10-1").replace("2029210-1", "2021-10-1"). \
            replace("2021210-1", "2021-10-1").replace("2029290-1", "2021-10-1"). \
            replace("2021290-1", "2021-10-1").replace("2029240-1", "2021-10-1"). \
            replace("202929-1", "2021-10-1").replace("2021410-1", "2021-10-1"). \
            replace("2024-10", "2021-10").replace("2029-10", "2021-10"). \
            replace("2021240-1", "2021-10-1").replace("202140-1", "2021-10-1"). \
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
            time_str = manual_recorrect(time_str, start_time, video_name)

            year = time_str[0:4]
            month = time_str[5:7]
            day = time_str[8:10]
            hh = time_str[11:13]
            mm = time_str[14:16]
            ss = time_str[17:19]
            time_str = f"{year}_{month}_{day}_{hh}_{mm}_{ss}"
            print(f"manual correct with {time_str}")
        return time_str.strip()


def readConfig(path, configFile=None):
    """
    Reads the settings.ini file located at the specified directory path
    If no file is found the system is exited
    
    Input:
        path: String path to the directory
        
    Output:
        config: A directory of dicts containing the configuration settings
    """

    config = configparser.ConfigParser(inline_comment_prefixes='#')
    if configFile is not None:
        configFile = configFile
    else:
        configFile = os.path.join(path, 'settings.ini')

    if (os.path.isfile(configFile)):
        config.read(configFile)
        return config
    else:
        print("Error loading configuration file:\n{0}\n Exiting....".format(configFile))
        sys.exit(0)


def writeConfig(path, updateValues, cfgFile=None):
    """
    Writes to the settings.ini file located at the specified directory path
    If no file is found the system is exited
    
    Input:
        path: String path to the directory
        updateValues: A Dict containing the new values o be added to the config file
        
    """
    config = configparser.ConfigParser(allow_no_value=True)
    if cfgFile is not None:
        configFile = cfgFile
    else:
        configFile = os.path.join(path, 'settings.ini')
    if (os.path.isfile(configFile)):
        config.read(configFile)

        for values in updateValues:
            if (values[1] is None) and (values[2] is None):
                config.remove_section(values[0])
                config.add_section(values[0])
            else:
                config.set(values[0], values[1], values[2])
        with open(configFile, 'w') as configfile:
            if sys.platform == 'linux':
                fcntl.flock(configfile.fileno(), fcntl.LOCK_EX)
            config.write(configfile)

        print("Updated configuration file: {}".format(configFile))
    else:
        print("Error loading configuration file:\n{0}\n Exiting....".format(configFile))
        sys.exit(0)


def findData(df_, *args):
    """
    Used to get data from .csv files
        
    Example: findData(df,'id',2) 
    - returns a dataframe with all information about id=2
    
    Example2: findData(df,'id', 2, 'camera', 1, 'frame')
    - returns a dataframe with the frames where id=2 has been located from camera=1
    
    Input:
        df_: Pandas dataframe containing data which has to be filtered
        args: A set of arguments which can be used to filter the supplied dataframe. See above examples for use case
        
    Output:
        tmp: Pandas dataframe based on the specified arguments
    
    Return: Dataframe (pandas)

    """

    # **************************
    # NEED A FIX, THAT VERIFIES THAT THE ARGUMENT IS REPRESENTED AS A COLUMN IN
    # THE CSV-FILE! OTHERWISE AN ERROR OCCURS
    # **************************

    tmp = df_
    if len(args) % 2:  # uneven number of arguments
        for idx, i in enumerate(args):
            if not idx % 2 and len(args) - idx == 1:
                tmp = tmp[i]
            elif idx % 2:
                continue
            else:
                tmp = tmp[tmp[i] == args[idx + 1]]
    else:  # even number of arguments
        for idx, i in enumerate(args):
            if idx % 2:
                continue
            tmp = tmp[tmp[i] == args[idx + 1]]
    return tmp


def csv2Tracks(csv, offset=0, minLen=10, maxFrame=None):
    """
    Loads all tracks in a CSV file into a dict of tracks, where the key = track id and value = track class
    The way the data is loaded depends on whether it is a 2D or 3D track.
    
    It should be noted that when loading a 3D track, if not a parent track, 
    then the bounding boxes will loaded so that both the bounding box from both views are present.
    This means the bounding box are then represented as 2D numpy arrays, instead of 1D arrays
    
    Input:
        csv: String path to csv file which has to be converted
        offset: The amount of offset applied to the frames
        minLen: The minimum lenght of a tracklet
        
    Output:
        tracks: A dict of Track objects
    """

    if (isinstance(csv, str)):
        if (not os.path.isfile(csv)):
            print("Error loading tracks. Could not find file: {0}".format(csv))
            return []
        csv = pd.read_csv(csv)
    if (isinstance(csv, pd.DataFrame)):
        uniqueIds = csv['id'].unique()
    else:
        print("Error loading tracks. 'csv2tracks' expects " +
              "either a Dataframe or path to a CSV file.")
        return []

    uniqueIds = csv['id'].unique()
    tracks = {}

    for trackId in uniqueIds:
        df_ = findData(csv, 'id', trackId)

        ## Load 2D track
        if ('cam' in csv.columns):
            uniqueCams = df_['cam'].unique()
            for camId in uniqueCams:
                df2_ = findData(df_, 'cam', camId)

                if maxFrame:
                    df2_ = df2_[df2_["frame"] <= maxFrame]
                    if len(df2_) == 0:  # If no valid detections left, go to the next tracklet
                        continue

                t = Track()
                t.cam = int(camId)
                t.x = np.array((df2_.x), dtype=float)
                t.y = np.array((df2_.y), dtype=float)
                t.cam_frame = np.array((df2_.frame), dtype=int)
                t.frame = np.array((df2_.frame), dtype=int) + offset
                t.id = trackId
                t.tl_x = np.array((df2_.tl_x), dtype=float)
                t.tl_y = np.array((df2_.tl_y), dtype=float)
                t.c_x = np.array((df2_.c_x), dtype=float)
                t.c_y = np.array((df2_.c_y), dtype=float)
                t.w = np.array((df2_.w), dtype=float)
                t.h = np.array((df2_.h), dtype=float)
                t.theta = np.array((df2_.theta), dtype=float)
                t.l_x = np.array((df2_.l_x), dtype=float)
                t.l_y = np.array((df2_.l_y), dtype=float)
                t.r_x = np.array((df2_.r_x), dtype=float)
                t.r_y = np.array((df2_.r_y), dtype=float)
                t.aa_tl_x = np.array((df2_.aa_tl_x), dtype=float)
                t.aa_tl_y = np.array((df2_.aa_tl_y), dtype=float)
                t.aa_w = np.array((df2_.aa_w), dtype=float)
                t.aa_h = np.array((df2_.aa_h), dtype=float)
                if (len(t.frame) > minLen):
                    key = "id{0}_cam{1}".format(t.id, t.cam)
                    t.id = key
                    tracks[t.id] = t


        ## Load 3D track
        else:
            ## Load all the triangulated positions as the main track
            t = Track()
            filtered = df_[df_['err'] != -1]
            t.x = np.array((filtered['3d_x']), dtype=float)
            t.y = np.array((filtered['3d_y']), dtype=float)
            t.z = np.array((filtered['3d_z']), dtype=float)

            t.tl_x = np.vstack((filtered['cam1_tl_x'].values, filtered['cam2_tl_x'].values))
            t.tl_y = np.vstack((filtered['cam1_tl_y'].values, filtered['cam2_tl_y'].values))
            t.c_x = np.vstack((filtered['cam1_c_x'].values, filtered['cam2_c_x'].values))
            t.c_y = np.vstack((filtered['cam1_c_y'].values, filtered['cam2_c_y'].values))
            t.w = np.vstack((filtered['cam1_w'].values, filtered['cam2_w'].values))
            t.h = np.vstack((filtered['cam1_h'].values, filtered['cam2_h'].values))
            t.theta = np.vstack((filtered['cam1_theta'].values, filtered['cam2_theta'].values))
            t.aa_tl_x = np.vstack((filtered['cam1_aa_tl_x'].values, filtered['cam2_aa_tl_x'].values))
            t.aa_tl_y = np.vstack((filtered['cam1_aa_tl_y'].values, filtered['cam2_aa_tl_y'].values))
            t.aa_w = np.vstack((filtered['cam1_aa_w'].values, filtered['cam2_aa_w'].values))
            t.aa_h = np.vstack((filtered['cam1_aa_h'].values, filtered['cam2_aa_h'].values))
            t.cam_frame = np.vstack((filtered["cam1_frame"].values, filtered["cam2_frame"].values))

            t.frame = np.array((filtered.frame), dtype=int) + offset
            t.id = trackId
            if (len(t.frame) > minLen):
                tracks[t.id] = t

            ## Load the parent tracks (Which consist of both the triangualted positions and the frames where only one view was present)
            t.parents = {}
            for camId in [1, 2]:
                filtered = df_[df_['cam{0}_x'.format(camId)] != -1]
                parent = Track()
                parent.cam = int(camId)
                parent.x = np.array((filtered['cam{0}_x'.format(camId)]), dtype=float)
                parent.y = np.array((filtered['cam{0}_y'.format(camId)]), dtype=float)

                parent.tl_x = np.array((filtered['cam{0}_tl_x'.format(camId)]), dtype=float)
                parent.tl_y = np.array((filtered['cam{0}_tl_y'.format(camId)]), dtype=float)
                parent.c_x = np.array((filtered['cam{0}_c_x'.format(camId)]), dtype=float)
                parent.c_y = np.array((filtered['cam{0}_c_y'.format(camId)]), dtype=float)
                parent.w = np.array((filtered['cam{0}_w'.format(camId)]), dtype=float)
                parent.h = np.array((filtered['cam{0}_h'.format(camId)]), dtype=float)
                parent.theta = np.array((filtered['cam{0}_theta'.format(camId)]), dtype=float)
                parent.aa_tl_x = np.array((filtered['cam{0}_aa_tl_x'.format(camId)]), dtype=float)
                parent.aa_tl_y = np.array((filtered['cam{0}_aa_tl_y'.format(camId)]), dtype=float)
                parent.aa_w = np.array((filtered['cam{0}_aa_w'.format(camId)]), dtype=float)
                parent.aa_h = np.array((filtered['cam{0}_aa_h'.format(camId)]), dtype=float)
                parent.cam_frame = np.array((filtered["cam{0}_frame".format(camId)]), dtype=int)

                parent.frame = np.array((filtered.frame), dtype=int) + offset
                parent.id = trackId
                t.parents[camId] = parent
    return tracks


def tracks2Csv(tracks, csvPath, overwriteCsv=False):
    """
    Saves list of 'Track' objects to a CSV file
    
    Input:
        tracks: List of Track objects
        csvPath: String path to csv file
        overwriteCsv: Whether to overwrite existing CSV files
        
    """

    df = tracks2Dataframe(tracks)
    # Write dataframe to CSV file
    if (overwriteCsv):
        fileName = csvPath
    else:
        fileName = checkFileName(csvPath)
    df.to_csv(fileName)
    print('CSV file with tracks stored in:', fileName)


def tracks2Dataframe(tracks):
    """
    Saves lsit of Track objects to pandas dataframe
    
    Input:
        tracks: List of Track objects
        
    Output:
        df: Pandas dataframe
    """

    if (len(tracks) == 0):
        print("Error saving to CSV. List of tracks is empty")
        return

    # Collect tracks into single dataframe
    df = pd.DataFrame()
    for t in tracks:
        df = df.append(t.toDataframe())

    df = df.sort_values(by=['frame', 'id'], ascending=[True, True])
    return df


def checkFileName(fn):
    """
    Return the next permutation of the specified file name if it exists.
    Next permutation is found by adding integer to file name
    
    Input:
        fn: String of filename, excluding .csv suffix
    
    Output:
        fn_: String of the new filename
    """

    fn_ = fn + '.csv'
    if fn_ and os.path.isfile(fn_):
        for k in range(0, 99):
            fn_ = fn + str(k) + '.csv'
            if not os.path.isfile(fn_):
                print('File already exists. New file name:', fn_)
                break
    return fn_


def frameConsistencyGraph(indecies, frames, coord, verbose=False):
    '''
    Constructs a Directed Acyclic Graph, where each node represents an index.
    Each node is connected to the possible nodes corresponding to detections in the previous and next frames
    The optimal path is found through minizing the reciprocal euclidan distance
    '''
    graph = nx.DiGraph()

    list_indecies = np.asarray([x for x in range(len(indecies))])

    # convert lists to numpy arrays
    if type(indecies) == list:
        indecies = np.asarray(indecies)
    if type(frames) == list:
        frames = np.asarray(frames)
    if type(coord) == list:
        coord = np.asarray(coord)

    # sort in ascending order
    sort_order = np.argsort(frames)

    indecies = indecies[sort_order]
    frames = frames[sort_order]
    coord = coord[sort_order]

    # Create a dict where the key is the frame and the element is a numpy array with the relevant indecies for the frame    
    fDict = {}
    for f in frames:
        fDict[f] = indecies[frames == f]

    # Go through each element in the dict
    prevNodes = []
    zeroCounter = 0
    for idx, key in enumerate(fDict):
        # Get the indecies for the frame (key)
        occurences = fDict[key]

        # For each occurance (i.e. index)
        currentNodes = []
        for occ in occurences:
            cNode = occ
            cIdx = list_indecies[indecies == occ]  # get the index needed to look up coordinates
            currentNodes.append((occ, cIdx))

            # If there are already soem nodes in the graph
            if prevNodes is not None:

                # for each of the previous nodes calculate the reciprocal euclidean distance to the current node, and use as the edge weight between the nodes
                for tpl in prevNodes:
                    pNode, pIdx = tpl

                    dist = np.linalg.norm(coord[pIdx][0] - coord[cIdx][0])
                    if (dist == 0):
                        if verbose:
                            print("0 distance between frame {} and {} - sorted indexes: {} and {}".format(frames[pIdx],
                                                                                                          frames[cIdx],
                                                                                                          pIdx, cIdx))
                        zeroCounter += 1
                        weight = 1
                    else:
                        weight = 1 / dist
                    graph.add_edge(pNode, cNode, weight=weight)

                    if verbose:
                        print("edge: {} - {} Weight: {}  Distance {}  -  previous 3D {} - current 3D {}".format(pNode,
                                                                                                                cNode,
                                                                                                                weight,
                                                                                                                dist,
                                                                                                                coord[
                                                                                                                    pIdx],
                                                                                                                coord[
                                                                                                                    cIdx]))

        prevNodes = currentNodes
    if zeroCounter and verbose:
        print("{} instances of 0 distance".format(zeroCounter))

    path = nx.dag_longest_path(graph)
    length = nx.dag_longest_path_length(graph)
    spatialDist = []
    temporalDist = []
    for idx in range(1, len(path)):
        pIdx = list_indecies[indecies == path[idx - 1]]
        cIdx = list_indecies[indecies == path[idx]]
        spatialDist.append(np.linalg.norm(coord[pIdx][0] - coord[cIdx][0]))
        temporalDist.append(frames[cIdx][0] - frames[pIdx][0])

    if verbose:
        print()
        print("Longest Path {}".format(path))
        print("Path length {}".format(length))
        print("Spatial Distances: {}".format(spatialDist))
        print("Total spatial distance: {}".format(np.sum(spatialDist)))
        print("Mean spatial distance: {}".format(np.mean(spatialDist)))
        print("Median spatial distance: {}".format(np.median(spatialDist)))
        print("Temporal Distances: {}".format(temporalDist))
        print("Total temporal distance: {}".format(np.sum(temporalDist)))
        print("Mean temporal distance: {}".format(np.mean(temporalDist)))
        print("Median temporal distance: {}".format(np.median(temporalDist)))
        print()

    return path, length, spatialDist, temporalDist


def getTrackletFeatures(multi_idx, df):
    '''
    Retrieves the frame and 3d coordiantes at each of the supplied indecies, for the supplied df
    Returns a tuple of lists
    '''
    frames = []
    coords = []

    # Restructure list so we only keep unique values, in an ordered list
    multi_idx = sorted(list(set(multi_idx)))

    first = multi_idx[0]
    last = multi_idx[-1]

    # Go through each idx, get the frame number and the 3D position
    for index in multi_idx:
        row = df.iloc[index]

        frames.append(int(row["frame"]))
        try:
            coords.append(np.asarray([row["3d_x"], row["3d_y"]]))
        except:
            print("extract 2D feature")
            coords.append(np.asarray([row['x'], row['y']]))
    # Check the index before and after the ones in the current list, if they exists, and if it belongs to the same tracklet
    for idx in [first - 1, last + 1]:
        if idx > 0 and idx < len(df) - 1:
            initRow = df.iloc[idx]

            if initRow["id"] == df.iloc[multi_idx[0]]["id"]:
                try:
                    pos = np.asarray([initRow["3d_x"], initRow["3d_y"], initRow["3d_z"]])
                    # Check whether the index has a valid 3D position
                    if not np.allclose(pos, np.ones(3) * -1):
                        multi_idx.append(idx)
                        frames.append(int(initRow.frame))
                        coords.append(pos)
                except:
                    print("extract 2D feature")
                    pos = np.asarray([initRow["x"], initRow["y"]])
                    # Check whether the index has a valid 3D position
                    if not np.allclose(pos, np.ones(2) * -1):
                        multi_idx.append(idx)
                        frames.append(int(initRow.frame))
                        coords.append(pos)

    return multi_idx, frames, coords


def getDropIndecies(df, verbose=False):
    '''
    Goes through a dataframe, finds all cases where there are several rows for the same frame in a single Tracklet.
    A graph is constructed where the distance is minimized. The indecies which should be removed from the dataframe is returned

    It is expected that the indecies of the dataframe are unique for each row
    '''

    ids = df.id.unique()  # array containing all unique tracklets ids
    drop_idx = []  # list to keep track of which indecies are not kept

    # Iterate over each unique ID in the dataframe
    for iID in ids:
        df_id = df[
            df.id == iID]  # Sub dataframe, containing all rows relevant for the current ID. Indecies are still that of the main dataframe

        frame_count = df_id["frame"].value_counts()  # How many times does a frame occur in the dataframe
        multi_assignment = frame_count[frame_count > 1].sort_index()  # isolating the frames with multiple assignments

        if len(multi_assignment) == 0:
            continue
        if verbose:
            print("ID {}".format(iID))

        counter = 0
        prevFrame = 0
        multi_idx = []
        keep_idx = []
        analyzed_frames = []
        all_frames = []

        # Iterate over the multiple assignments (A pandas Series)
        for idx, sIdx in enumerate(multi_assignment.items()):
            frame, count = sIdx
            all_frames.append(frame)  # Keep track of all frames that we look at

            # If first frame with multiple assignments, then skip to the next
            if idx == 0:
                prevFrame = frame
                continue

            # If the current and previous frames are continous i.e. they follow each other in the video, and therefore related
            if (frame - prevFrame) == 1:
                counter += 1
                prevIdx = list(df_id[df_id[
                                         "frame"] == prevFrame].index.values)  # Save the indecies in the main dataframe, for the previous frame, as the first frame in the series will be left out otherwise
                curIdx = list(df_id[df_id[
                                        "frame"] == frame].index.values)  # Save the indecies in the main dataframe, for the current frame

                multi_idx.extend(prevIdx + curIdx)  # Keep track of all indecies by combining them into one list

            # If the frames are not continous                       
            else:
                # If there is a previous series of frames, then analyze it
                if counter > 0:
                    # Get the indecies needed for the graph, their corresponding frames and 3D positions
                    multi_idx, frames, coords = getTrackletFeatures(multi_idx, df)

                    # Create graph and get and keep the indecies which minimizes the distance between the possible nodes
                    new_idx, _, _, _ = frameConsistencyGraph(multi_idx, frames, coords, verbose)
                    keep_idx.extend(new_idx)

                    # The frames which have been analyzed are kept
                    analyzed_frames.extend(sorted(list(set(frames))))

                    multi_idx = []

                counter = 0
            prevFrame = frame

        # Analyzing last set of multi detections across several connected frames, if any
        if len(multi_idx) > 0:
            # Get the indecies needed for the graph, their corresponding frames and 3D positions
            multi_idx, frames, coords = getTrackletFeatures(multi_idx, df)

            # Create graph and get and keep the indecies which minimizes the distance between the possible nodes
            new_idx, _, _, _ = frameConsistencyGraph(multi_idx, frames, coords, verbose)
            keep_idx.extend(new_idx)

            # The frames which have been analyzed are kept
            analyzed_frames.extend(sorted(list(set(frames))))

            multi_idx = []

        # Analyzing single-frame multi detections
        for idx, sIdx in enumerate(multi_assignment.items()):
            frame, count = sIdx

            # If the current frame is not in the list of already analyzed frames
            if frame not in analyzed_frames:
                # Get the indecies for the frame
                dfIdx = list(df_id[df_id["frame"] == frame].index.values)

                # Get the indecies needed for the graph, their corresponding frames and 3D positions
                multi_idx, frames, coords = getTrackletFeatures(dfIdx, df)

                # Create graph and get and keep the indecies which minimizes the distance between the possible nodes
                new_idx, _, _, _ = frameConsistencyGraph(multi_idx, frames, coords, verbose)
                keep_idx.extend(new_idx)

                # The frames which have been analyzed are kept
                analyzed_frames.extend(sorted(list(set(frames))))

                multi_idx = []

        all_idx = []
        # Find the indecies related to all of the frames which we have investigated
        for f in all_frames:
            all_idx.extend(list(df_id[df_id["frame"] == f].index.values))

        # Filter out all the indecies which we did not end up keeping
        drop_idx.extend([x for x in all_idx if x not in keep_idx])

    if verbose:
        print("Dropped indecies: {}".format(drop_idx))
    return drop_idx


def extractRoi(frame, pos, dia):
    """
    Extracts a region of interest with size dia x dia in the provided frame, at the specied position
    
    Input:
        frame: Numpy array containing the frame
        pos: 2D position of center of ROI
        dia: Integer used as width and height of the ROI
        
    Output:
        patch: Numpy array containing hte extracted ROI
    """

    h, w = frame.shape[:2]
    xMin = max(int(pos[0] - dia / 2) + 1, 0)
    xMax = min(xMin + dia, w)
    yMin = max(int(pos[1] - dia / 2) + 1, 0)
    yMax = min(yMin + dia, h)
    patch = frame[yMin:yMax, xMin:xMax]
    return patch


def rotate(img, angle, center):
    """
    Rotates the input image by the given angle around the given center
    
    Input:
        img: Input image
        angle: Angle in degrees
        center: Tuple consisting of the center the image should be rotated around
        
    Output:
        dst: The rotated image
    """

    rows, cols = img.shape[:2]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def prepareCams(path):
    """
    Loads the camera objects stored in a pickle file at the provided path
    
    Input:
        path: Path to the folder where the camera.pkl file is located
        
    Output:
        cams: A dict containing the extracted camera objects
    """

    cam1Path = os.path.join(path, 'cam1.pkl')
    cam2Path = os.path.join(path, 'cam2.pkl')
    if (not os.path.isfile(cam1Path)):
        print("Error finding camera calibration file: \n {0}".format(cam1Path))
        sys.exit(0)
    if (not os.path.isfile(cam2Path)):
        print("Error finding camera calibration file: \n {0}".format(cam2Path))
        sys.exit(0)

    cam1ref = os.path.join(path, 'cam1_references.json')
    cam2ref = os.path.join(path, 'cam2_references.json')
    if (not os.path.isfile(cam1ref)):
        print("Error finding camera corner reference file: \n {0}".format(cam1ref))
        sys.exit(0)
    if (not os.path.isfile(cam2ref)):
        print("Error finding camera corner reference file: \n {0}".format(cam2ref))
        sys.exit(0)

    cams = {}
    cam1 = joblib.load(cam1Path)
    cam1.calcExtrinsicFromJson(cam1ref)
    cams[1] = cam1

    cam2 = joblib.load(cam2Path)
    cam2.calcExtrinsicFromJson(cam2ref)
    cams[2] = cam2

    print("")
    print("Camera 1:")
    print(" - position: \n" + str(cam1.getPosition()))
    print(" - rotation: \n" + str(cam1.getRotationMat()))
    print("")

    print("Camera 2:")
    print(" - position: \n" + str(cam2.getPosition()))
    print(" - rotation: \n" + str(cam2.getRotationMat()))
    print("")

    return cams


def getROI(path, camId):
    """
    Loads the JSON camera parameters and reads the Region of Interest that has been manually set
    """

    # Load json file
    with open(os.path.join(path, "cam{}_references.json".format(camId))) as f:
        data = f.read()

    # Remove comments
    pattern = re.compile('/\*.*?\*/', re.DOTALL | re.MULTILINE)
    data = re.sub(pattern, ' ', data)

    # Parse json
    data = json.loads(data)

    # Convert to numpy arrays
    x_coords = []
    y_coords = []

    for i, entry in enumerate(data):
        x_coords.append(int(entry["camera"]["x"]))
        y_coords.append(int(entry["camera"]["y"]))

    tl = np.asarray([np.min(x_coords), np.min(y_coords)], dtype=int)
    br = np.asarray([np.max(x_coords), np.max(y_coords)], dtype=int)

    return tl, br


def getTankROI(path, camId):
    """
    Loads the JSON camera parameters and reads the Region of Interest that has been manually set
    """

    # Load json file
    with open(os.path.join(path, "cam{}_references.json".format(camId))) as f:
        data = f.read()

    # Remove comments
    pattern = re.compile('/\*.*?\*/', re.DOTALL | re.MULTILINE)
    data = re.sub(pattern, ' ', data)

    # Parse json
    data = json.loads(data)

    tl = int(data[0]["camera"]["x"]), int(data[0]["camera"]["y"])
    tr = int(data[1]["camera"]["x"]), int(data[1]["camera"]["y"])
    br = int(data[2]["camera"]["x"]), int(data[2]["camera"]["y"])
    bl = int(data[3]["camera"]["x"]), int(data[3]["camera"]["y"])

    # 左上，左下 的x
    xmin = min(tl[0], bl[0])
    # 左上，右上 的y
    ymin = min(tl[1], tr[1])
    # 右上，右下 的x
    xmax = max(tr[0], br[0])
    # 左下，右下 的y
    ymax = max(bl[1], br[1])

    return xmin, ymin, xmax, ymax


def load_EXP_region_pos_setting(config_floder, camNO):
    config = readConfig(config_floder)
    c = config['ExpArea']
    region_pos = c.get(f'{camNO}_area')
    region_dict = {}

    for iline in region_pos.split("\n"):
        region_name, region_tl_x, region_tl_y, region_br_x, region_br_y = iline.split(',')
        region_dict[region_name] = (int(region_tl_x), int(region_tl_y), int(region_br_x), int(region_br_y))
    return region_dict


def load_dist_map(config_floder, camId, RegionName):
    config = readConfig(config_floder)
    region_no = RegionName.split("_")[0]
    c = config['Aquarium']
    if camId == 1:
        ratio = c.get(f'top_ratio_{region_no}')
    else:
        ratio = c.get(f'side_ratio_{region_no}')
    return float(ratio)


def load_EXP_region_name(config_floder):
    config = readConfig(config_floder)
    c = config['ExpArea']
    region_name = c.get(f'exp_list')
    return region_name.split(',')


def load_Video_start_time(config_floder, video_name, time_part='VideoStartTime'):
    config = readConfig(config_floder)
    c = config[time_part]
    time_list = c.get(video_name)
    print(video_name)
    if time_list is not None:
        start_time = time_list.split()
        return start_time
    else:
        return []


def load_Cam_list(config_floder):
    config = readConfig(config_floder)
    c = config['ExpArea']
    region_name = c.get(f'cam_list')
    return region_name.split(',')


def load_time_pos_setting(config_floder, camId):
    config = readConfig(config_floder)
    c = config['ExpArea']
    time_pos = c.get(f'{camId}_time_pos')
    t_tl_x, t_tl_y, t_br_x, t_br_y = time_pos.split(',')
    return int(t_tl_y), int(t_br_y), int(t_tl_x), int(t_br_x)


def applyROIBBs(bboxes, tl, br):
    """
    Checks and keeps the detected bounding boxes only if they are fully within the ROI.

    Input:
        bboxes: List of bbox tuples

    Output:
        roi_bboxes: List of bbox tuples within ROI
    """

    if bboxes is None:
        return None

    roi_bboxes = []

    for bbox in bboxes:
        if bbox[0] >= tl[0] and bbox[1] >= tl[1] and bbox[2] <= br[0] and bbox[3] <= br[1]:
            roi_bboxes.append(bbox)

    if len(roi_bboxes) == 0:
        roi_bboxes = None

    return roi_bboxes


# https://www.jb51.net/article/164697.htm
def calAngle(v1, v2):
    '''
    计算两个向量之间的夹角
    AB = [1,-3,5,-1]
    CD = [4,1,4.5,4.5]
    ang1 = angle(AB, CD)
    :param v1: [Ax, Ay, Bx, By]
    :param v2: [Cx, Cy, Dx, Dy]
    :return:
    '''
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def calGIOU(boxes1, boxes2):
    '''
    cal GIOU of two boxes or batch boxes
    such as: (1)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[15,15,25,25]])
            boxes2 = np.asarray([[5,5,10,10]])
            and res is [-0.49999988  0.25       -0.68749988]
            (2)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            boxes2 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            and res is [1. 1. 1.]
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    # cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # ===========cal IOU=============#
    # cal Intersection
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1Area + boxes2Area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    # ===========cal enclose area for GIOU=============#
    enclose_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = np.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # cal GIOU
    gious = ious - 1.0 * (enclose_area - union_area) / enclose_area

    return gious


def getRealTime(ID_T, part):
    time_date = time.strptime(ID_T[4:], "%Y%m%d%H%M%S")
    unix_time = time.mktime(time_date)
    t = datetime.datetime.fromtimestamp(unix_time + (int(part) - 1) * 60).strftime("%Y-%m-%d-%H-%M-%S")
    process_result = str(t)
    return process_result


def getIPAddrs():
    if (platform.system() == 'Windows'):
        # 获取本机计算机名称
        hostname = socket.gethostname()
        # 获取本机ip
        ip_list = socket.gethostbyname(hostname)
        if type(ip_list) is str:
            ip_list = [ip_list]

    elif (platform.system() == 'Linux'):
        p = Popen("hostname -I", shell=True, stdout=PIPE)
        data = p.stdout.read()  # 获取命令输出内容
        data = str(data, encoding='UTF-8')  # 将输出内容编码成字符串
        ip_list = data.split(' ')  # 用空格分隔输出内容得到包含所有IP的列表
        if "\n" in ip_list:  # 发现有的系统版本输出结果最后会带一个换行符
            ip_list.remove("\n")
    return ip_list


def getBaseDir(ip_addr):
    if (platform.system() == 'Windows'):
        # 获取本机电脑名
        # if '10.2.173.213' in ip_addr:
        base_dir = "E:\\data\\3D_pre"
    elif (platform.system() == 'Linux'):
        if '10.2.151.129' in ip_addr:
            base_dir = "/home/data/HJZ/zef/"
        elif '10.2.151.128' in ip_addr:
            base_dir = "/home/huangjinze/code/data/zef/"
        elif '10.2.151.127' in ip_addr:
            base_dir = "/home/huangjinze/code/data/zef/"
    return base_dir


def get_tankDragMap(root_path, view_no=3, time_period=12):
    '''

    :param root_path:
    :return: {
        ppd1: {
            D1_T1: {
                expose_T1: [file1, file2, file3,....],
                expose_T2: [file1, file2, file3,....],
            }
        }
    }
    '''
    for dayTank, drag in tank_drag_map.items():
        video_path = os.path.join(root_path, dayTank, 'cut_video')
        try:
            cut_video_file = sorted([_.split(".")[0] for _ in os.listdir(video_path) if _.endswith('avi')])
        except:
            continue
        drag_tank_map[drag][dayTank] = {}
        # for idx, video_name in enumerate(cut_video_file):
        #     drag_tank_map[drag][dayTank][0].append(video_name)
        # if dayTank == 'D1_T5':
        #     drag_tank_map[drag][dayTank][0] = ['2021_10_11_21_40_00_ch13', '2021_10_11_21_40_00_ch14', '2021_10_11_21_40_00_ch16']
        #     for idx, video_name in enumerate(cut_video_file):
        #         if idx // (3*5)+1 not in drag_tank_map[drag][dayTank]:
        #             drag_tank_map[drag][dayTank][idx // (3*5)+1] = [video_name]
        #         else:
        #             drag_tank_map[drag][dayTank][idx // (3*5)+1].append(video_name)
        # else:
        #     for idx, video_name in enumerate(cut_video_file):
        #         if len(cut_video_file) > 150: # 3分钟划分
        #             if idx // (3*5) not in drag_tank_map[drag][dayTank]:
        #                 drag_tank_map[drag][dayTank][idx // (3*5)] = [video_name]
        #             else:
        #                 drag_tank_map[drag][dayTank][idx // (3*5)].append(video_name)
        #         else:
        #             if idx // (3*3) not in drag_tank_map[drag][dayTank]:
        #                 drag_tank_map[drag][dayTank][idx // (3*3)] = [video_name]
        #             else:
        #                 drag_tank_map[drag][dayTank][idx // (3*3)].append(video_name)
    return drag_tank_map


def get_trackFileInfo(root_path, processer, filename):
    '''

    :param root_path:
    :param processer: sortTracker / bg_processer / ...
    :param filename:
    :return: {
                    'root_path': root_path,
                    'filepath': E:\data\3D_pre\D8_T4\sortTracker\1_6PPDQ1ppm\2021_10_23_21_15_59_ch10.csv,
                    'filename': 2021_10_23_21_15_59_ch10,
                    'DayTank': D8_T4,
                    'drag': 6PPDQ1ppm,
                    'region_name': 1_6PPDQ1ppm,
                    'camNO': ch10,
                }
    '''
    infos = []
    for iDayTank, drag in tank_drag_map.items():
        for no in ['1_', '2_', '3_', '4_']:
            trackfloder = os.path.join(root_path, iDayTank, processer, no + drag)
            print(trackfloder)
            if not os.path.exists(trackfloder):
                continue
            trackfiles = os.listdir(trackfloder)
            if filename + ".csv" in trackfiles:
                camNO = filename.split("_")[-1]
                infos.append({
                    'root_path': root_path,
                    'filepath': os.path.join(trackfloder, filename + ".csv"),
                    'filename': filename,
                    'exp_time': '_'.join(filename.split("_")[: -1]),
                    'DayTank': iDayTank,
                    'drag': drag,
                    'region_name': no + drag,
                    'camNO': camNO,
                    'cam_view': camera_id_map[camNO],
                })
    return infos


def get_filname(root_path, drag_name, exposeT):
    trace_filename_list = []
    tank_info = get_tankDragMap(root_path)[drag_name]
    for DayTank, exposeInfo in tank_info.items():
        for iexposeT, file_list in exposeInfo.items():
            if iexposeT == exposeT:
                trace_filename_list.extend(file_list)
    return trace_filename_list
