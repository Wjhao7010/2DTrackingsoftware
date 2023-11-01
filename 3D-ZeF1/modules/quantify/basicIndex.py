import cv2
import numpy as np
import math, sys

sys.path.append(".")
from common.utility import calAngle, calGIOU
import os
import pandas as pd
import argparse
from common.utility import *


def find_aquair_info(fish_pos, aquair_info):
    cX, cY = fish_pos
    # print(aquair_info)
    # print(cX, cY)
    for iexp_region in aquair_info.split('\n'):
        region_name, tl_x, tl_y, br_x, br_y = iexp_region.split(',')
        if ((cX >= float(tl_x)) and (cY >= float(tl_y))) and ((cX <= float(br_x)) and (cY <= float(br_y))):
            return (region_name, float(tl_x), float(tl_y), float(br_x), float(br_y))
    return [None] * 5


class BehaviorIndex(object):
    def __init__(self, track_info, object_feature, relation_feature, window_size=3, view='top'):
        '''
        :param track_info: 需要计算运动指标的总帧数
        :param object_feature: 需要计算运动指标的名称
        :param relation_feature: 各个鱼之间的关系矩阵类型
        :param window_size: 需要考虑的平均帧数，最小为3
        '''
        assert window_size > 2, 'window size is larger than 2'
        assert len(object_feature) > 0, 'object feature such as velocity, angle...'
        # assert len(relation_feature) > 0, 'the way to construct matrix'
        # 检察track_info中的格式
        track_info['width'] = track_info['boxesX2'] - track_info['boxesX1']
        track_info['height'] = track_info['boxesY2'] - track_info['boxesY1']

        # 视角：top， left， right
        self.view = view
        self.aquair_info = c.get(view)
        # 当前轨迹中的track ID
        # 通过~取反，选取不包含数字-1的行
        track_info = track_info[~track_info['track_id'].isin([-1])]

        self.tracks = track_info.copy()
        self.window_size = window_size
        self.object_feature = object_feature
        self.relation_feature = relation_feature

        # 生成当前track中的所有trackid <class 'numpy.ndarray'>
        self.unique_trackid = list(self.tracks['track_id'].unique())
        # 生成当前track中的所有frameid <class 'numpy.ndarray'>
        self.unique_frameid = list(self.tracks['frame_id'].unique())

    def run(self):
        '''
        计算任意三帧之间的运动指标信息，为字典格式
        {
            frame1: {
                    velocity: {
                            trackid1: xxx,
                            trackid2: xxx,
                            trackid3: xxx
                        },
                    behavior_type: {
                            trackid1: xxx,
                            trackid2: xxx,
                            trackid3: xxx
                    },

                }
        }
        :return:
        '''
        index_info = {}
        for idx_frameid in range(len(self.unique_frameid) - self.window_size + 1):
            if (idx_frameid + 1) % 100 == 0:
                print(f"processing frame_id {self.unique_frameid[idx_frameid]} to id: "
                      f"{self.unique_frameid[idx_frameid + self.window_size]}")
            # 存储每一帧中目标跟踪的信息，可能有多个目标
            # 取出 idx_frameid 需要的偏移量
            frameid_id_list = self.unique_frameid[idx_frameid: idx_frameid + self.window_size]
            # 偏移量对应的行索引
            trackid_Index_id_no = self.tracks['frame_id'].isin(frameid_id_list)
            # 当前帧偏移量 对应的 数据
            track_info = self.tracks[trackid_Index_id_no]

            # 统计trackid出现的次数
            track_id_counts = track_info['track_id'].value_counts()
            # 找出出现次数等于window_size的track id
            trackid_list = track_id_counts[track_id_counts == self.window_size].index.tolist()
            # 保留下trackid的信息
            track_info = track_info[track_info['track_id'].isin(trackid_list)]

            # 计算node的特征表达
            # 选取当前 win_size 中最大的值作为win_size的代表
            index_info[max(frameid_id_list)] = {}

            index_info[max(frameid_id_list)]['CenterX'], index_info[max(frameid_id_list)][
                'CenterY'] = self.getID_CenterPos(
                track_info, trackid_list
            )
            if 'water_level' in self.object_feature and (self.view == 'left' or self.view == 'right'):
                index_info[max(frameid_id_list)]['water_level'] = self.get_WaterLevel(
                    track_info, trackid_list
                )
            if 'distance' in self.object_feature:
                index_info[max(frameid_id_list)]['distance'] = self.getIDDist(
                    track_info, trackid_list
                )
            if 'velocity' in self.object_feature:
                index_info[max(frameid_id_list)]['velocity'] = self.getIDVelocity(
                    track_info, trackid_list
                )
            if 'angle' in self.object_feature:
                index_info[max(frameid_id_list)]['angle'] = self.getIDAngle(
                    track_info, trackid_list
                )
            if 'w_mean' in self.object_feature:
                index_info[max(frameid_id_list)]['w_mean'] = self.getID_WInfo(
                    track_info, trackid_list
                )[0]
            if 'h_mean' in self.object_feature:
                index_info[max(frameid_id_list)]['h_mean'] = self.getID_HInfo(
                    track_info, trackid_list
                )[0]
            if 'giou' in self.object_feature:
                index_info[max(frameid_id_list)]['giou'] = self.getIDGIOU(
                    track_info, trackid_list
                )

            # print(trackid_list)
            if fish_num < 2:
                # print(f"frameid_id_list is {frameid_id_list}")
                # print(f"can not construct graph {trackid_list}")
                continue
            else:
                if 'dispersion_nnd' in self.relation_feature:
                    index_info[max(frameid_id_list)]['dispersion_nnd'] = self.getIDNND(
                        track_info, trackid_list
                    )
                if 'dispersion_delaunay' in self.relation_feature:
                    if len(trackid_list) == 2:
                        index_info[max(frameid_id_list)]['dispersion_delaunay'] = self.getIDNND(
                            track_info, trackid_list
                        )
                    else:
                        index_info[max(frameid_id_list)]['dispersion_delaunay'] = self.getDelaunay(
                            track_info, trackid_list
                        )

            #     panic_index = self.getPanic(index_info, track_list, window_size=self.window_size)
            #     index_info[frameid_3]['panic'] = panic_index

        return index_info

    def getID_CenterPos(self, track_info, trackid_list):
        '''
        :param track_info_list:
        :param trackid_list:
        :return:
        '''
        id_centerX = {}
        id_centerY = {}
        for track_id in trackid_list:
            # 筛选id为 track_id的数据
            track_id_info = track_info[
                track_info['track_id'] == track_id
                ]

            # 计算每个id 的bbox的纵横比的变化率
            id_centerX[track_id] = track_id_info[['centerX']].mean().values[0]
            id_centerY[track_id] = track_id_info[['centerY']].mean().values[0]
        return id_centerX, id_centerY

    def get_WaterLevel(self, track_info, trackid_list, water_level_num=2):
        '''
        :param track_info_list:
        :param trackid_list:
        :return: 水面深度分为三层，
        '''

        id_WaterLevel = {}
        for track_id in trackid_list:
            # 筛选id为 track_id的数据
            track_id_info = track_info[
                track_info['track_id'] == track_id
                ]

            # 计算每个id 的bbox的纵横比的变化率
            id_centerX = track_id_info[['centerX']].mean().values[0]
            id_centerY = track_id_info[['centerY']].mean().values[0]

            region_name, tl_x, tl_y, br_x, br_y = find_aquair_info((id_centerX, id_centerY), self.aquair_info)
            if region_name is None:
                continue
            else:
                level_depth = (br_y - tl_y) / water_level_num
                id_WaterLevel[track_id] = str(int((id_centerY - tl_y) // level_depth))
        return id_WaterLevel

    def getID_WInfo(self, track_info, trackid_list):
        '''
        :param track_info_list:
        :param trackid_list:
        :return:
        '''
        id_w_mean = {}
        id_w_std = {}
        for track_id in trackid_list:
            # 筛选id为 track_id的数据
            track_id_info = track_info[
                track_info['track_id'] == track_id
                ]
            # 计算每个id 的bbox的纵横比的变化率
            diff_id_data = track_id_info[['width']].dropna()
            id_w_std[track_id] = diff_id_data.std().values[0]
            id_w_mean[track_id] = diff_id_data.mean().values[0]
        return id_w_mean, id_w_std

    def getID_HInfo(self, track_info, trackid_list):
        '''
        :param track_info_list:
        :param trackid_list:
        :return:
        '''
        id_h_mean = {}
        id_h_std = {}
        for track_id in trackid_list:
            # 筛选id为 track_id的数据
            track_id_info = track_info[
                track_info['track_id'] == track_id
                ]
            # 计算每个id 的bbox的纵横比的变化率
            diff_id_data = track_id_info[['height']].dropna()
            id_h_mean[track_id] = diff_id_data.mean().values[0]
            id_h_std[track_id] = diff_id_data.std().values[0]
        return id_h_mean, id_h_std

    def getIDDist(self, track_info, trackid_list):
        '''
        :param track_info_list:
        :param trackid_list:
        :return:
        '''
        id_dist = {}
        for track_id in trackid_list:
            # 筛选id为 track_id的数据
            track_id_info = track_info[
                track_info['track_id'] == track_id
                ]
            # 计算相邻两帧之间的差
            diff_id_data = track_id_info[['centerX', 'centerY']].diff(1).dropna()
            if self.view == 'top':
                dist = diff_id_data['centerX'] ** 2 + diff_id_data['centerY'] ** 2
            else:
                dist = diff_id_data['centerY'] ** 2
            id_dist[track_id] = np.sqrt(dist.values).sum()
        return id_dist

    def getIDVelocity(self, track_info, trackid_list):
        '''
        :param track_info_list:
        :param trackid_list:
        :return:
        '''
        id_velocity = {}
        for track_id in trackid_list:
            # 筛选id为 track_id的数据
            track_id_info = track_info[
                track_info['track_id'] == track_id
                ]
            # 计算相邻两帧之间的差
            diff_id_data = track_id_info[['centerX', 'centerY']].diff(1).dropna()
            if self.view == 'top':
                dist = diff_id_data['centerX'] ** 2 + diff_id_data['centerY'] ** 2
            else:
                dist = diff_id_data['centerY'] ** 2
            id_velocity[track_id] = np.sqrt(dist.values).mean()
        return id_velocity

    def getIDAngle(self, track_info, trackid_list):
        id_angle = {}
        for track_id in trackid_list:
            pos = track_info[
                track_info['track_id'] == track_id
                ][['centerX', 'centerY']].values
            angle = 0

            for i in range(pos.shape[0] - 2):
                x1, y1 = pos[i, :]
                x2, y2 = pos[i + 1, :]
                x3, y3 = pos[i + 2, :]
                angle += calAngle([x1, y1, x2, y2], [x2, y2, x3, y3])

            id_angle[track_id] = angle / (pos.shape[0] - 2)
        return id_angle

    def getIDNND(self, frame_info, track_list):
        '''
        计算各个点之间的距离矩阵
        :param frame_info:
        :param track_list:
        :return:
        '''
        DistanceMatrix = pd.DataFrame(
            np.zeros((
                len(track_list),
                len(track_list)
            )), columns=track_list, index=track_list
        )

        unique_frameid = list(frame_info['frame_id'].unique())
        for idframe in unique_frameid:
            iDistanceMatrix = pd.DataFrame(
                np.zeros((
                    len(track_list),
                    len(track_list)
                )), columns=track_list, index=track_list
            )

            iframeid_Index = frame_info['frame_id'] == idframe
            # 选出同一帧的所有跟踪数据
            iframe_info = frame_info[iframeid_Index]

            for i in range(len(track_list)):
                for j in range(i + 1, len(track_list)):
                    # 这里要获取trackid为i的值在iframe_info 中对应的索引i_trackid_index,
                    # j_trackid_index，后续才能进行计算
                    # iframe_trackid 对应值为i所对应的行索引为：

                    i_trackid_index = iframe_info[
                        iframe_info['track_id'] == track_list[i]
                        ].index.values[0]
                    j_trackid_index = iframe_info[
                        iframe_info['track_id'] == track_list[j]
                        ].index.values[0]
                    # print('track_id ({0}){1} -- dist -- ({2}){3}'.format(
                    #     i_trackid_index, track_list[i],
                    #     j_trackid_index, track_list[j]
                    #     )
                    # )
                    # 计算第i个值与第j个值的距离
                    distance = np.sqrt(
                        (iframe_info.loc[j_trackid_index, 'centerX'] - iframe_info.loc[
                            i_trackid_index, 'centerX']) ** 2 +
                        (iframe_info.loc[j_trackid_index, 'centerY'] - iframe_info.loc[i_trackid_index, 'centerY']) ** 2
                    ) / max(iframe_info.loc[i_trackid_index, 'width'], iframe_info.loc[i_trackid_index, 'height'])
                    # 找到第i个元素和第j个元素在iframe_trackid列表中的索引(i_track_index, j_track_index)
                    # ，根据索引值对 DistanceMatrix 进行赋值
                    i_track_index = iframe_info.loc[i_trackid_index, 'track_id']
                    j_track_index = iframe_info.loc[j_trackid_index, 'track_id']
                    iDistanceMatrix[i_track_index][j_track_index] = distance
                    iDistanceMatrix[j_track_index][i_track_index] = distance
            DistanceMatrix += iDistanceMatrix
        return DistanceMatrix / len(unique_frameid)

    def getDelaunay(self, track_info, track_list):
        # https://www.cnblogs.com/zeroing0/p/13657685.html
        # Check if a point is insied a rectangle
        def rect_contains(rect, point):
            if point[0] < rect[0]:
                return False
            elif point[1] < rect[1]:
                return False
            elif point[0] > rect[2]:
                return False
            elif point[1] > rect[3]:
                return False
            return True

        # 距离矩阵
        DistanceMatrix = pd.DataFrame(
            np.zeros((
                len(track_list),
                len(track_list)
            )), columns=track_list, index=track_list
        )
        # 维护一个字典：{(x,y): track_id}
        pos_trackid = {}

        # 按照trackid分组求均值
        frame_info = track_info.groupby('track_id').mean()

        # Create an instance of Subdiv2d
        # 图片分辨率大小为
        img_width = 1920
        img_height = 1080
        rect = (0, 0, img_width, img_height)
        subdiv = cv2.Subdiv2D(rect)
        # Create an array of points

        points = []

        for track_id in frame_info.index.tolist():
            # trackid 是行索引，根据行索引选择值
            x, y = frame_info.loc[track_id]['centerX'], frame_info.loc[track_id]['centerY']
            points.append((int(x), int(y)))
            pos_trackid[(int(x), int(y))] = track_id

        # Insert points into subdiv
        for p in points:
            subdiv.insert(p)

        trangleList = subdiv.getTriangleList()
        for t in trangleList:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            if (
                    rect_contains(rect, pt1)
                    and rect_contains(rect, pt2)
                    and rect_contains(rect, pt3)
            ):
                # https://www.sohu.com/a/209558642_797291

                p1_ind = pos_trackid[(int(pt1[0]), int(pt1[1]))]
                p2_ind = pos_trackid[(int(pt2[0]), int(pt2[1]))]
                p3_ind = pos_trackid[(int(pt3[0]), int(pt3[1]))]
                edge_p1p2 = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) / max(
                    frame_info.loc[p1_ind, 'width'], frame_info.loc[p1_ind, 'height'])
                edge_p2p3 = math.sqrt((pt2[0] - pt3[0]) ** 2 + (pt2[1] - pt3[1]) ** 2) / max(
                    frame_info.loc[p1_ind, 'width'], frame_info.loc[p1_ind, 'height'])
                edge_p1p3 = math.sqrt((pt1[0] - pt3[0]) ** 2 + (pt1[1] - pt3[1]) ** 2) / max(
                    frame_info.loc[p1_ind, 'width'], frame_info.loc[p1_ind, 'height'])

                DistanceMatrix[p1_ind][p2_ind] = edge_p1p2
                DistanceMatrix[p2_ind][p1_ind] = edge_p1p2

                DistanceMatrix[p1_ind][p3_ind] = edge_p1p3
                DistanceMatrix[p3_ind][p1_ind] = edge_p1p3

                DistanceMatrix[p2_ind][p3_ind] = edge_p2p3
                DistanceMatrix[p3_ind][p2_ind] = edge_p2p3

        return DistanceMatrix

    def getIDBehavior(self, track_info, track_list):
        '''
        :param track_info_list:
        :param trackid_list:
        :return:
        '''
        id_behavior = {}
        for track_id in track_list:
            # 筛选id为 track_id的数据
            track_id_info = track_info[
                track_info['track_id'] == track_id
                ]
            # 计算每个id 的bbox的纵横比的变化率
            node_label = track_id_info[['behavior_type']].value_counts().index.values[0][0]
            id_behavior[track_id] = node_label
        return id_behavior

    def getIDGIOU(self, track_info, track_list):
        '''
        :param track_info_list:
        :param trackid_list:
        :return:
        '''
        id_giou = {}
        for track_id in track_list:
            # 筛选id为 track_id的数据
            track_id_info = track_info[
                track_info['track_id'] == track_id
                ]
            # 计算每个id 的bbox的纵横比的变化率
            pre_box = track_id_info[['boxesX1', 'boxesY1', 'boxesX2', 'boxesY2']].shift(-1).dropna()
            next_box = track_id_info[['boxesX1', 'boxesY1', 'boxesX2', 'boxesY2']].shift(1).dropna()
            id_giou[track_id] = np.mean(calGIOU(pre_box.values, next_box.values))
        return id_giou

    # 其实还可以从相互作用力方面进行构图
    def getVisualField(self, track_info, track_list, motion_angle=150):
        '''
        计算视野，计算每个个体与其他个体运动方向之间的夹角
        :param track_info:
        :param track_list:
        :return:
        '''
        VisualMatrix = pd.DataFrame(
            np.zeros((
                len(track_list),
                len(track_list)
            )), columns=track_list, index=track_list
        )
        # 计算第i条鱼与第j条鱼之间的夹角向量
        vec_id = {}
        for track_id in track_list:
            pos = track_info[
                track_info['track_id'] == track_id
                ][['centerX', 'centerY']].values
            # 取最后两个点作为计算视野的向量
            vec_id[track_id] = list(pos[-1, :]) + list(pos[-2, :])

        for id_src, vec_src in vec_id.items():
            # 对于视线出发者来说
            for id_tgt, vec_tgt in vec_id.items():
                # 对于被观察者来说
                dx1 = vec_src[2] - vec_src[0]
                dy1 = vec_src[3] - vec_src[1]
                dx2 = vec_tgt[2] - vec_tgt[0]
                dy2 = vec_tgt[3] - vec_tgt[1]
                # 计算id_src与水平轴的夹角，记为angle1，如果为负数，则转化为正数
                angle1 = math.atan2(dy1, dx1)
                angle1 = int(angle1 * 180 / math.pi)
                angle1 = angle1 + 360 if angle1 < 0 else angle1

                # 计算id_tat与水平轴的夹角，记为angle2，如果为负数，则转化为正数
                angle2 = math.atan2(dy2, dx2)
                angle2 = int(angle2 * 180 / math.pi)
                angle2 = angle2 + 360 if angle2 < 0 else angle2
                # 计算范围
                range1 = angle1 - motion_angle
                range2 = angle1 + motion_angle
                # rang1 和 range2在正常范围内，即150-210之间
                if range1 > 0 and range2 < 360:
                    if angle2 >= range1 and angle2 <= range2:
                        VisualMatrix[id_src][id_tgt] = 1
                    else:
                        VisualMatrix[id_src][id_tgt] = 0
                else:
                    # range1为负数，-30 则需要转为正数 330，此时范围为330-360
                    range1 = range1 + 360 if range1 < 0 else range1
                    # range2为大于360的数，390 则需要转为小于360的数，此时范围为0-30
                    range2 = range2 - 360 if range1 > 360 else range2

                    new_rg1 = min(range1, range2)
                    new_rg2 = max(range1, range2)
                    # 此时的有效范围：0 到 new_rg1， new_rg2 到360
                    # 此时的无效范围：new_rg1 到 new_rg2
                    if angle2 > new_rg1 and angle2 < new_rg2:
                        VisualMatrix[id_src][id_tgt] = 0
                    else:
                        VisualMatrix[id_src][id_tgt] = 1

        return VisualMatrix

    # ====================================== fish_group information =============================#
    def getDispersion(self):
        '''
        构造一个3为矩阵，第0,1维度存储某一帧中的距离信息，第2个维度对应frame维度
        最后在frame维度求均值，压缩为一个2维矩阵
        计算过程：
        1. 计算每帧中目标之间的两两距离
        2.
        :return:
        '''
        DistanceMatrix = np.zeros(
            (
                len(self.unique_frameid),
                len(self.unique_trackid),
                len(self.unique_trackid)
            )
        )
        for idframe in self.unique_frameid:
            iframeid_Index = self.tracks['frame_id'] == idframe
            iframe_info = self.tracks[iframeid_Index]

            if iframe_info.shape[0] >= 1:
                # 当前帧中所包含的trackid
                iframe_trackid = list(iframe_info['track_id'].unique())

                for i in range(len(iframe_trackid)):
                    for j in range(i + 1, len(iframe_trackid)):
                        # 这里要获取trackid为i的值在iframe_info 中对应的索引i_trackid_index,
                        # j_trackid_index，后续才能进行计算
                        # iframe_trackid 对应值为i所对应的行索引为：

                        i_trackid_index = iframe_info[
                            iframe_info['track_id'] == iframe_trackid[i]
                            ].index.values[0]
                        j_trackid_index = iframe_info[
                            iframe_info['track_id'] == iframe_trackid[j]
                            ].index.values[0]
                        print('track_id ({0}){1} -- dist -- ({2}){3}'.format(
                            i_trackid_index, iframe_trackid[i],
                            j_trackid_index, iframe_trackid[j]
                        )
                        )
                        # 计算第i个值与第j个值的距离
                        distance = np.sqrt(
                            (iframe_info.loc[i_trackid_index, 'centerX'] - iframe_info.loc[
                                j_trackid_index, 'centerX']) ** 2 +
                            (iframe_info.loc[i_trackid_index, 'centerY'] - iframe_info.loc[
                                j_trackid_index, 'centerY']) ** 2
                        ) / max(iframe_info.loc[i_trackid_index, 'width'], iframe_info.loc[i_trackid_index, 'height'])
                        # 找到第i个元素和第j个元素在iframe_trackid列表中的索引(i_track_index, j_track_index)
                        # ，根据索引值对 DistanceMatrix 进行赋值
                        i_track_index = self.unique_trackid.index(iframe_info.loc[i_trackid_index, 'track_id'])
                        j_track_index = self.unique_trackid.index(iframe_info.loc[j_trackid_index, 'track_id'])
                        frame_index = self.unique_frameid.index(idframe)
                        DistanceMatrix[frame_index][i_track_index][j_track_index] = distance

            else:
                continue
        return DistanceMatrix

    def getGroupPath(self):
        all_path = 0.0
        for idtrack in self.unique_trackid:
            # 获取tracks中trackid值为id的索引
            # print('current track id is :', id)
            itrackid_Index = self.tracks['track_id'] == idtrack
            # 先对视频帧按时间进行排序
            itrack_info = self.tracks[itrackid_Index].sort_values(by='frame_id')

            if itrack_info.shape[0] > 1:
                path_pos = itrack_info[['centerX', 'centerY']]
                # 在行维度进行一阶差分
                path_pos_diff = path_pos.diff(1, axis=0)
                path_pos_diff.dropna(inplace=True)

                moving_dist = 0.0
                for x, y in path_pos_diff.values:
                    moving_dist += np.sqrt(x ** 2 + y ** 2)
                # 计算当前id路径在总路径中的比例
                all_path += moving_dist
            else:
                continue
        return all_path


def merge_view_index(IndexValue, view):
    exp = {}
    aquair_info = c.get(view)
    for iframe_no, indexes_value in IndexValue.items():
        # indexes_value: {
        #   centerX: {
        #             track_id1: XXX,
        #             track_id2: XXX,
        #             track_id3: XXX,
        #       }
        #   velocity: {
        #             track_id1: XXX,
        #             track_id2: XXX,
        #             track_id3: XXX,
        #       }
        #    ....
        # }
        track_ids = list(indexes_value['CenterX'].keys())
        for track_id in track_ids:
            region_name = find_aquair_info(
                (indexes_value['CenterX'][track_id], indexes_value['CenterY'][track_id]),
                aquair_info
            )[0]
            if region_name is None:
                print(indexes_value['CenterX'][track_id], indexes_value['CenterY'][track_id])
                print("????????????????")
                continue
            # region_name = "2_CK", "1", "{box_np}_{potency}"....
            if region_name not in exp:
                exp[region_name] = {f'{view}-' + i: [] for i in edge_feature + node_feature}
                # exp[region_name]:
                # exp{
                #   "ck": {
                #       velocity: [1,2,3,...],
                #       angle: [1,2,3,...],
                #   }
                #   "1": {
                #       velocity: [1,2,3,...],
                #       angle: [1,2,3,...],
                #   }
                # }
            for index in edge_feature + node_feature:
                if index in list(indexes_value.keys()):
                    exp[region_name][f'{view}-' + index].append(indexes_value[index][track_id])
                else:
                    exp[region_name][f'{view}-' + index].append(-1)
    return exp


def statistic_individual_index(index_value):
    # ================================== 移动距离 ================================== #
    Index_dict = {}
    distance = sum(index_value['top-distance'])

    # ================================== 速度 ================================== #
    velocity = distance / stat_interval / (len(index_value['top-distance']) / 25)

    # ================================== 加速度 ================================== #
    # y_as, yas_inter = np.histogram(index_value['top-velocity'], bins=5, range=(0, 10))
    # yas = y_as/len(index_value['top-velocity'])
    # Index_dict = {}
    # for interval, _as in zip(yas_inter, yas):
    #     Index_dict[f'AccSpeed_{str(interval)}'] = [_as]

    # ================================== 转向角 ================================== #
    # y_aa, yaa_inter = np.histogram(index_value['top-angle'], bins=9, range=(0, 180))
    # yaa = y_aa / len(index_value['top-angle'])
    #
    # for interval, a in zip(yaa_inter, yaa):
    #     Index_dict[f'Angle_{str(interval)}'] = [a]

    # ================================== 静止时间 ================================== #
    stop_time = len([_ for _ in index_value['top-velocity'] if _ < 5]) / len(index_value['top-distance'])

    # ================================== 在顶部和底部的时间 ================================== #
    _water_level_ = [str(_) for _ in range(2)]
    try:
        top_time = len([_ for _ in index_value['left-water_level'] if _ == _water_level_[0]]) / len(
            index_value['left-water_level'])
        bottom_time = len([_ for _ in index_value['left-water_level'] if _ == _water_level_[-1]]) / len(
            index_value['left-water_level'])
    except:
        top_time = len([_ for _ in index_value['right-water_level'] if _ == _water_level_[0]]) / len(
            index_value['right-water_level'])
        bottom_time = len([_ for _ in index_value['right-water_level'] if _ == _water_level_[-1]]) / len(
            index_value['right-water_level'])

    Index_dict['distance'] = [distance]
    Index_dict['velocity'] = [velocity]
    Index_dict['stop_time'] = [stop_time]
    Index_dict['top_time'] = [top_time]
    Index_dict['bottom_time'] = [bottom_time]
    return Index_dict


def statistic_joint_index(index_value):
    # ================================== 加速度与转向角 ================================== #
    rows = 25
    cols = 18
    joint_dist = np.zeros((int(rows), int(cols)))
    for iav, iaa in zip(index_value['top-velocity'], index_value['top-angle']):
        row_index = iav // 1 if iav < 25 else 24
        col_index = (iaa - 1) // 10 if iaa == 180 else iaa // 10
        joint_dist[math.ceil(row_index)][math.ceil(col_index)] += 1
    return joint_dist


def run(data_top, data_right, data_left, joint_dist=False):
    # print(data)
    all_exp = {_.split(",")[0]: {} for _ in c.get('top').split('\n')}

    for data, view in zip([data_top, data_left, data_right], ['top', 'left', 'right']):
        print(f"view:{view}")
        data.columns = header_names
        # 计算基础指标
        IndexValue = BehaviorIndex(data, node_feature, edge_feature, view=view).run()
        # 因为是多视角的，所以需要合并两个视角的指标
        exp = merge_view_index(IndexValue, view)

        for iregion in all_exp.keys():
            if iregion in exp:
                all_exp[iregion].update(exp[iregion])

    result = {}

    if joint_dist:
        assert 'velocity' in node_feature and 'angle' in node_feature
        joint_res = {}
        for exp_region, index_value in all_exp.items():
            joint_res[exp_region] = statistic_joint_index(index_value)
            result[exp_region] = statistic_individual_index(index_value)
        return result, joint_res
    else:
        for exp_region, index_value in all_exp.items():
            result[exp_region] = statistic_individual_index(index_value)
        return result, None


if __name__ == '__main__':

    edge_feature = ['dispersion_delaunay']
    node_feature = [
        'velocity', 'angle',
        # 'w_mean', 'w_std', 'h_mean', 'h_std',
        'giou', 'distance', 'water_level', 'CenterX', 'CenterY'
    ]
    header_names = ['frame_id', 'track_id', 'boxesX1', 'boxesY1', 'boxesX2', 'boxesY2', 'centerX', 'centerY']
    ap = argparse.ArgumentParser()

    ap.add_argument("-tr", "--track_floder",
                    default="E:\\data/3D_pre/exp_pre/sort_tracker/",
                    help="Path to folder")

    ap.add_argument("-tid", "--t_ID", default="D01")
    ap.add_argument("-lid", "--l_ID", default="D02")
    ap.add_argument("-rid", "--r_ID", default="D04")
    ap.add_argument("-no", "--no", default="222-223")

    ap.add_argument("-o", "--outputPath", default="E:\\data\\3D_pre/exp_pre/indicators/")
    ap.add_argument("-if", "--indicator_file",
                    default="NO_RegionName.csv",
                    help="'RegionName' will replace by region name")

    ap.add_argument("-cp", "--configPath", default="E:\\data/3D_pre/exp_pre")
    ap.add_argument("-fn", "--fish_num", default=1)
    # ap.add_argument("-si", "--stat_interval", default=15)

    args = vars(ap.parse_args())

    outputPath = args["outputPath"]
    indicator_file = args["indicator_file"]

    track_floder = args["track_floder"]
    file_no_range = args["no"]
    t_ID = args["t_ID"]
    r_ID = args["r_ID"]
    l_ID = args["l_ID"]

    start_no, end_no = file_no_range.split("-")
    fish_num = int(args["fish_num"])
    stat_interval = int(end_no) - int(start_no)
    configPath = args["configPath"]

    config = readConfig(configPath)
    c = config['ExpArea']

    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    track_files_tops = sorted(os.listdir(os.path.join(track_floder, t_ID)))
    track_file_rights = sorted(os.listdir(os.path.join(track_floder, r_ID)))
    track_file_lefts = sorted(os.listdir(os.path.join(track_floder, l_ID)))

    merge_res = {}
    stat_top = pd.DataFrame()
    stat_left = pd.DataFrame()
    stat_right = pd.DataFrame()
    for file_no in range(int(start_no), int(end_no)):
        print("=" * 18)
        print(file_no)
        print("=" * 18)
        track_file_top = os.path.join(track_floder, t_ID, track_files_tops[file_no])
        track_file_right = os.path.join(track_floder, r_ID, track_file_rights[file_no])
        track_file_left = os.path.join(track_floder, l_ID, track_file_lefts[file_no])

        indicator_file = os.path.join(outputPath, indicator_file)

        # 一分钟中所有的运动轨迹

        data_top = pd.read_csv(track_file_top, index_col=None)
        data_right = pd.read_csv(track_file_right, index_col=None)
        data_left = pd.read_csv(track_file_left, index_col=None)

        stat_top = pd.concat([stat_top, data_top])
        stat_left = pd.concat([stat_left, data_left])
        stat_right = pd.concat([stat_right, data_right])

    result, joint_stat = run(stat_top, stat_right, stat_left, joint_dist=True)

    # print(merge_res)
    for RegionName, index_values in result.items():
        save_indicator_file = indicator_file.replace("RegionName", RegionName)
        sin_indicator_file = save_indicator_file.replace("NO", f"{start_no}-{end_no}")
        print(f"save data in to {sin_indicator_file}")
        pd.DataFrame.from_dict(index_values).to_csv(sin_indicator_file)

    for RegionName, index_values in joint_stat.items():
        save_indicator_file = indicator_file.replace("RegionName", RegionName)
        sin_indicator_file = save_indicator_file.replace("NO", f"{start_no}-{end_no}-joint")
        print(f"save data in to {sin_indicator_file}")
        index_name = ['v' + str(_) for _ in range(25)]
        col_name = ['a' + str(_) for _ in range(0, 180, 10)]
        pd.DataFrame(index_values, index=index_name, columns=col_name).to_csv(sin_indicator_file)
