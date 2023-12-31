import pandas as pd
import os
import shutil
import sys

sys.path.append("../../")
sys.path.append(".")
import time
from common.utility import writeConfig

sample_map = {
    'D1_T1': [
        # 确定开始后的5min，每隔4分钟采一次样
        ('2021-10-11 21:45:00', '2021-10-11 21:56:00', 'ch01_20211011211110.mp4'),
        ('2021-10-11 21:57:00', '2021-10-11 22:28:00', 'ch01_20211011215622.mp4'),
        ('2021-10-11 22:29:00', '2021-10-11 23:13:00', 'ch01_20211011222823.mp4'),
        ('2021-10-11 23:14:00', '2021-10-11 23:58:00', 'ch01_20211011231326.mp4'),
        ('2021-10-11 23:59:00', '2021-10-12 00:43:00', 'ch01_20211011235828.mp4'),
        ('2021-10-12 00:44:00', '2021-10-12 01:28:00', 'ch01_20211012004329.mp4'),
        ('2021-10-12 01:29:00', '2021-10-12 02:13:00', 'ch01_20211012012832.mp4'),
        ('2021-10-12 02:14:00', '2021-10-12 02:20:00', 'ch01_20211012021334.mp4'),
        ('2021-10-12 02:21:00', '2021-10-12 09:50:00', 'ch01_20211012022040.mkv'),

        ('2021-10-11 21:45:00', '2021-10-11 21:56:00', 'ch02_20211011211141.mp4'),
        ('2021-10-11 21:57:00', '2021-10-11 22:03:00', 'ch02_20211011215658.mp4'),
        ('2021-10-11 22:04:00', '2021-10-11 22:48:00', 'ch02_20211011220319.mp4'),
        ('2021-10-11 22:49:00', '2021-10-11 23:33:00', 'ch02_20211011224821.mp4'),
        ('2021-10-11 23:34:00', '2021-10-12 00:18:00', 'ch02_20211011233323.mp4'),
        ('2021-10-12 00:19:00', '2021-10-12 01:03:00', 'ch02_20211012001824.mp4'),
        ('2021-10-12 01:04:00', '2021-10-12 01:48:00', 'ch02_20211012010327.mp4'),
        ('2021-10-12 01:49:00', '2021-10-12 02:00:00', 'ch02_20211012014829.mp4'),
        ('2021-10-12 02:01:00', '2021-10-12 09:50:00', 'ch02_20211012020037.mkv'),

        ('2021-10-11 21:45:00', '2021-10-12 02:01:00', 'ch03_20211011211320.mp4'),
        ('2021-10-12 02:02:00', '2021-10-12 09:50:00', 'ch03_20211012020141.mp4'),
    ],
    'D1_T2': [
        ('2021-10-11 21:46:00', '2021-10-12 17:05:00', 'ch04_20211011211405.mp4'),
        ('2021-10-12 17:06:00', '2021-10-12 09:46:00', 'ch04_20211012170521.mp4'),

        ('2021-10-11 21:46:00', '2021-10-12 00:41:00', 'ch05_20211011211501.mp4'),
        ('2021-10-12 00:42:00', '2021-10-12 03:12:00', 'ch05_20211012004155.mp4'),
        ('2021-10-12 03:13:00', '2021-10-12 07:44:00', 'ch05_20211012031248.mp4'),
        ('2021-10-12 07:45:00', '2021-10-12 09:46:00', 'ch05_20211012074458.mp4'),

        ('2021-10-11 21:46:00', '2021-10-12 01:08:00', 'ch06_20211011211524.mp4'),
        ('2021-10-12 01:09:00', '2021-10-12 02:22:00', 'ch06_20211012010814.mp4'),
        ('2021-10-12 02:23:00', '2021-10-12 04:58:00', 'ch06_20211012022223.mp4'),
        ('2021-10-12 04:59:00', '2021-10-12 07:14:00', 'ch06_20211012045838.mp4'),
        ('2021-10-12 07:15:00', '2021-10-12 09:31:00', 'ch06_20211012071429.mp4'),
        ('2021-10-12 09:32:00', '2021-10-12 09:46:00', 'ch06_20211012093101.mp4'),
    ],
    'D1_T3': [
        ('2021-10-11 21:44:00', '2021-10-12 02:23:00', 'ch07_20211011211553.mp4'),
        ('2021-10-12 02:24:00', '2021-10-12 09:44:00', 'ch07_20211012022338.mp4'),

        ('2021-10-11 21:44:00', '2021-10-12 02:18:00', 'ch08_20211011211710.mp4'),
        ('2021-10-12 02:19:00', '2021-10-12 09:44:00', 'ch08_20211012021845.mp4'),

        ('2021-10-11 21:44:00', '2021-10-12 02:28:00', 'ch09_20211011211710.mp4'),
        ('2021-10-12 02:29:00', '2021-10-12 09:44:00', 'ch09_20211012022843.avi'),
    ],
    'D1_T4': [
        ('2021-10-11 21:29:00', '2021-10-12 02:01:00', 'ch10_20211011211738.mp4'),
        ('2021-10-12 02:02:00', '2021-10-12 09:29:00', 'ch10_20211012020142.mp4'),

        ('2021-10-11 21:29:00', '2021-10-12 02:03:37', 'ch11_20211011212301.mp4'),
        ('2021-10-12 02:04:00', '2021-10-12 09:29:00', 'ch11_20211012020337.mp4'),

        ('2021-10-11 21:29:00', '2021-10-12 02:02:00', 'ch12_20211011212313.mp4'),
        ('2021-10-12 02:02:00', '2021-10-12 09:29:00', 'ch12_20211012020154.mp4'),
    ],
    'D1_T5': [
        ('2021-10-11 21:30:00', '2021-10-12 02:26:00', 'ch13_20211011212340.mp4'),
        ('2021-10-12 02:27:00', '2021-10-12 09:30:00', 'ch13_20211012022645.mp4'),
        ('2021-10-11 21:30:00', '2021-10-12 02:24:00', 'ch14_20211011212403.mp4'),
        ('2021-10-12 02:25:00', '2021-10-12 09:30:00', 'ch14_20211012022441.mp4'),
        ('2021-10-11 21:30:00', '2021-10-12 02:19:00', 'ch16_20211011212452.mp4'),
        ('2021-10-12 02:20:00', '2021-10-12 09:30:00', 'ch16_20211012021942.mp4'),
    ],

    'D2_T1': [
        ('2021-10-12 21:08:00', '2021-10-13 09:08:00', 'ch01_20211012200000.mp4'),
        ('2021-10-12 21:08:00', '2021-10-13 06:41:00', 'ch02_20211012200000.mp4'),
        ('2021-10-13 06:42:00', '2021-10-13 09:08:00', 'ch02_20211013064146.mp4'),
        ('2021-10-12 21:08:00', '2021-10-13 09:08:00', 'ch03_20211012200000.mp4'),
    ],
    'D2_T2': [
        ('2021-10-12 21:05:00', '2021-10-13 09:05:00', 'ch04_20211012200000.mp4'),
        ('2021-10-12 21:05:00', '2021-10-13 02:14:00', 'ch05_20211012201724.mp4'),
        ('2021-10-13 02:15:00', '2021-10-13 06:40:00', 'ch05_20211013021444.mp4'),
        ('2021-10-13 06:41:00', '2021-10-13 09:05:00', 'ch05_20211013064018.mp4'),
        ('2021-10-12 21:05:00', '2021-10-13 00:08:00', 'ch06_20211012200000.mp4'),
        ('2021-10-13 00:09:00', '2021-10-13 03:08:00', 'ch06_20211013000846.mp4'),
        ('2021-10-13 03:09:00', '2021-10-13 05:16:00', 'ch06_20211013030812.mp4'),
        ('2021-10-13 05:17:00', '2021-10-13 07:22:00', 'ch06_20211013051620.mp4'),
        ('2021-10-13 07:23:00', '2021-10-13 09:05:00', 'ch06_20211013072219.mp4'),
    ],
    'D2_T3': [
        ('2021-10-12 21:09:00', '2021-10-13 09:09:00', 'ch07_20211012200000.mp4'),
        ('2021-10-12 21:09:00', '2021-10-13 09:09:00', 'ch08_20211012200000.mp4'),
        ('2021-10-12 21:09:00', '2021-10-13 04:30:00', 'ch09_20211012200000.mp4'),
        ('2021-10-13 04:31:00', '2021-10-13 09:09:00', 'ch09_20211013043003.mp4'),
    ],
    'D2_T4': [
        ('2021-10-12 21:00:00', '2021-10-13 09:00:00', 'ch10_20211012200000.mp4'),
        ('2021-10-12 21:00:00', '2021-10-12 22:11:00', 'ch11_20211012200000.mp4'),
        ('2021-10-12 22:12:00', '2021-10-13 09:00:00', 'ch11_20211012221136.mp4'),
        ('2021-10-12 21:00:00', '2021-10-13 09:00:00', 'ch12_20211012200000.mp4'),
    ],
    'D2_T5': [
        ('2021-10-12 20:57:00', '2021-10-13 08:57:00', 'ch13_20211012200000.mp4'),
        ('2021-10-12 20:57:00', '2021-10-13 08:57:00', 'ch14_20211012200000.mp4'),
        ('2021-10-12 20:57:00', '2021-10-13 08:57:00', 'ch16_20211012200000.mp4'),
    ],

    'D3_T1': [
        ('2021-10-13 19:10:00', '2021-10-13 21:42:00', 'ch01_20211013183000.mp4'),
        ('2021-10-13 21:43:00', '2021-10-14 07:10:00', 'ch01_20211013214243.mp4'),
        ('2021-10-13 19:10:00', '2021-10-14 07:10:00', 'ch02_20211013183000.mp4'),
        ('2021-10-13 19:10:00', '2021-10-14 03:12:00', 'ch03_20211013183000.mp4'),
        ('2021-10-14 03:13:00', '2021-10-14 07:10:00', 'ch03_20211014031259.mp4'),
    ],
    'D3_T2': [
        ('2021-10-13 19:10:00', '2021-10-14 12:12:24', 'ch04_20211013183000.mp4'),
        ('2021-10-14 12:13:00', '2021-10-14 07:10:00', 'ch04_20211014121224.mp4'),

        ('2021-10-13 19:10:00', '2021-10-13 20:31:00', 'ch05_20211013183000.mp4'),
        ('2021-10-13 20:33:00', '2021-10-14 04:08:19', 'ch05_20211013203210.mp4'),
        ('2021-10-14 04:09:00', '2021-10-14 10:09:25', 'ch05_20211014040819.mp4'),
        ('2021-10-14 10:40:00', '2021-10-14 07:00:00', 'ch05_20211014103925.mp4'),

        ('2021-10-13 19:10:00', '2021-10-13 19:06:49', 'ch06_20211013183000.mp4'),
        ('2021-10-13 19:47:00', '2021-10-14 01:09:13', 'ch06_20211013194649.mp4'),
        ('2021-10-14 02:00:00', '2021-10-14 04:05:16', 'ch06_20211014015913.mp4'),
        ('2021-10-14 04:04:00', '2021-10-14 06:01:38', 'ch06_20211014040516.mp4'),
        ('2021-10-14 06:12:00', '2021-10-14 07:00:00', 'ch06_20211014061138.mp4'),
    ],
    'D3_T3': [
        ('2021-10-13 19:10:00', '2021-10-14 07:00:00', 'ch07_20211013183000.mp4'),

        ('2021-10-13 19:10:00', '2021-10-14 03:05:44', 'ch08_20211013183000.mp4'),
        ('2021-10-14 03:46:00', '2021-10-14 07:00:00', 'ch08_20211014034544.mp4'),

        ('2021-10-13 19:10:00', '2021-10-14 07:00:00', 'ch09_20211013183000.mp4'),
    ],
    'D3_T4': [
        ('2021-10-13 19:10:00', '2021-10-14 07:00:00', 'ch10_20211013183000.mp4'),

        ('2021-10-13 19:10:00', '2021-10-14 07:00:00', 'ch11_20211013183000.mp4'),

        ('2021-10-13 19:10:00', '2021-10-13 22:01:25', 'ch12_20211013183000.mp4'),
        ('2021-10-13 22:22:00', '2021-10-14 07:00:00', 'ch12_20211013222125.mp4'),
    ],
    'D3_T5': [
        ('2021-10-13 19:10:00', '2021-10-14 07:00:00', 'ch13_20211013183000.mp4'),

        ('2021-10-13 19:10:00', '2021-10-14 01:08:25', 'ch14_20211013183000.mp4'),
        ('2021-10-14 01:29:00', '2021-10-14 07:00:00', 'ch14_20211014012825.mp4'),

        ('2021-10-13 22:22:00', '2021-10-14 03:02:51', 'ch16_20211013183000.mp4'),
        ('2021-10-13 03:43:00', '2021-10-14 07:00:00', 'ch16_20211014034251.mp4'),
    ],
    'D4_T1': [
        ('2021-10-14 20:01:00', '2021-10-15 08:02:00', 'ch01_20211014180000.mp4'),
        ('2021-10-14 20:01:00', '2021-10-15 08:02:00', 'ch02_20211014180000.mp4'),
        ('2021-10-14 20:01:00', '2021-10-15 08:02:00', 'ch03_20211014180000.mp4')
    ],
    'D4_T2': [
        ('2021-10-14 20:01:00', '2021-10-15 08:02:12', 'ch04_20211014180000.mp4'),

        ('2021-10-14 20:01:00', '2021-10-14 23:03:31', 'ch05_20211014180000.mp4'),
        ('2021-10-14 23:23:31', '2021-10-15 06:07:29', 'ch05_20211014232331.mp4'),
        ('2021-10-15 06:07:29', '2021-10-15 08:02:00', 'ch05_20211015060729.mp4'),

        ('2021-10-14 20:01:00', '2021-10-14 20:07:53', 'ch06_20211014180355.mp4'),
        ('2021-10-14 20:58:00', '2021-10-15 03:09:42', 'ch06_20211014205753.mp4'),
        ('2021-10-15 03:10:00', '2021-10-15 05:09:56', 'ch06_20211015030942.mp4'),
        ('2021-10-15 05:50:00', '2021-10-15 08:02:00', 'ch06_20211015054956.mp4'),
    ],
    'D4_T3': [
        ('2021-10-14 20:01:00', '2021-10-15 08:02:00', 'ch07_20211014180000.avi'),
        ('2021-10-14 20:01:00', '2021-10-15 08:02:00', 'ch08_20211014180000.avi'),
        ('2021-10-14 20:01:00', '2021-10-15 08:02:00', 'ch09_20211014180000.avi')
    ],
    'D4_T4': [
        ('2021-10-14 20:01:00', '2021-10-14 21:08:42', 'ch10_20211014180000.mp4'),
        ('2021-10-14 21:29:00', '2021-10-15 08:02:00', 'ch10_20211014212842.mp4'),
        ('2021-10-14 20:01:00', '2021-10-15 08:02:00', 'ch11_20211014180000.mp4'),
        ('2021-10-14 20:01:00', '2021-10-15 08:02:00', 'ch12_20211014180000.mp4')
    ],
    'D4_T5': [
        ('2021-10-14 20:01:00', '2021-10-15 08:02:00', 'ch13_20211014180000.mp4'),
        ('2021-10-14 20:01:00', '2021-10-15 08:02:00', 'ch14_20211014180000.mp4'),
        ('2021-10-14 20:01:00', '2021-10-15 08:02:00', 'ch16_20211014180000.mp4')
    ],
    'D5_T1': [
        ('2021-10-15 21:01:00', '2021 10 15 22 36 49', 'ch01_20211015190000.mp4'),
        ('2021 10 15 22 36 49', '2021-10-16 08:02:00', 'ch01_20211015223649.mp4'),

        ('2021-10-15 21:01:00', '2021 10 16 03 16 23', 'ch02_20211015190000.mp4'),
        ('2021 10 16 03 16 23', '2021-10-16 08:02:00', 'ch02_20211016031623.mp4'),

        ('2021-10-15 21:01:00', '2021 10 15 22 59 03', 'ch03_20211015190000.mp4'),
        ('2021 10 15 22 59 03', '2021-10-16 08:02:00', 'ch03_20211015225903.mp4'),
    ],
    'D5_T2': [
        ('2021-10-15 21:01:00', '2021 10 16 04 20 18', 'ch04_20211015190000.mp4'),
        ('2021 10 16 04 20 18', '2021-10-16 09:02:00', 'ch04_20211016042018.mp4'),

        ('2021-10-15 21:01:00', '2021 10 16 01 34 59', 'ch05_20211015190000.mp4'),
        ('2021 10 16 01 34 59', '2021-10-16 09:02:00', 'ch05_20211016013459.mp4'),

        ('2021-10-15 21:01:00', '2021 10 15 19 50 59', 'ch06_20211015190000.mp4'),
        ('2021 10 15 19 50 59', '2021 10 16 02 23 30', 'ch06_20211015195059.mp4'),
        ('2021 10 16 02 23 30', '2021 10 16 05 02 13', 'ch06_20211016022330.mp4'),
        ('2021 10 16 05 02 13', '2021 10 16 07 27 43', 'ch06_20211016050213.mp4'),
        ('2021 10 16 07 27 43', '2021-10-16 09:02:00', 'ch06_20211016072743.mp4'),
    ],
    'D5_T3': [
        ('2021-10-15 21:01:00', '2021 10 16 03 54 44', 'ch07_20211015190000.mp4'),
        ('2021 10 16 03 54 44', '2021-10-16 09:02:00', 'ch07_20211016035444.mp4'),
        ('2021-10-15 21:01:00', '2021-10-16 09:02:00', 'ch08_20211015190000.mp4'),
        ('2021-10-15 21:01:00', '2021-10-16 09:02:00', 'ch09_20211015190000.mp4'),
    ],
    'D5_T4': [
        ('2021-10-15 21:01:00', '2021-10-16 09:02:00', 'ch10_20211015190000.mp4'),
        ('2021-10-15 21:01:00', '2021 10 16 01 04 45', 'ch11_20211015190000.mp4'),
        ('2021 10 16 01 04 45', '2021-10-16 09:02:00', 'ch11_20211016010445.mp4'),
        ('2021-10-15 21:01:00', '2021-10-16 09:02:00', 'ch12_20211015190000.mp4')
    ],
    'D5_T5': [
        ('2021-10-15 21:01:00', '2021-10-16 09:02:00', 'ch13_20211015190000.mp4'),
        ('2021-10-15 21:01:00', '2021-10-16 09:02:00', 'ch14_20211015190000.mp4'),
        ('2021-10-15 21:01:00', '2021-10-16 09:02:00', 'ch16_20211015190000.mp4')
    ],
    'D6_T1': [
        ('2021-10-16 20:30:00', '2021-10-17 08:31:00', 'ch01_20211016193000.mp4'),
        ('2021-10-16 20:30:00', '2021-10-17 08:31:00', 'ch02_20211016193000.mp4'),
        ('2021-10-16 20:30:00', '2021-10-17 08:31:00', 'ch03_20211016193000.mp4'),
    ],
    'D6_T2': [
        ('2021-10-16 20:30:00', '2021-10-17 00:16:00', 'ch04_20211016193000.mp4'),
        ('2021-10-17 00:17:00', '2021-10-17 08:31:00', 'ch04_20211017001654.mp4'),

        ('2021-10-16 20:30:00', '2021-10-16 21:14:00', 'ch05_20211016193000.mp4'),
        ('2021-10-16 21:15:00', '2021-10-17 05:30:00', 'ch05_20211016211439.mp4'),
        ('2021-10-17 05:31:00', '2021-10-17 08:31:00', 'ch05_20211017053051.mp4'),

        ('2021-10-16 20:30:00', '2021-10-17 00:32:00', 'ch06_20211016193000.mp4'),
        ('2021-10-17 00:33:00', '2021-10-17 03:46:00', 'ch06_20211017003246.mp4'),
        ('2021-10-17 03:47:00', '2021-10-17 06:29:00', 'ch06_20211017034656.mp4'),
        ('2021-10-17 06:30:00', '2021-10-17 08:31:00', 'ch06_20211017062950.mp4'),
    ],
    'D6_T3': [
        ('2021-10-16 20:30:00', '2021-10-17 08:31:00', 'ch07_20211016193000.mp4'),
        ('2021-10-16 20:30:00', '2021-10-17 08:31:00', 'ch08_20211016193000.mp4'),
        ('2021-10-16 20:30:00', '2021-10-17 08:31:00', 'ch09_20211016193000.mp4'),
    ],
    'D6_T4': [
        ('2021-10-16 20:15:00', '2021-10-17 08:31:00', 'ch10_20211016193000.mp4'),
        ('2021-10-16 20:15:00', '2021-10-17 08:31:00', 'ch11_20211016193000.mp4'),
        ('2021-10-16 20:15:00', '2021-10-17 05:15:38', 'ch12_20211016193000.mp4'),
        ('2021-10-17 05:15:38', '2021-10-17 08:31:00', 'ch12_20211017051538.mp4'),
    ],
    'D6_T5': [
        ('2021-10-16 20:15:00', '2021-10-17 08:31:00', 'ch13_20211016193000.mp4'),
        ('2021-10-16 20:15:00', '2021-10-17 04:19:27', 'ch14_20211016193000.mp4'),
        ('2021-10-17 04:19:27', '2021-10-17 08:31:00', 'ch14_20211017041927.mp4'),
        ('2021-10-16 20:15:00', '2021-10-17 08:31:00', 'ch16_20211016193000.mp4'),
    ],

    'D7_T1': [
        ('2021-10-17 20:15:00', '2021-10-17 22:55:00', 'ch01_20211017190000.mp4'),
        ('2021-10-17 22:56:00', '2021-10-18 08:15:00', 'ch01_20211017225513.mp4'),
        ('2021-10-17 20:15:00', '2021-10-18 08:15:00', 'ch02_20211017190000.mp4'),
        ('2021-10-17 20:15:00', '2021-10-18 08:15:00', 'ch03_20211017191914.mp4'),
    ],
    'D7_T2': [
        ('2021-10-17 20:15:00', '2021-10-17 20:20:39', 'ch04_20211017190000.mp4'),
        ('2021-10-17 20:20:39', '2021-10-18 08:15:00', 'ch04_20211017202039.mp4'),

        ('2021-10-17 20:15:00', '2021-10-18 01:39:01', 'ch05_20211017190514.mp4'),
        ('2021-10-18 01:39:01', '2021-10-18 07:28:43', 'ch05_20211018013901.mp4'),
        ('2021-10-18 07:28:43', '2021-10-18 08:15:00', 'ch05_20211018072843.mp4'),

        ('2021-10-17 20:15:00', '2021-10-18 00:02:01', 'ch06_20211017190000.mp4'),
        ('2021-10-18 00:02:01', '2021-10-18 02:17:52', 'ch06_20211018000201.mp4'),
        ('2021-10-18 02:17:52', '2021-10-18 04:16:46', 'ch06_20211018021752.mp4'),
        ('2021-10-18 04:16:46', '2021-10-18 06:12:58', 'ch06_20211018041646.mp4'),
        ('2021-10-18 06:12:58', '2021-10-18 08:07:06', 'ch06_20211018061258.mp4'),
        ('2021-10-18 08:07:06', '2021-10-18 08:15:00', 'ch06_20211018080706.mp4'),
    ],
    'D7_T3': [
        ('2021-10-17 20:15:00', '2021-10-18 08:15:00', 'ch07_20211017190000.mp4'),
        ('2021-10-17 20:15:00', '2021-10-18 08:15:00', 'ch08_20211017190000.mp4'),
        ('2021-10-17 20:15:00', '2021-10-18 08:15:00', 'ch09_20211017190000.mp4'),
    ],
    'D7_T4': [
        ('2021-10-17 20:15:00', '2021-10-18 03:08:35', 'ch10_20211017190000.mp4'),
        ('2021-10-18 03:08:35', '2021-10-18 08:15:00', 'ch10_20211018030835.mp4'),
        ('2021-10-17 20:15:00', '2021-10-17 19:27:58', 'ch11_20211017190000.mp4'),
        ('2021-10-17 19:27:58', '2021-10-18 08:15:00', 'ch11_20211017192758.mp4'),
        ('2021-10-17 20:15:00', '2021-10-18 05:41:34', 'ch12_20211017190000.mp4'),
        ('2021-10-18 05:41:34', '2021-10-18 08:15:00', 'ch12_20211018054134.mp4'),
    ],
    'D7_T5': [
        ('2021-10-17 20:15:00', '2021-10-18 03:46:14', 'ch13_20211017190000.mp4'),
        ('2021-10-18 03:46:14', '2021-10-18 08:15:00', 'ch13_20211018034614.mp4'),

        ('2021-10-17 20:15:00', '2021-10-18 08:15:00', 'ch14_20211017190000.mp4'),

        ('2021-10-17 20:15:00', '2021-10-18 08:15:00', 'ch16_20211017190000.mp4'),
    ],
}

exp_time = {
    'D1_T1': {
        'start_t': '2021-10-11 21:49:59',
        'end_t': '2021-10-12 09:50:00',
    },
    'D1_T2': {
        'start_t': '2021-10-11 21:50:00',
        'end_t': '2021-10-12 09:50:00',
    },
    'D1_T3': {
        'start_t': '2021-10-11 21:50:00',
        'end_t': '2021-10-12 09:50:00',
    },
    'D1_T4': {
        'start_t': '2021-10-11 22:00:00',
        'end_t': '2021-10-12 09:00:00',
    },
    'D1_T5': {
        'start_t': '2021-10-11 22:00:00',
        'end_t': '2021-10-12 09:00:00',
    },

    'D2_T1': {
        'start_t': '2021-10-12 21:12:13',
        'end_t': '2021-10-13 09:10:00',
    },
    'D2_T2': {
        'start_t': '2021-10-12 21:10:00',
        'end_t': '2021-10-13 09:10:00',
    },
    'D2_T3': {
        'start_t': '2021-10-12 21:10:00',
        'end_t': '2021-10-13 09:10:00',
    },
    'D2_T4': {
        'start_t': '2021-10-12 21:09:59',
        'end_t': '2021-10-13 09:09:59',
    },
    'D2_T5': {
        'start_t': '2021-10-12 21:10:00',
        'end_t': '2021-10-13 09:10:00',
    },

    'D3_T1': {
        'start_t': '2021-10-13 19:19:59',
        'end_t': '2021-10-14 07:19:59',
    }, 'D3_T2': {
        'start_t': '2021-10-13 19:10:00',
        'end_t': '2021-10-14 07:10:00',
    }, 'D3_T3': {
        'start_t': '2021-10-13 19:10:00',
        'end_t': '2021-10-14 07:10:00',
    }, 'D3_T4': {
        'start_t': '2021-10-13 19:09:59',
        'end_t': '2021-10-14 07:09:59',
    }, 'D3_T5': {
        'start_t': '2021-10-13 19:09:59',
        'end_t': '2021-10-14 07:09:59',
    },

    'D4_T1': {
        'start_t': '2021-10-14 20:19:59',
        'end_t': '2021-10-15 08:02:00',
    }, 'D4_T2': {
        'start_t': '2021-10-14 20:01:00',
        'end_t': '2021-10-15 08:02:00',
    }, 'D4_T3': {
        'start_t': '2021-10-14 20:01:00',
        'end_t': '2021-10-15 08:02:00',
    }, 'D4_T4': {
        'start_t': '2021-10-14 20:09:59',
        'end_t': '2021-10-15 08:09:59',
    }, 'D4_T5': {
        'start_t': '2021-10-14 20:09:59',
        'end_t': '2021-10-15 08:09:59',
    },
    'D5_T1': {
        'start_t': '2021-10-15 21:19:59',
        'end_t': '2021-10-16 09:02:00',
    }, 'D5_T2': {
        'start_t': '2021-10-15 21:01:00',
        'end_t': '2021-10-16 09:02:00',
    }, 'D5_T3': {
        'start_t': '2021-10-15 21:01:00',
        'end_t': '2021-10-16 09:02:00',
    }, 'D5_T4': {
        'start_t': '2021-10-15 21:09:59',
        'end_t': '2021-10-16 09:09:59',
    }, 'D5_T5': {
        'start_t': '2021-10-15 21:09:59',
        'end_t': '2021-10-16 09:09:59',
    },
    'D6_T1': {
        'start_t': '2021-10-16 20:30:00',
        'end_t': '2021-10-17 08:31:00',
    }, 'D6_T2': {
        'start_t': '2021-10-16 20:30:00',
        'end_t': '2021-10-17 08:31:00',
    }, 'D6_T3': {
        'start_t': '2021-10-16 20:30:00',
        'end_t': '2021-10-17 08:31:00',
    }, 'D6_T4': {
        'start_t': '2021-10-16 20:29:59',
        'end_t': '2021-10-17 08:31:59',
    }, 'D6_T5': {
        'start_t': '2021-10-16 20:29:59',
        'end_t': '2021-10-17 08:31:59',
    },
    'D7_T1': {
        'start_t': '2021-10-17 20:15:00',
        'end_t': '2021-10-18 08:15:00',
    }, 'D7_T2': {
        'start_t': '2021-10-17 20:15:00',
        'end_t': '2021-10-18 08:15:00',
    }, 'D7_T3': {
        'start_t': '2021-10-17 20:15:00',
        'end_t': '2021-10-18 08:15:00',
    }, 'D7_T4': {
        'start_t': '2021-10-17 20:15:59',
        'end_t': '2021-10-18 08:15:59',
    }, 'D7_T5': {
        'start_t': '2021-10-17 20:15:00',
        'end_t': '2021-10-18 08:15:00',
    },
}

if __name__ == '__main__':
    base_cfg_path = "E:\\data\\3D_pre"
    base_setting_files = 'settings.ini'
    fre_seq = {
        'D1': "12min",
        'D2': "20min",
    }
    update_day = [
        # 'D1_T1',
        # 'D1_T2', 'D1_T3',
        # 'D1_T4',
        # 'D1_T5',
        # 'D2_T1',
        # 'D2_T2', 'D2_T3',
        # 'D2_T4',
        # 'D2_T5',
        # 'D3_T1',
        # 'D3_T2',
        # 'D3_T3',
        # 'D3_T4', 'D3_T5',
        # 'D4_T1',
        # 'D4_T2',
        # 'D4_T3',
        # 'D4_T4', 'D4_T5',
        # 'D5_T1',
        # 'D5_T2',
        # 'D5_T3',
        # 'D5_T4',
        # 'D5_T5',
        # 'D6_T1',
        # 'D6_T2',
        # 'D6_T3',
        # 'D6_T4', 'D6_T5',
        # 'D7_T1',
        # 'D7_T2', 'D7_T3',
        # 'D7_T4',
        'D7_T5',
    ]

    for DayTank, video_info in sample_map.items():
        DayTank_setting_path = os.path.join(base_cfg_path, DayTank)
        DayTank_setting_file = os.path.join(DayTank_setting_path, 'settings.ini')

        if not os.path.exists(DayTank_setting_path):
            os.makedirs(DayTank_setting_path)
        if os.path.isfile(DayTank_setting_file):
            print(f"{DayTank} base settings is existed! ")
        else:
            day = DayTank[1]
            if int(day) == 1:
                shutil.copy(
                    os.path.join(base_cfg_path, base_setting_files),
                    DayTank_setting_file
                )
            else:
                shutil.copy(
                    os.path.join(base_cfg_path, f"D{str(int(day) - 1)}_{DayTank[3:]}", base_setting_files),
                    DayTank_setting_file
                )


        if DayTank in update_day:
            writeConfig(DayTank_setting_path, [('VideoStartTime', None, None)])
            writeConfig(DayTank_setting_path, [('VideoStartTime_Failed', None, None)])

            datetime_range = pd.date_range(start=exp_time[DayTank]['start_t'], end=exp_time[DayTank]['end_t'],
                                           freq=fre_seq['D2'])
            date_list = [x.strftime('%Y_%m_%d_%H_%M_%S') for x in list(datetime_range)]
            print(f"date list length: {len(date_list)}")
            for time_info in video_info:
                iv_start_T, iv_end_T, ivideo = time_info
                try:
                    iv_start_unix_time = time.mktime(time.strptime(iv_start_T, "%Y-%m-%d %H:%M:%S"))
                except:
                    iv_start_unix_time = time.mktime(time.strptime(iv_start_T, "%Y %m %d %H %M %S"))
                try:
                    iv_end_unix_time = time.mktime(time.strptime(iv_end_T, "%Y-%m-%d %H:%M:%S"))
                except:
                    iv_end_unix_time = time.mktime(time.strptime(iv_end_T, "%Y %m %d %H %M %S"))
                cut_time_list = []
                for idate in date_list:
                    idate_unix_time = time.mktime(time.strptime(idate, "%Y_%m_%d_%H_%M_%S"))
                    if idate_unix_time >= iv_start_unix_time and idate_unix_time <= iv_end_unix_time:
                        cut_time_list.append(idate)
                    writeConfig(DayTank_setting_path, [('VideoStartTime', ivideo, '\n'.join(cut_time_list))])
