#!/bin/bash
# dos2unix filename
# 使用vi打开文本文件
# vi dos.txt
# 命令模式下输入
# :set fileformat=unix
# :w

# 1
for value in "D1_T1", "D1_T2", "D1_T3“, ”D1_T4“, ”D1_T5”, “D2_T1", "D2_T2", "D2_T3“, ”D2_T4“, ”D2_T5”,"D3_T1", "D3_T2", "D3_T3“, ”D3_T4“, ”D3_T5”,"D4_T1", "D4_T2", "D4_T3“, ”D4_T4“, ”D4_T5”,
do
     mkdir /home/data/HJZ/zef/D1_T1/bak && mv /home/data/HJZ/zef/D1_T1/bak /home/data/HJZ/zef/D1_T1/bak
done


#rm /home/data/HJZ/zef/D1_T5/bg_processed/1_CK/2021_10_12_02_36_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/2_CK/2021_10_12_02_36_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/3_CK/2021_10_12_02_36_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/4_CK/2021_10_12_02_36_00_ch16.csv
#python modules/detection/BgDetector.py --video_name 2021_10_12_02_36_00_ch16.avi --region_names 1_CK,2_CK,3_CK,4_CK --path /home/data/HJZ/zef/D1_T5/

# 2
#rm /home/data/HJZ/zef/D1_T5/bg_processed/1_CK/2021_10_12_04_12_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/2_CK/2021_10_12_04_12_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/3_CK/2021_10_12_04_12_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/4_CK/2021_10_12_04_12_00_ch16.csv
#python modules/detection/BgDetector.py --video_name 2021_10_12_04_12_00_ch16.avi --region_names 1_CK,2_CK,3_CK,4_CK --path /home/data/HJZ/zef/D1_T5/

#rm /home/data/HJZ/zef/D1_T5/bg_processed/1_CK/2021_10_12_04_48_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/2_CK/2021_10_12_04_48_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/3_CK/2021_10_12_04_48_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/4_CK/2021_10_12_04_48_00_ch16.csv
#python modules/detection/BgDetector.py --video_name 2021_10_12_04_48_00_ch16.avi --region_names 1_CK,2_CK,3_CK,4_CK --path /home/data/HJZ/zef/D1_T5/
#
#rm /home/data/HJZ/zef/D1_T5/bg_processed/1_CK/2021_10_12_05_12_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/2_CK/2021_10_12_05_12_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/3_CK/2021_10_12_05_12_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/4_CK/2021_10_12_05_12_00_ch16.csv
#python modules/detection/BgDetector.py --video_name 2021_10_12_05_12_00_ch16.avi --region_names 1_CK,2_CK,3_CK,4_CK --path /home/data/HJZ/zef/D1_T5/
#
#rm /home/data/HJZ/zef/D1_T5/bg_processed/1_CK/2021_10_12_06_00_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/2_CK/2021_10_12_06_00_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/3_CK/2021_10_12_06_00_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/4_CK/2021_10_12_06_00_00_ch16.csv
#python modules/detection/BgDetector.py --video_name 2021_10_12_06_00_00_ch16.avi --region_names 1_CK,2_CK,3_CK,4_CK --path /home/data/HJZ/zef/D1_T5/
#
#rm /home/data/HJZ/zef/D1_T5/bg_processed/1_CK/2021_10_12_06_24_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/2_CK/2021_10_12_06_24_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/3_CK/2021_10_12_06_24_00_ch16.csv
#rm /home/data/HJZ/zef/D1_T5/bg_processed/4_CK/2021_10_12_06_24_00_ch16.csv
#python modules/detection/BgDetector.py --video_name 2021_10_12_06_24_00_ch16.avi --region_names 1_CK,2_CK,3_CK,4_CK --path /home/data/HJZ/zef/D1_T5/

#python threading_interpolation.py --DayTank D1_T1
#python threading_interpolation.py --DayTank D1_T2
#python threading_interpolation.py --DayTank D1_T3
#python threading_interpolation.py --DayTank D1_T4
#python threading_interpolation.py --DayTank D1_T5
#
#python threading_interpolation.py --DayTank D2_T1
#python threading_interpolation.py --DayTank D2_T2
#python threading_interpolation.py --DayTank D2_T3
#python threading_interpolation.py --DayTank D2_T4
##python threading_interpolation.py --DayTank D2_T5
#
#python threading_interpolation.py --DayTank D3_T1
#python threading_interpolation.py --DayTank D3_T2
#python threading_interpolation.py --DayTank D3_T3
#python threading_interpolation.py --DayTank D3_T4
#python threading_interpolation.py --DayTank D3_T5
#
#python threading_interpolation.py --DayTank D4_T1
#python threading_interpolation.py --DayTank D4_T2
#python threading_interpolation.py --DayTank D4_T3
#python threading_interpolation.py --DayTank D4_T4
#python threading_interpolation.py --DayTank D4_T5
#
##python threading_interpolation.py --DayTank D5_T1
#python threading_interpolation.py --DayTank D5_T2
#python threading_interpolation.py --DayTank D5_T4
#python threading_interpolation.py --DayTank D5_T5
#
#python threading_interpolation.py --DayTank D6_T1
#python threading_interpolation.py --DayTank D6_T2
##python threading_interpolation.py --DayTank D6_T4
#python threading_interpolation.py --DayTank D6_T5
#
#python threading_interpolation.py --DayTank D7_T1
#python threading_interpolation.py --DayTank D7_T2
#python threading_interpolation.py --DayTank D7_T4
#python threading_interpolation.py --DayTank D7_T5
#
#python threading_interpolation.py --DayTank D8_T2
#python threading_interpolation.py --DayTank D8_T4
