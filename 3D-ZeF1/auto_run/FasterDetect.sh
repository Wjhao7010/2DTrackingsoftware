#!/bin/bash
# dos2unix filename
# 使用vi打开文本文件
# vi dos.txt
# 命令模式下输入
# :set fileformat=unix
# :w
cd /home/huangjinze/code/3D-ZeF


python threading_bgdetect.py --DayTank D1_T1
python threading_bgdetect.py --DayTank D1_T2
python threading_bgdetect.py --DayTank D1_T3
python threading_bgdetect.py --DayTank D1_T4
python threading_bgdetect.py --DayTank D1_T5
python threading_bgdetect.py --DayTank D2_D1
python threading_bgdetect.py --DayTank D2_T2
python threading_bgdetect.py --DayTank D2_T3
python threading_bgdetect.py --DayTank D2_T4
python threading_bgdetect.py --DayTank D2_T5
python threading_bgdetect.py --DayTank D3_D1
python threading_bgdetect.py --DayTank D3_T2
python threading_bgdetect.py --DayTank D3_T3
python threading_bgdetect.py --DayTank D3_T4
python threading_bgdetect.py --DayTank D3_T5
python threading_bgdetect.py --DayTank D4_D1
python threading_bgdetect.py --DayTank D4_T2
python threading_bgdetect.py --DayTank D4_T3
python threading_bgdetect.py --DayTank D4_T4
python threading_bgdetect.py --DayTank D4_T5

python threading_bgdetect.py --DayTank D5_D1
python threading_bgdetect.py --DayTank D5_T2
python threading_bgdetect.py --DayTank D5_T4
python threading_bgdetect.py --DayTank D5_T5

python threading_bgdetect.py --DayTank D6_D1
python threading_bgdetect.py --DayTank D6_T2
python threading_bgdetect.py --DayTank D6_T4
python threading_bgdetect.py --DayTank D6_T5

python threading_bgdetect.py --DayTank D7_D1
python threading_bgdetect.py --DayTank D7_T2
python threading_bgdetect.py --DayTank D7_T4
python threading_bgdetect.py --DayTank D7_T5

python threading_bgdetect.py --DayTank D8_T2
python threading_bgdetect.py --DayTank D8_T4
