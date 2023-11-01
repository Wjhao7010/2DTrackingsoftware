#!/bin/bash
# dos2unix filename
# 使用vi打开文本文件
# vi dos.txt
# 命令模式下输入
# :set fileformat=unix
# :w
cd /home/huangjinze/code/3D-ZeF


python threading_cutTank.py --DayTank D4_T3
python threading_cutTank.py --DayTank D7_T2
python threading_cutTank.py --DayTank D7_T3
python threading_cutTank.py --DayTank D7_T4
python threading_cutTank.py --DayTank D7_T5
