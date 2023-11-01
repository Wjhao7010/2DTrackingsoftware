import cv2
frame_area = cv2.imread("C:\\Users\\w\\Desktop\\4.png")
#
# import pytesseract
# from PIL import Image
# pytesseract.pytesseract.tesseract_cmd = 'D:\\program\\ocr\\tesseract.exe'
# s = pytesseract.image_to_string(frame_area)  #不加lang参数的话，默认进行英文识别
# print(s)

import easyocr
reader = easyocr.Reader(['en'], gpu=True, recog_network='latin_g2')
result = reader.readtext(frame_area.copy())
print(result)

from paddleocr import PaddleOCR, draw_ocr

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True)  # need to run only once to download and load model into memory
result = ocr.ocr(frame_area, det=False)
for line in result:
    print(line)