import cv2
import os
import numpy as np

'''
ffmpeg cut video cmd:
ffmpeg  -i ./whiteTop.mp4 -vcodec copy -acodec copy -ss 00:00:00 -to 00:00:05 ./cutout1.mkv
'''



def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def capture_video(color):
    video_frag = settings[color]['video_path']
    output_dir = settings[color]['image_path']
    # 開啟影片檔案 Open the file
    cap = cv2.VideoCapture(video_frag)

    # 設定輸出的資料夾，預設是此檔案所在位置在建立一個output Set output folder here
    output_dir = output_dir

    # 如果沒有上述資料夾就建立資料夾 If no output_dir floder than create one
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 設定第一幀的編號 Set first frame's name
    c = startno
    if not cap.isOpened():
        print("check video file")
        exit(111)
    # 用迴圈從影片檔案讀取幀，並顯示出來 Use while to get the frame from video
    cnt = 0
    while True:
        ret, frame = cap.read()

        # 下采样
        # frame = cv2.resize()

        if cnt < jump_frame:
            cnt += 1
            continue
        else:
            cv2.imwrite(output_dir + '/' + str(c).zfill(8) + '.jpg', frame)
            cv2.imshow('frame', frame)
            c = c + 1
            # 重置标识符
            cnt = 0
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print(cap.get(0))
        print(c)
        print("="*80)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    verbose = True
    root_path = "E:\\data\\3D_pre\\cam2"
    jump_frame = 0
    settings = {
        'l_top': {
            'video_path': f'{root_path}\\left.mp4',
            'image_path': f'{root_path}\\top_left\\left',
        },
        'top_l': {
            'video_path': f'{root_path}\\top.mp4',
            'image_path': f'{root_path}\\top_left\\top',
        },
        'r_top': {
            'video_path': f'{root_path}\\right.mp4',
            'image_path': f'{root_path}\\top_right\\right',
        },
        # 'top_r': {
        #     'video_path': f'{root_path}\\top.mp4',
        #     'image_path': f'{root_path}\\top_right\\top',
        # },

    }
    startno = 1
    for i in ['l_top', 'top_l', 'r_top', 'top_r']:
        try:
            capture_video(i)
        except Exception as e:
            print(e)
        finally:
            print("video to image: done!")
