# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
    $ python track.py --weights /home/huangjinze/code/yolov5/runs/train/yolov5m14/weights/best.pt --img 640 --save-txt --nosave --source /home/data/HJZ/MFT/test/cruise4/img1/
"""

import argparse
import sys
import time
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
sys.path.append('.')
sys.path.append('../../')

from common.utility import getRealTime
import os
import torch
import torch.backends.cudnn as cudnn
import cv2
import matplotlib.pyplot as plt

from modules.yolo5.models.experimental import attempt_load
from modules.yolo5.utils.datasets import LoadStreams, LoadImages
from modules.yolo5.utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression, \
    apply_classifier, scale_coords, strip_optimizer, set_logging, increment_path, save_one_box
from modules.yolo5.utils.plots import Annotator, colors
from modules.yolo5.utils.torch_utils import select_device, load_classifier, time_sync

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.1,  # confidence threshold
        iou_thres=0.1,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        save_crop=False,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        show=True,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        config_path=None,  # use FP16 half-precision inference
        region_name=None,  # use FP16 half-precision inference
        ):
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    ID_T = os.path.split(source)[1].split(".")[0]
    camNO = ID_T.split("_")[-1]
    Video_name = source.split("/")[-1]

    # Directories
    if not os.path.exists(project):
        os.makedirs(project)
    txt_path = str(f"{project}/{Video_name.replace('.avi', '.csv')}")
    f = open(txt_path, 'w+')
    f.write(
        'Filename,Object ID,Annotation tag,'
        'Upper left corner X,Upper left corner Y,'
        'Lower right corner X,Lower right corner Y,'
        'Confidence,Frame'
        '\n'
    )

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(
            source, img_size=imgsz, stride=stride,
            auto=pt, camNO=camNO, region_name=region_name,
            config_path=config_path
        )
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()



    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        pred = model(img, augment=augment, visualize=visualize)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        txtyxy = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                        # line = (int(p.stem), int(idx), *txtyxy, 1, -1, -1, -1) # label format
                        line = (frame, -1, 0, *txtyxy, conf, frame)  # label format
                        f.write(('%g,' * len(line)).rstrip(',') % line + '\n')

                        if show:
                            cv2.rectangle(im0, (int(txtyxy[0]), int(txtyxy[1])), (int(txtyxy[2]), int(txtyxy[3])),
                                          (0, 255, 0), 2)
                            # print("=======================")
                            try:
                                cv2.imshow(f'demo', im0)
                                # cv2.namedWindow(f'demo{frameCount}', 0)
                                # cv2.resizeWindow(f'demo{frameCount}', 676, 380)
                                # cv2.imshow(f'demo{frameCount}', show_img)
                                cv2.waitKey(100)
                                # cv2.destroyWindow(f'demo{frameCount}')
                            except:
                                plt.imshow(im0)
                                plt.show()

                            # plt.imshow(show_img)
                            # plt.show()


            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='../Sessions/train_ZeF_front/zerbrafish_front7/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--config_path', type=str, default='/home/data/HJZ/zef/exp_pre/', help='config path')
    # parser.add_argument('--config_path', type=str,
    #                     default='E:\\data\\3D_pre\\D7_T4',
    #                     help='config path'
    #                     )
    parser.add_argument('--region_name', type=str, default='2_6PPDQ500ppb', help='config path')
    parser.add_argument('--source', type=str,
                        default='E:\\data\\3D_pre\\D7_T4/cut_video/2021_10_18_05_15_59_ch11.avi',
                        help='file/dir/URL/glob, 0 for webcam'
                        )
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=2, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', default=True ,action='store_true', help='save results to *.txt')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', default=True, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project',
                        default='E:\\data\\3D_pre\\D7_T4\\processed\\2_6PPDQ500ppb',
                        help='save results to project/name'
                        )
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--show', default=False, action='store_true', help='show result')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# python detect.py --source /home/data/HJZ/zef/exp_pre/D04_20210918195000/2021_09_18_19_52_03_D01.avi --weights /home/huangjinze/code/3D-ZeF/modules/Sessions/train_ZeF_front/zerbrafish_front7/weights/best.pt --img 640 --project /home/data/HJZ/zef/exp_pre/yolo_process/1_1 --save-txt --max-det 5
# python detect.py --source /home/data/HJZ/zef/exp_pre/D01_20210918195000/2021_09_18_19_52_02_D04.avi --weights /home/huangjinze/code/3D-ZeF/modules/Sessions/train_ZeF_top/zerbrafish_top3/weights/best.pt --img 640 --project /home/data/HJZ/zef/exp_pre/yolo_process/1_1 --save-txt --max-det 5 --camId 1