"""Configuration class for applications"""

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

import torch

config = configparser.ConfigParser()
config.read("config.ini")
base_dir = Path('')

os.environ["NLS_LANG"] = "AMERICAN_AMERICA.AL32UTF8"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@dataclass()
class Media:
    media_path = base_dir.joinpath(config["MEDIA"]["MEDIA_PATH"])

    def __init__(self):
        self.media_path.mkdir(exist_ok=True, parents=False)


@dataclass()
class YoloConfig:
    weights = base_dir.joinpath(config["YOLOV5-PERSON"]["WEIGHTS"])
    img_size = int(config["YOLOV5-PERSON"]["IMG_SIZE"])
    conf_thresh = float(config["YOLOV5-PERSON"]["CONF_THRES"])
    iou_thresh = float(config["YOLOV5-PERSON"]["IOU_TRHESH"])
    device = device
    view_img = config["YOLOV5-PERSON"]["VIEW_IMG"]
    save_txt = config["YOLOV5-PERSON"]["SAVE_TXT"]
    save_conf = config["YOLOV5-PERSON"]["SAVE_CONF"]
    classes = int(config["YOLOV5-PERSON"]["CLASSES"])
    agnostic_nms = config["YOLOV5-PERSON"]["AGNOSTIC_NMS"]
    augment = config["YOLOV5-PERSON"]["AUGMENT"]
    update = config["YOLOV5-PERSON"]["UPDATE"]
    project = base_dir.joinpath(config["YOLOV5-PERSON"]["PROJECT"])
    name = config["YOLOV5-PERSON"]["NAME"]
    exist_ok = config["YOLOV5-PERSON"]["EXIST_OK"]
