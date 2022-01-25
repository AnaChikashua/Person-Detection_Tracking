import numpy
import torch
import cv2

from config import yolo_config
from models.experimental import attempt_load
from utils.datasets import (letterbox)
from utils.general import (non_max_suppression, scale_coords)
from pathlib import Path


class Yolo:
    def __init__(self):
        self.weights = yolo_config.weights
        self.view_img = yolo_config.view_img
        self.image_size = yolo_config.img_size
        self.device = yolo_config.device
        self.augment = yolo_config.augment
        self.conf_thresh = yolo_config.conf_thresh
        self.iou_thresh = yolo_config.iou_thresh
        self.classes = yolo_config.classes
        self.save_conf = yolo_config.save_conf
        self.agnostic_nms = yolo_config.agnostic_nms
        self.device = yolo_config.device
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(self.weights, self.device)
        if self.half:
            self.model.half()

    def load_image(self, image: numpy.array) -> numpy.array:

        img = letterbox(image, self.image_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = numpy.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def concat_image(self, images: list) -> numpy.array:
        images = tuple([self.load_image(image) for image in images])
        img = torch.cat(images, 0)
        return img

    def detect_image(self, image: list) -> torch.tensor:
        img = self.concat_image(image)
        pred = self.model(img, augment=self.augment)[0]
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.classes,
                                   agnostic=self.agnostic_nms)

        _detections = list()
        for det in pred:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image[0].shape).round()
            _detections.append(det.detach().cpu())
        return _detections


def draw_boxes(image: numpy.array, bounding_boxes, text: str = 'Face', color: tuple = (0, 255, 0)) -> numpy.array:
    for bounding_box in bounding_boxes:
        x1 = int(bounding_box[0])
        y1 = int(bounding_box[1])
        x2 = int(bounding_box[2])
        y2 = int(bounding_box[3])
        text = round(float(bounding_box[4]), 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        cv2.rectangle(image, (x1, y1 - 35), (x1 + len(str(text)) * 19 + 80, y1), color, -1)

        cv2.putText(image, f"{text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    return image


def on_video(video_path: Path):
    yolo = Yolo()
    video = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    im_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_video_path = Path('media/output').joinpath(video_path.name)
    writer = cv2.VideoWriter(str(save_video_path), fourcc, 24, (im_width, im_height))
    while True:
        ret_value, frame = video.read()
        if not ret_value:
            break
        bounding_boxes = yolo.detect_image([frame])
        image = draw_boxes(frame, bounding_boxes[0])
        if writer is not None:
            writer.write(image)


if __name__ == '__main__':
    # video_path1 = Path('media/input/10.39.44.94_27_20210603103737951_1_3_0_5_0.mp4')
    # on_video(video_path1)
    yolo = Yolo()
    image_path = 'media/input/getty_517194189_373099.jpg'
    dest_path = 'media/output/getty_517194189_373099.jpg'
    image_frame = cv2.imread(image_path)
    bounding_boxes = yolo.detect_image([image_frame])
    image = draw_boxes(image_frame, bounding_boxes[0])
    cv2.imwrite(dest_path, image)
