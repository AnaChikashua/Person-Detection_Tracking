import numpy
import torch
import cv2

from config import yolo_config
from models.experimental import attempt_load
from utils.datasets import (letterbox)
from utils.general import (non_max_suppression, scale_coords)
from pathlib import Path
from centroidtracker import CentroidTracker

# number of frame between detection and tracking
detectInt = 10


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

    def load_image(self, _image: numpy.array) -> numpy.array:

        img = letterbox(_image, self.image_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = numpy.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def concat_image(self, images: list) -> numpy.array:
        images = tuple([self.load_image(_image) for _image in images])
        img = torch.cat(images, 0)
        return img

    def detect_image(self, _image: list) -> torch.tensor:
        img = self.concat_image(_image)
        pred = self.model(img, augment=self.augment)[0]
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.classes,
                                   agnostic=self.agnostic_nms)

        _detections = list()
        for det in pred:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], _image[0].shape).round()
            _detections.append(det.detach().cpu())
        return _detections


def draw_boxes(_image: numpy.array, _bounding_boxes, text: str = 'Face', color: tuple = (0, 255, 0)) -> numpy.array:
    for bounding_box in _bounding_boxes:
        x1 = int(bounding_box[0])
        y1 = int(bounding_box[1])
        x2 = int(bounding_box[2])
        y2 = int(bounding_box[3])
        text = round(float(bounding_box[4]), 2)
        cv2.rectangle(_image, (x1, y1), (x2, y2), color, 4)
        cv2.rectangle(_image, (x1, y1 - 35), (x1 + len(str(text)) * 19 + 80, y1), color, -1)

        cv2.putText(_image, f"{text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    return _image


def on_video(video_path: Path):
    yolo = Yolo()
    ct = CentroidTracker()
    video = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    im_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_video_path = Path('media/output').joinpath(video_path.name)
    writer = cv2.VideoWriter(str(save_video_path), fourcc, 24, (im_width, im_height))
    current_frame = 0
    while True:
        ret_value, frame = video.read()
        if not ret_value:
            break
        rects = []
        if current_frame % detectInt == 0:
            bounding_boxes = yolo.detect_image([frame])
            frame = draw_boxes(frame, bounding_boxes[0])
            for box in bounding_boxes[0]:
                rects.append(numpy.array(box[:4]).astype("int"))
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        current_frame += 1
        if writer is not None:
            writer.write(frame)


if __name__ == '__main__':
    video_path1 = Path('media/input/example_01.mp4')
    on_video(video_path1)
    # yolo = Yolo()
    # image_path = 'media/input/getty_517194189_373099.jpg'
    # dest_path = 'media/output/getty_517194189_373099.jpg'
    # image_frame = cv2.imread(image_path)
    # bounding_boxes = yolo.detect_image([image_frame])
    # image = draw_boxes(image_frame, bounding_boxes[0])
    # cv2.imwrite(dest_path, image)
