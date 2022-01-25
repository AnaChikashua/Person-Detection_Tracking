import sys
import time
import unittest
from pathlib import Path
import numpy as np

import cv2

sys.path.insert(0, './src/yolov5')

from src.yolov5.detect import Yolo


class TestYolo(unittest.TestCase):
    def setUp(self) -> None:
        self.img_path = Path('media/inputs/test/zidane.jpg')
        self.model = Yolo()
        self.img = cv2.cvtColor(cv2.imread(str(self.img_path)), cv2.COLOR_BGR2RGB)

    def test_detection_one_img(self):
        start = time.time()
        output = self.model.detect_image(self.img)
        print(output)
        end = time.time()
        print('one image took ', end - start)

    def test_detection_multiple_imgs(self):
        print(self.img.shape)
        imgs = np.array([self.img for i in range(10)])
        print(imgs.shape)

        outputs = self.model.detect_image(imgs)
        print(outputs)
