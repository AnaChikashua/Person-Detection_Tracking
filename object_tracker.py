from centroidtracker import CentroidTracker
from detect import Yolo, draw_boxes
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--detectInt", type=int, default=20,
                help="number of frame between detection")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

ct = CentroidTracker()
(H, W) = (None, None)

print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
yolo = Yolo()

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
detectInt = args["detectInt"]
current_Frame = 0
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    rects = []

    if current_Frame % detectInt == 0:
        bounding_boxes = yolo.detect_image([frame])
        image = draw_boxes(frame, bounding_boxes[0])

        # loop over the detections
        for box in bounding_boxes[0]:
            rects.append(np.array(box[:4]).astype("int"))
    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    current_Frame += 1
    if key == ord(" "):
        break

cv2.destroyAllWindows()
vs.stop()
