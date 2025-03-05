# In the name of God

#TODO: add yolo to this!

import ImpactUtils
import cv2
import ultralytics

def test_ball_detection(
        video_source: int=1
    ) -> None:
    """
    A demo to verify that the chosen YOLO model is working 
    properly and it is able to detect the balls!
    """
    video_capture = cv2.VideoCapture(video_source)
    if not video_capture.isOpened():
        print("Unable to initialize video capture!")
        return
    
    while (True):
        is_captured, frame = video_capture.read()
        if not is_captured:
            continue
