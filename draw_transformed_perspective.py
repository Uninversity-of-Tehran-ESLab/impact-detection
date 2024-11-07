# In the name of God
#TODO: Fix this!

from ImpactUtils import ImpactUtils

from typing import Sequence, Tuple

import cv2
import numpy as np 


cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret or cv2.waitKey(24) == 27:
        break
    corners, ids, frame = ImpactUtils.detect_markers(frame, draw_markers=True)
    cv2.imshow('frame', frame)

    if len(corners) < 4:
        continue

    transformed = ImpactUtils.draw_transformed_perspective(frame, corners)
    cv2.imshow('frame1', transformed) # Transformed Capture
 
cap.release()
cv2.destroyAllWindows()