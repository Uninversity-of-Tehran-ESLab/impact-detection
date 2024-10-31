from impact_utils import detect_markers

from typing import Sequence

import numpy as np 
import cv2
from cv2.typing import MatLike 


def draw_transformed_perspective(
        frame: Sequence[MatLike],
        corners: Sequence[MatLike],
    ) -> MatLike:

    PERSPECTIVE = np.float32(
        [[0, 0], [400, 0],
        [0, 640], [400, 640]]
    )
    points = np.array([corner[0][0] for corner in corners])
    print(points)
    squeezed_points =np.squeeze(points)
    matrix = cv2.getPerspectiveTransform(squeezed_points, PERSPECTIVE)
    return cv2.warpPerspective(frame, matrix, (500, 600))

 
cap = cv2.VideoCapture(1)
 
while True:
    ret, frame = cap.read()

    if not ret or cv2.waitKey(24) == 27:
        break
    corners, _, frame = detect_markers(frame, draw_markers=True)
    cv2.imshow('frame', frame)

    if len(corners) < 4:
        continue

    transformed = draw_transformed_perspective(frame, corners)
    cv2.imshow('frame1', transformed) # Transformed Capture
 
cap.release()
cv2.destroyAllWindows()