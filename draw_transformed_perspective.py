# In the name of God
#TODO: Fix this!

from impact_utils import detect_markers

from typing import Sequence, Tuple

import numpy as np 
import cv2
from cv2.typing import MatLike 


def __find_center(marker_corners: Sequence[MatLike]) -> Tuple[float, float]:
    # The Sequence only contains one item
    marker_corners = marker_corners[0]
    total = np.array([0, 0])
    for corner in marker_corners:
        total += corner
    return total / len(marker_corners)
    
    
def sort_markers(markers_corners: Sequence[MatLike]) -> Sequence[MatLike]:
    # Grouping two left makers and two right markers
    markers_corners = sorted(
        markers_corners,
        key=lambda x: __find_center(x)[0]
    )

    left_markers = markers_corners[0:1]
    right_markers = markers_corners[2:3]

    # Sorting based on height
    left_markers = sorted(
        left_markers,
        key=lambda x: __find_center(x)[1]
    )
    
    right_markers = sorted(
        right_markers,
        key=lambda x: __find_center(x)[1]
    )

    return np.array([left_markers, right_markers])

def draw_transformed_perspective(
        frame: Sequence[MatLike],
        corners: Sequence[MatLike],
    ) -> MatLike: 

    PERSPECTIVE = np.float32(
        [[0, 0], [0, 450],
        [300, 0], [300, 450]]
    )

    corners = sort_markers(corners)

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
    corners, ids, frame = detect_markers(frame, draw_markers=True)
    cv2.imshow('frame', frame)

    if len(corners) < 4:
        continue

    transformed = draw_transformed_perspective(frame, corners)
    cv2.imshow('frame1', transformed) # Transformed Capture
 
cap.release()
cv2.destroyAllWindows()