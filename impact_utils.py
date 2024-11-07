# In the name of God

import cv2
from cv2.typing import MatLike
from typing import Sequence, Tuple

def detect_markers(
        frame: MatLike,
        draw_markers: bool = False,
        aruco_dictionary: int = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    ) -> Tuple[Sequence[MatLike], MatLike, MatLike | None]:
    
    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        gray_scale_frame, 
        aruco_dictionary, 
    )
    frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    
    return corners, ids, (frame_markers if draw_markers else None)

def apply_color_filter(
        frame: MatLike
)
    cv2.