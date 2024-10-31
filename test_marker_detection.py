# In the name of God

import cv2
import impact_utils
#TODO: make this a function and add it to main!

def test_marker_detection():
    """
        A demo to verify that markers are being detected
        correctly and the printed markers are in good condition
    """

    video_capture = cv2.VideoCapture(1)
    if not video_capture.isOpened():
        print("No")
        exit()

    while (True):
        is_captured, frame = video_capture.read()
        if not is_captured:
            continue
        
        _, _, frame_markers = impact_utils.detect_markers(frame=frame,
                                                          draw_markers=True)

        cv2.imshow("CAP", frame_markers)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break