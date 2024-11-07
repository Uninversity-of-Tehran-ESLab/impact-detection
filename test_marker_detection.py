# In the name of God

import cv2
import ImpactUtils
#TODO: make this a function and add it to main!

def test_marker_detection(source: int | str) -> None:
    """
        A demo to verify that markers are being detected
        correctly and the printed markers are in good condition
        you can either enter the source number to use a camera or 
        give a file(picture)
    """

    match type(source):
        case type(int):
            video_capture = cv2.VideoCapture(source)
            if not video_capture.isOpened():
                print("No")
                exit()

            while (True):
                is_captured, frame = video_capture.read()
                if not is_captured:
                    continue
                
                _, _, frame_markers = ImpactUtils.detect_markers(frame=frame,
                                                                draw_markers=True)

                cv2.imshow("CAP", frame_markers)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        case type(str):
            frame = cv2.imread(source)
            _, _, frame_markers = ImpactUtils.detect_markers(frame=frame,
                                                                draw_markers=True)

            cv2.imshow("CAP", frame_markers)