# In the name of God
import cv2
import numpy as np
import ImpactUtils

def nothing(x):
    pass

def adjust_color_mask(
        source: int | str,
        # mask_file_path: str
    ) -> None:
    """
        Gives an interactive windows where you can adjust and fine
        tune your mask based on either a video feedback or a picture
    """
    # Creating the track_bars:
    track_bar_window_name = "TrackBar"
    ImpactUtils.create_track_bar(window_name=track_bar_window_name)

    match type(source):        
        case int:
            video_capture = cv2.VideoCapture(source)
            while True:
                is_captured, frame = video_capture.read()
                if not is_captured:
                    continue
                # Get trackbar positions
                track_bar_position = ImpactUtils.get_track_bar_position(
                    window_name=track_bar_window_name
                )
                mask, filtered_image = ImpactUtils.get_mask(frame, *track_bar_position)

                # Display the result
                cv2.imshow('Filtered Image', filtered_image)
                cv2.imshow('Mask', mask)

                if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                    break

            cv2.destroyAllWindows()
