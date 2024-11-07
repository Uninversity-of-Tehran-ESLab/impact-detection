# In the name of God

# Internal
from typing import Sequence, Tuple, Optional

# External
import cv2
import numpy as np

class ImpactUtils:
    @staticmethod
    def __find_center(marker_corners: Sequence[cv2.typing.MatLike]) -> Tuple[float, float]:
        # The Sequence only contains one item
        marker_corners = marker_corners[0]
        total = 0
        for corner in marker_corners:
            total += corner
        return total / len(marker_corners)

    @staticmethod
    def sort_markers(markers_corners: Sequence[cv2.typing.MatLike]) -> Sequence[cv2.typing.MatLike]:
        # Grouping two left makers and two right markers
        markers_corners = sorted(
            markers_corners,
            key=lambda x: ImpactUtils.__find_center(x)[0]
        )

        left_markers = markers_corners[0:2]
        right_markers = markers_corners[2:4]

        # Sorting based on height
        left_markers = sorted(
            left_markers,
            key=lambda x: ImpactUtils.__find_center(x)[1]
        )
        
        right_markers = sorted(
            right_markers,
            key=lambda x: ImpactUtils.__find_center(x)[1]
        )

        return np.concatenate((left_markers, right_markers))

    @staticmethod
    def draw_transformed_perspective(
            frame: Sequence[cv2.typing.MatLike],
            corners: Sequence[cv2.typing.MatLike],
            width: int = 450,
            height: int = 300
        ) -> cv2.typing.MatLike: 

        width, height = 450, 300
        perspective = np.float32(
            [[0, 0], [0, height], 
            [width, 0], [width, height]]
        )

        corners = ImpactUtils.sort_markers(corners)

        points = np.array([ImpactUtils.__find_center(corner) for corner in corners])

        squeezed_points =np.squeeze(points)
        matrix = cv2.getPerspectiveTransform(squeezed_points, perspective)
        return cv2.warpPerspective(frame, matrix, (width, height))


    @staticmethod
    def nothing(x: any) -> None:
        """
        Does nothing!
        """
        pass

    @staticmethod
    def create_track_bar(
        window_name: str = "TrackBars",
        track_bar_names: Tuple[str, ...] = [
            "Lower H",
            "Upper H",
            "Lower S",
            "Upper S",
            "Lower V",
            "Upper V",
        ],
    ) -> None:
        """
            Creates a track bar for setting hsv values 
        """
        cv2.namedWindow(window_name)
        cv2.createTrackbar(track_bar_names[0], window_name, 0, 179, nothing)
        cv2.createTrackbar(track_bar_names[1], window_name, 179, 179, nothing)
        cv2.createTrackbar(track_bar_names[2], window_name, 0, 255, nothing)
        cv2.createTrackbar(track_bar_names[3], window_name, 255, 255, nothing)
        cv2.createTrackbar(track_bar_names[4], window_name, 0, 255, nothing)
        cv2.createTrackbar(track_bar_names[5], window_name, 255, 255, nothing)

    @staticmethod
    def get_track_bar_position(
        window_name: str = "TrackBars",
        track_bar_names: Tuple[str, ...] = [
            "Lower H",
            "Upper H",
            "Lower S",
            "Upper S",
            "Lower V",
            "Upper V",
        ],
    ) -> Tuple[Tuple[int, int], ...]:
        """
            Reads the data from the track bar
        """
        lower_h = cv2.getTrackbarPos(track_bar_names[0], window_name)
        upper_h = cv2.getTrackbarPos(track_bar_names[1], window_name)
        lower_s = cv2.getTrackbarPos(track_bar_names[2], window_name)
        upper_s = cv2.getTrackbarPos(track_bar_names[3], window_name)
        lower_v = cv2.getTrackbarPos(track_bar_names[4], window_name)
        upper_v = cv2.getTrackbarPos(track_bar_names[5], window_name)

        return ((lower_h, upper_h), (lower_s, upper_s), (lower_v, upper_v))

    @staticmethod
    def get_mask(
        image: cv2.typing.MatLike,
        hue: Tuple[int, int],
        saturation: Tuple[int, int],
        value: Tuple[int, int],
        apply: bool = True,
    ) -> Tuple[cv2.typing.MatLike, Optional[cv2.typing.MatLike]]:

        """
            Given an image and a range for hsv, generates a mask and can apply 
            it if apply is True
        """

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([hue[0], saturation[0], value[0]])
        upper_bound = np.array([hue[1], saturation[1], value[1]])

        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        filtered_image = cv2.bitwise_and(image, image, mask=mask) if apply else None

        return mask, filtered_image

    @staticmethod
    def detect_markers(
        frame: cv2.typing.MatLike,
        draw_markers: bool = False,
        aruco_dictionary: int = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
    ) -> Tuple[Sequence[cv2.typing.MatLike], cv2.typing.MatLike, cv2.typing.MatLike | None]:
        """
        Detects the markers on the given frame and can optionally draw them
        on the frame as well!
        """

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray_scale_frame,
            aruco_dictionary,
        )
        frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

        return corners, ids, (frame_markers if draw_markers else None)
