# In the name of God

# Internal
from typing import List, Sequence, Tuple, Optional
import math
from datetime import datetime


# External
import cv2
import pyaudio
import numpy as np
import matplotlib as mpl
from ultralytics import YOLO
from matplotlib import pyplot as plt


class ImpactUtils:
    @staticmethod
    def detect_impact_sound(data, threshold):
        audio_data = np.frombuffer(data, dtype=np.int16)
        peak_amplitude = np.abs(audio_data).max()
        return peak_amplitude > threshold

    @staticmethod
    def generate_markers(
        marker_ids: List[int],
        file_path: str,
        marker_size=400, 
        number_of_markers_per_line=2,
        dictionary_name=cv2.aruco.DICT_6X6_250
    ) -> None:
        """
        Generates ARUCO markers and saves them as a grid in an image file.

        This method creates a figure with a grid of ARUCO markers based on the provided
        marker IDs, using the specified ARUCO dictionary. Each marker is rendered as a
        subplot in an A4-sized figure (35.08 x 24.80 cm). The markers are arranged in a
        grid with the specified number of markers per line. The resulting figure is saved
        to the given file path. If the marker_ids list is empty, no markers are generated,
        but an empty figure is still saved.

        Args:
            marker_ids (List[int]): List of marker IDs to generate. Each ID corresponds
                to a unique ARUCO marker in the dictionary.
            file_path (str): Path where the output image file will be saved (e.g., 'markers.png').
            marker_size (int, optional): Size of each marker in pixels (width and height).
                Defaults to 400.
            number_of_markers_per_line (int, optional): Number of markers to place in each
                row of the grid. Defaults to 2.
            dictionary_name (int, optional): Predefined ARUCO dictionary to use for marker
                generation (e.g., cv2.aruco.DICT_6X6_250). Defaults to cv2.aruco.DICT_6X6_250.

        Returns:
            None: The function saves the figure to the specified file path and does not
                return any value.

        Example:
            >>> import cv2
            >>> from utils import MathUtils
            >>> MathUtils.generate_markers([1, 2, 3, 4], 'markers.png', marker_size=200)
        """
        a4_dimensions = (35.08, 24.80)

        figure = plt.figure(figsize=a4_dimensions, dpi=100)
        number_of_lines = math.ceil(len(marker_ids) / number_of_markers_per_line)

        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_name)
        for sub_plot_index, marker_id in enumerate(marker_ids, 1):
            axis = figure.add_subplot(
                number_of_lines, 
                number_of_markers_per_line, 
                sub_plot_index
            )
            image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
            axis.imshow(image, cmap='gray', interpolation="nearest")
            plt.imshow(image, cmap=mpl.cm.gray, interpolation="nearest")
            axis.axis("off")

        plt.savefig(file_path, dpi=100, bbox_inches='tight')
        plt.close()

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
        cv2.createTrackbar(track_bar_names[0], window_name, 0, 179, ImpactUtils.nothing)
        cv2.createTrackbar(track_bar_names[1], window_name, 179, 179, ImpactUtils.nothing)
        cv2.createTrackbar(track_bar_names[2], window_name, 0, 255, ImpactUtils.nothing)
        cv2.createTrackbar(track_bar_names[3], window_name, 255, 255, ImpactUtils.nothing)
        cv2.createTrackbar(track_bar_names[4], window_name, 0, 255, ImpactUtils.nothing)
        cv2.createTrackbar(track_bar_names[5], window_name, 255, 255, ImpactUtils.nothing)

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
        Generates a mask for an image based on HSV color range and optionally applies it.

        This method converts the input image to the HSV color space and creates a binary mask
        where pixels within the specified HSV ranges are set to 255, and others to 0. If `apply`
        is True, it also generates a filtered image by applying the mask to the input image using
        a bitwise AND operation. The input image is assumed to be in BGR format, as expected by
        OpenCV's `cvtColor` function.

        Args:
            image (cv2.typing.MatLike): Input image in BGR format.
            hue (Tuple[int, int]): Min and max hue values (0-179) for the mask.
            saturation (Tuple[int, int]): Min and max saturation values (0-255) for the mask.
            value (Tuple[int, int]): Min and max value (brightness) values (0-255) for the mask.
            apply (bool, optional): If True, applies the mask to the input image to produce a
                filtered image. Defaults to True.

        Returns:
            Tuple[cv2.typing.MatLike, Optional[cv2.typing.MatLike]]: A tuple containing:
                - The binary mask (single-channel image with values 0 or 255).
                - The filtered image (same format as input) if `apply` is True, otherwise None.

        Example:
            >>> import cv2
            >>> img = cv2.imread('image.jpg')
            >>> mask, filtered = ImpactUtils.get_mask(img, (0, 10), (100, 255), (100, 255), apply=True)
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
        Detects ARUCO markers in a given frame and optionally draws them on a copy of the frame.

        This method converts the input frame to grayscale and uses the specified ARUCO dictionary
        to detect markers. It returns the detected marker corners, their IDs, and, if `draw_markers`
        is True, a copy of the input frame with the markers drawn on it. The input frame is assumed
        to be in BGR format, as expected by OpenCV's `cvtColor` function.

        Args:
            frame (cv2.typing.MatLike): Input frame in BGR format where markers are to be detected.
            draw_markers (bool, optional): If True, returns a copy of the frame with detected
                markers drawn on it. Defaults to False.
            aruco_dictionary (int, optional): Predefined ARUCO dictionary for marker detection.
                Defaults to `cv2.aruco.DICT_6X6_250`.

        Returns:
            Tuple[Sequence[cv2.typing.MatLike], cv2.typing.MatLike, cv2.typing.MatLike | None]:
                A tuple containing:
                - Sequence of detected marker corners (list of arrays with shape (4, 2)).
                - Array of marker IDs (shape (N, 1) where N is the number of detected markers,
                  or empty if none detected).
                - Copy of the input frame with drawn markers if `draw_markers` is True,
                  otherwise None.

        Example:
            >>> import cv2
            >>> frame = cv2.imread('frame.jpg')
            >>> corners, ids, marked_frame = MathUtils.detect_markers(
            ...     frame, draw_markers=True)
        """

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray_scale_frame,
            aruco_dictionary,
        )
        frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

        return corners, ids, (frame_markers if draw_markers else None)

    # Private methods
    @staticmethod
    def __find_center(marker_corners: Sequence[cv2.typing.MatLike]) -> Tuple[float, float]:
        # The Sequence only contains one item
        marker_corners = marker_corners[0]
        total = 0
        for corner in marker_corners:
            total += corner
        return total / len(marker_corners)


if __name__ == '__main__':
    pass