# In the name of God

import cv2
import math
import matplotlib as mpl
from typing import List
from matplotlib import pyplot as plt

def generate_markers(marker_ids: List[int],
                     file_path: str,
                     marker_size=400, 
                     number_of_markers_per_line=4,
                     dictionary=cv2.aruco.DICT_6X6_250) -> None:
    """
        Generates aruco markers withe the given ids and
        stores them in the given pdf file path
    """    
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary)

    a4_dimensions = (24.80, 35.08)
    figure = plt.figure(figsize=a4_dimensions, dpi=100)
    number_of_lines = math.ceil(len(marker_ids) / number_of_markers_per_line)

    print(number_of_lines)

    for sub_plot_index, marker_id in enumerate(marker_ids, 1):
        axis = figure.add_subplot(
            number_of_lines, 
            number_of_markers_per_line, 
            sub_plot_index
        )
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        plt.imshow(image, cmap=mpl.cm.gray, interpolation="nearest")
        axis.axis("off")
    
    plt.savefig(file_path, dpi=100)
    plt.show()


if __name__ == '__main__':
    generate_markers(
        marker_ids=[1, 2, 3, 4],
        file_path="markers.pdf",
        marker_size=400
    )