import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
import os
import math

def generate_aruco_pdf(
    num_markers_per_page: int,
    num_pages: int,
    output_filename: str = "A4_ArUco_Markers.pdf",
    aruco_dict_name: int = cv2.aruco.DICT_4X4_50,
):
    """Generates a multi-page PDF with a specified number of ArUco markers per page.

    This function automatically calculates the optimal grid layout to maximize marker 
    size on an A4 page. It then generates the requested number of pages, each 
    containing a grid of markers with sequential IDs.

    Args:
        num_markers_per_page (int): The number of markers to place on each page.
        num_pages (int): The total number of pages to generate.
        output_filename (str): The path for the output PDF file.
        aruco_dict_name (int): The OpenCV ArUco dictionary to use for generation.

    Side Effects:
        - Creates a PDF file at the specified path.
        - Prints generation progress to the console.
        - Temporarily creates and deletes PNG image files in the current directory.
    """
    if num_markers_per_page <= 0 or num_pages <= 0:
        print("Error: Number of markers and pages must be positive integers.")
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_name)
    page_width, page_height = A4
    margin = 10 * mm
    gap = 5 * mm

    best_layout = (0, 0)
    max_marker_size = 0
    print(f"Calculating optimal grid for {num_markers_per_page} markers per page...")
    
    for cols in range(1, num_markers_per_page + 1):
        rows = math.ceil(num_markers_per_page / cols)
        size_w = (page_width - 2 * margin - (cols - 1) * gap) / cols
        size_h = (page_height - 2 * margin - (rows - 1) * gap) / rows
        current_size = min(size_w, size_h)

        if current_size > max_marker_size:
            max_marker_size = current_size
            best_layout = (rows, cols)

    rows, cols = best_layout
    marker_size = max_marker_size
    print(f"Optimal layout: {rows} rows x {cols} columns.")
    print(f"Marker Size: {marker_size/mm:.1f} x {marker_size/mm:.1f} mm per marker.")

    canva = canvas.Canvas(output_filename, pagesize=A4)
    content_width = cols * marker_size + (cols - 1) * gap
    content_height = rows * marker_size + (rows - 1) * gap
    x_offset = (page_width - content_width) / 2
    y_offset = (page_height - content_height) / 2
    
    marker_id_counter = 0

    for page_num in range(num_pages):
        print(f"Generating Page {page_num + 1} of {num_pages}...")
        
        for i in range(num_markers_per_page):
            row_idx = i // cols
            col_idx = i % cols
            
            pos_x = x_offset + col_idx * (marker_size + gap)
            pos_y = y_offset + (rows - 1 - row_idx) * (marker_size + gap)
            
            img_filename = f"marker_{marker_id_counter}.png"
            
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id_counter, 500)
            cv2.imwrite(img_filename, marker_img)
            
            canva.drawImage(img_filename, pos_x, pos_y, width=marker_size, height=marker_size)
            
            label = f"ID: {marker_id_counter}"
            canva.setFont("Helvetica", 8)
            canva.drawCentredString(pos_x + marker_size / 2, pos_y - 4 * mm, label)
            
            os.remove(img_filename)
            marker_id_counter += 1

        if page_num < num_pages - 1:
            canva.showPage()

    canva.save()
    total_markers = num_markers_per_page * num_pages
    print(f"Successfully created {output_filename} with {total_markers} total markers across {num_pages} pages.")


if __name__ == "__main__":
    try:
        markers_per_page_count = int(input("Enter the number of ArUco markers PER PAGE: "))
        page_count = int(input("Enter the total number of PAGES: "))
        
        generate_aruco_pdf(
            num_markers_per_page=markers_per_page_count,
            num_pages=page_count
        )
    except ValueError:
        print("Invalid input. Please enter whole numbers.")
    except Exception as e:
        print(f"An error occurred: {e}")