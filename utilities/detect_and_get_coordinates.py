import cv2
import numpy as np
from ultralytics import YOLO
from correct_prespective import correct_perspective

def detect_and_get_coordinates(image, model, pixel_to_cm_ratio):
    """
    Runs YOLO object detection and maps pixel coordinates to real-world coordinates.

    Args:
        image (np.ndarray): The input image (should be the perspective-corrected view).
        model (YOLO): The pre-trained YOLO model object.
        pixel_to_cm_ratio (tuple): A tuple (ratio_x, ratio_y) for converting pixels
                                   to centimeters.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The image with bounding boxes and coordinates drawn on it.
            - list: A list of tuples, where each tuple contains the (x, y)
                    real-world coordinates in cm for each detected ball.
    """
    results = model.predict(image, verbose=False)
    annotated_image = image.copy()
    real_world_coords = []
    
    ratio_x, ratio_y = pixel_to_cm_ratio

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            center_x_px = (x1 + x2) / 2
            center_y_px = (y1 + y2) / 2
            
            real_x_cm = center_x_px * ratio_x
            real_y_cm = center_y_px * ratio_y
            real_world_coords.append((real_x_cm, real_y_cm))
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"Ball: ({real_x_cm:.1f}, {real_y_cm:.1f}) cm"
            
            cv2.putText(
                annotated_image, label, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            
    return annotated_image, real_world_coords

if __name__ == "__main__":
    MARKER_IDS_TO_FIND = [1, 3, 4, 6] 
    WALL_WIDTH_CM = 300  
    WALL_HEIGHT_CM = 200 

    OUTPUT_IMAGE_WIDTH_PX = 600
    OUTPUT_IMAGE_HEIGHT_PX = 400
    
    YOLO_MODEL_PATH = 'path.pt'
    
    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    print("Calculating coordinate mapping ratio...")
    px_to_cm_x = WALL_WIDTH_CM / OUTPUT_IMAGE_WIDTH_PX
    px_to_cm_y = WALL_HEIGHT_CM / OUTPUT_IMAGE_HEIGHT_PX
    ratio = (px_to_cm_x, px_to_cm_y)
    
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("Running... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        warped_image = correct_perspective(
            frame, 
            MARKER_IDS_TO_FIND, 
            (OUTPUT_IMAGE_WIDTH_PX, OUTPUT_IMAGE_HEIGHT_PX)
        )
        
        cv2.imshow("Original Camera Feed", frame)
        
        if warped_image is not None:
            final_image, coords = detect_and_get_coordinates(warped_image, yolo_model, ratio)
            
            if coords:
                print(f"Real-time coordinates (cm):  X={coords[0][0]:.1f}, Y={coords[0][1]:.1f}", end='\r')

            cv2.imshow("Final Result", final_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("\nApplication closed.")