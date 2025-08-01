import cv2
import os

OUTPUT_FOLDER = "recorded_videos"
BASE_FILENAME = "recording"
FILE_EXTENSION = ".avi"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20.0

def record_video():
    """
    Captures video from a USB camera, displays a live feed, and saves it.

    This function initializes the default camera, sets up a VideoWriter object,
    and enters a loop to record video. It automatically determines the next
    available filename in the specified output folder to prevent overwriting
    existing files.

    The recording is displayed in a window titled 'Recording...'. To stop the
    recording and save the file, the user must press the 'q' key while this
    window is in focus.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created directory: {OUTPUT_FOLDER}")

    file_number = 1
    while True:
        output_path = os.path.join(OUTPUT_FOLDER, f"{BASE_FILENAME}_{file_number}{FILE_EXTENSION}")
        if not os.path.exists(output_path):
            break
        file_number += 1

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    print(f"Recording started. Output will be saved to '{output_path}'.")
    print("Press 'q' in the video window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        writer.write(frame)
        cv2.imshow('Recording... (Press Q to stop)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\nRecording stopped. Video saved successfully to: {output_path}")

if __name__ == '__main__':
    record_video()