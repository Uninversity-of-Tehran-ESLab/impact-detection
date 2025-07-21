import cv2
import os

def extract_sampled_frames(video_path, output_folder, sample_rate):
    """
    Extracts frames from a single video file at a given sample rate.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    saved_frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break

        if frame_count % sample_rate == 0:
            filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved_frame_count += 1

        frame_count += 1

    video.release()
    print(f"-> Extracted {saved_frame_count} frames from {os.path.basename(video_path)}")

def process_videos_in_folder(input_folder, output_folder, sample_rate):
    """
    Finds all videos in an input folder and processes them.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    
    print(f"Scanning for videos in '{input_folder}'...")
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(input_folder, filename)
            extract_sampled_frames(video_path, output_folder, sample_rate)

if __name__ == '__main__':
    input_videos_folder = "./videos"
    output_frames_folder = "dataset/all_frames"
    
    frame_sample_rate = 20

    process_videos_in_folder(input_videos_folder, output_frames_folder, frame_sample_rate)