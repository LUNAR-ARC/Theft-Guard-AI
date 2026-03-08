import cv2
import os
from tqdm import tqdm

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/frames"

def extract_all():
    for category in ["shoplifting", "normal"]:
        video_dir = os.path.join(INPUT_DIR, category)
        save_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(save_dir, exist_ok=True)

        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        # Sir, this bar monitors the progress through the video files
        for v_file in tqdm(video_files, desc=f"Processing {category} videos"):
            video_path = os.path.join(video_dir, v_file)
            cap = cv2.VideoCapture(video_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            count = 0
            
            # This bar monitors frames within the current video
            with tqdm(total=total_frames, desc=f" > {v_file}", leave=False) as pbar:
                while True:
                    success, frame = cap.read()
                    if not success: break
                    
                    if count % 3 == 0:
                        frame_name = f"{v_file.split('.')[0]}_f{count}.jpg"
                        cv2.imwrite(os.path.join(save_dir, frame_name), frame)
                    
                    count += 1
                    pbar.update(1)
            cap.release()
    print("\nSir, frame extraction is fully complete.")

if __name__ == "__main__":
    extract_all()