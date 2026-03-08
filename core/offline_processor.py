import cv2
import os
from tqdm import tqdm
from core.detector import TheftDetector
from core.behavior import BehaviorEngine
from utils.alerts import AlertManager

class OfflineProcessor:
    def __init__(self):
        self.detector = TheftDetector()
        self.engine = BehaviorEngine()
        self.alerts = AlertManager()

    def process(self, video_path, output_name="processed_output.mp4"):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs("results", exist_ok=True)
        out = cv2.VideoWriter(f"results/{output_name}", 
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Sir, the tqdm bar ensures you can monitor the offline speed
        pbar = tqdm(total=total_frames, desc="Analyzing CCTV Footage")
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # EFFICIENCY HACK: Skip detection every 4 frames
            if frame_idx % 5 == 0:
                tracks = self.detector.process_frame(frame)
                for track in tracks:
                    if not track.is_confirmed(): continue
                    bbox = [int(x) for x in track.to_ltrb()]
                    risk = self.engine.analyze(frame, track.track_id, bbox)
                    self.alerts.draw(frame, track.track_id, risk, bbox)
            
            out.write(frame)
            frame_idx += 1
            pbar.update(1)

        cap.release()
        out.release()
        print(f"\nSir, offline processing finished. Saved to results/{output_name}")