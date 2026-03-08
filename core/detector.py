from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class TheftDetector:
    def __init__(self, yolo_path='models/yolov8n.pt'):
        # Sir, we use YOLOv8n for the <1s latency requirement
        self.model = YOLO(yolo_path)
        self.tracker = DeepSort(max_age=30, n_init=3)

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0: # 0 is the 'person' class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], conf, 'person'))
        
        return self.tracker.update_tracks(detections, frame=frame)