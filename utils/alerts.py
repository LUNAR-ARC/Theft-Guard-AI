import cv2
import time
import threading
from playsound import playsound

class AlertManager:
    def __init__(self, cooldown=8):
        self.last_alert = 0
        self.cooldown = cooldown

    def trigger(self, sound_path='utils/alarm.mp3'):
        if time.time() - self.last_alert > self.cooldown:
            threading.Thread(target=lambda: playsound(sound_path), daemon=True).start()
            self.last_alert = time.time()

    def draw(self, frame, track_id, risk, bbox):
        x1, y1, x2, y2 = bbox
        color = (0, 0, 255) if risk > 0.85 else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} Risk:{risk:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)