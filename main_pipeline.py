import cv2
from core.detector import TheftDetector
from core.behavior import BehaviorEngine
from utils.alerts import AlertManager

def main():
    detector = TheftDetector()
    engine = BehaviorEngine()
    alerts = AlertManager()
    
    cap = cv2.VideoCapture(0) # Change to filename for offline video
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        tracks = detector.process_frame(frame)
        
        for track in tracks:
            if not track.is_confirmed(): continue
            
            tid = track.track_id
            bbox = [int(x) for x in track.to_ltrb()]
            
            risk = engine.analyze(frame, tid, bbox)
            
            if risk > 0.8: # Threshold from config
                alerts.trigger()
                
            alerts.draw(frame, tid, risk, bbox)
            
        cv2.imshow("TheftGuardAI - Live System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()