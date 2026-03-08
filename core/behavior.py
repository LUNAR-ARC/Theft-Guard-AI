import torch
import cv2
import numpy as np
from collections import deque
from model import CNNLSTMModel

class BehaviorEngine:
    # Sir, we added 'invert_labels' here to fix the TypeError
    def __init__(self, model_path='models/cnn_lstm_theft_model.pth', invert_labels=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNLSTMModel(num_classes=2)
        
        # Loading weights with the new security standard
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        except:
            state_dict = torch.load(model_path, map_location=self.device)
            
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        
        self.invert_labels = invert_labels
        self.history = {}
        self.buffer_size = 10

    def analyze(self, frame, track_id, bbox):
        try:
            x1, y1, x2, y2 = bbox
            # Focusing on the upper body for better hand-motion detection
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0: return 0.0
            
            img = cv2.resize(crop, (64, 64))
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            
            if track_id not in self.history:
                self.history[track_id] = deque(maxlen=self.buffer_size)
            
            self.history[track_id].append(img)
            
            if len(self.history[track_id]) == self.buffer_size:
                seq = torch.stack(list(self.history[track_id])).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(seq)
                    probs = torch.softmax(output, dim=1)[0]
                    
                    # Sir, if invert_labels is True, we flip the classes
                    if self.invert_labels:
                        risk = probs[0].item() # Class 0 becomes Theft
                    else:
                        risk = probs[1].item() # Class 1 stays Theft
                        
                    return float(risk)
            return 0.0
        except Exception as e:
            return 0.0