# 🔒 TheftGuardAI

> AI-powered theft detection system combining local YOLOv8 object detection with Groq API visual intelligence for real-time SAFE/RISK verdicts.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA%20Vision-purple?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=flat-square&logo=flask)

---

## 📌 Overview

TheftGuardAI is a dual-model theft detection system that processes video feeds using a locally trained YOLOv8 model for object detection, then passes suspicious frames to Groq's LLaMA Vision API for higher-level contextual analysis. The system outputs **SAFE** or **RISK** verdicts via a dark-themed Flask dashboard.

This is LUNAR's primary reference project — it demonstrates end-to-end ML pipeline integration, custom model training, and real-time inference on live video.

---

## 🧠 Architecture

```
Video Input (Webcam / File)
        │
        ▼
  YOLOv8 Detection
  (Local ONNX Model)
        │
  Object Bounding Boxes
        │
        ▼
  CNN-LSTM Classifier          ← Trained from scratch for behavioral patterns
  (Custom Anomaly Model)
        │
  Suspicious Frame Flagged?
        │
       YES
        │
        ▼
  Groq API (LLaMA Vision)     ← Cloud inference for contextual verdict
        │
        ▼
  SAFE / RISK Verdict
        │
        ▼
  Flask Dashboard              ← Live feed + verdict display
```

---

## 🗂️ Project Structure

```
TheftGuardAI/
├── models/
│   ├── yolov8_theft.pt          # Fine-tuned YOLOv8 weights
│   └── cnn_lstm_anomaly.h5      # Custom CNN-LSTM model
├── backend/
│   ├── app.py                   # Flask entry point
│   ├── detector.py              # YOLOv8 inference pipeline
│   ├── groq_client.py           # Groq API integration
│   └── cnn_lstm_model.py        # CNN-LSTM inference wrapper
├── frontend/
│   ├── templates/
│   │   └── index.html           # Dark-themed dashboard
│   └── static/
│       ├── style.css
│       └── dashboard.js
├── training/
│   ├── train_yolo.py            # YOLOv8 fine-tuning script
│   └── train_cnn_lstm.py        # Custom model training
├── utils/
│   └── frame_utils.py           # Frame sampling and preprocessing
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU (recommended) or CPU
- Groq API key ([Get one here](https://console.groq.com/))

### 1. Clone the Repository

```bash
git clone https://github.com/LUNAR-ARC/TheftGuardAI.git
cd TheftGuardAI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Linux/macOS
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**

```
flask
ultralytics
opencv-python
groq
tensorflow
numpy
Pillow
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
CAMERA_INDEX=0
CONFIDENCE_THRESHOLD=0.6
```

---

## 🤖 Model Training

### Train YOLOv8 (Fine-tuning)

```bash
python training/train_yolo.py \
  --data dataset/theft_data.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

### Train CNN-LSTM (From Scratch)

```bash
python training/train_cnn_lstm.py \
  --dataset_path dataset/sequences/ \
  --epochs 30 \
  --seq_length 16
```

Trained models are saved to the `models/` directory automatically.

---

## 🚀 Running the Application

```bash
python backend/app.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

The dashboard will display:
- Live camera feed with bounding box overlays
- Real-time YOLOv8 detection confidence
- LLaMA Vision contextual analysis
- Final **SAFE** / **RISK** verdict with timestamp log

---

## 🔌 Pipeline Flow (Detailed)

1. **Frame Capture** — OpenCV reads frames from webcam or video file at configurable FPS.
2. **YOLOv8 Detection** — Each frame is passed through the fine-tuned YOLOv8 model. Objects are detected and bounding boxes are returned.
3. **Behavioral Analysis** — Detected sequences are fed into the CNN-LSTM model, which captures temporal behavior patterns (e.g., loitering, reaching, concealment).
4. **Flag Trigger** — If confidence exceeds the threshold, the frame is flagged as suspicious.
5. **Groq Vision Call** — The flagged frame (base64 encoded) is sent to Groq's LLaMA Vision API with a structured prompt requesting a SAFE/RISK verdict and reasoning.
6. **Dashboard Update** — Flask emits a Socket.IO event to update the frontend in real time.

---

## 📡 API Reference

### `POST /analyze`

Manually submit a frame for analysis.

**Request:**
```json
{
  "frame": "<base64_encoded_image>"
}
```

**Response:**
```json
{
  "verdict": "RISK",
  "confidence": 0.87,
  "reasoning": "Individual observed concealing item under clothing.",
  "timestamp": "2025-04-12T14:32:01Z"
}
```

---

## 🛠️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Groq API authentication key |
| `CAMERA_INDEX` | `0` | OpenCV camera device index |
| `CONFIDENCE_THRESHOLD` | `0.6` | Minimum YOLOv8 confidence to trigger LLaMA |
| `FRAME_SAMPLE_RATE` | `5` | Process every Nth frame |

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Behavioral Model | Custom CNN-LSTM (TensorFlow/Keras) |
| Cloud Vision | Groq API — LLaMA Vision |
| Backend | Flask + Socket.IO |
| Frontend | HTML/CSS/JS (Dark Theme) |
| Video Processing | OpenCV |

---

## 📄 License

MIT License. See `LICENSE` for details.
