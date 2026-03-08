import os, cv2, base64, requests, threading, time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# --- GROQ CREDENTIALS ---
GROQ_API_KEY = "YOUR API"
API_URL = "API URL"

# Local Perception Engine
yolo_model = YOLO('yolov8n.pt')
app.config['UPLOAD_FOLDER'] = 'data/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global States
current_source = 0
latest_intel = "Monitoring Local Feed... Cloud Bridge Active."
is_alert_active = False
source_changed = False
analysis_lock = threading.Lock()
is_analyzing = False
current_environment = "retail"  # default — changed by user via /set_env route

# --- Per-environment prompt definitions ---
ENVIRONMENT_PROMPTS = {
    "retail": (
        "You are a retail store security AI.\n"
        "Look for shoplifting signs: hiding items in clothing or bags, removing price tags, "
        "bypassing checkout, swapping labels, or acting nervously near shelves.\n"
    ),
    "office": (
        "You are an office security AI.\n"
        "Look for theft signs: someone taking equipment (laptops, phones, hard drives) "
        "that doesn't belong to them, accessing restricted drawers or cabinets without authorization, "
        "photographing confidential documents, or loitering near other people's desks suspiciously.\n"
    ),
    "home": (
        "You are a home security AI.\n"
        "Look for theft signs: someone taking valuables (jewelry, cash, electronics, keys), "
        "searching through drawers or cupboards they wouldn't normally access, "
        "acting furtively or checking if they're being watched, or carrying items out without permission.\n"
    ),
    "warehouse": (
        "You are a warehouse security AI.\n"
        "Look for theft signs: concealing stock items in personal bags or clothing, "
        "removing goods without scanning, tampering with packaging, "
        "or moving inventory in unusual patterns.\n"
    ),
    "parking": (
        "You are a parking lot security AI.\n"
        "Look for vehicle theft signs: attempting to break into a car, smashing windows, "
        "tampering with locks or tires, someone looking into multiple vehicles suspiciously, "
        "or removing items from an unlocked car.\n"
    ),
}


def build_prompt(environment):
    """Return the correct AI prompt for the chosen environment."""
    base = ENVIRONMENT_PROMPTS.get(environment, ENVIRONMENT_PROMPTS["retail"])
    return (
        base +
        "\nYou MUST respond in exactly this format:\n"
        "VERDICT: [SAFE or RISK]\n"
        "REASON: [one sentence explanation]\n\n"
        "Do not add anything else."
    )

# --- Persistent HTTP session with auto-retry ---
# This reuses the TCP connection instead of opening a new one every call,
# which prevents the "write operation timed out" error on slow networks.
_session = requests.Session()
_retry = Retry(
    total=3,               # retry up to 3 times
    backoff_factor=1,      # wait 1s, 2s, 4s between retries
    status_forcelist=[429, 500, 502, 503, 504],  # retry on these HTTP errors
    allowed_methods=["POST"]
)
_session.mount("https://", HTTPAdapter(max_retries=_retry))


def compress_frame(frame, max_width=640, quality=75):
    """Shrink + compress frame before sending — reduces payload size ~70%,
    which is the main cause of write timeouts on slow connections."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def call_groq_vision(frame):
    global latest_intel, is_alert_active, is_analyzing
    try:
        base64_img = compress_frame(frame)

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_prompt(current_environment)  # dynamic per environment
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }],
            "temperature": 0.1,
            "max_tokens": 100
        }

        # Increased timeout: (connect_timeout, read_timeout)
        # connect=5s  — time to establish TCP connection
        # read=30s    — time to wait for full response after connecting
        response = _session.post(API_URL, json=payload, headers=headers, timeout=(5, 30))

        if response.status_code == 200:
            res_text = response.json()['choices'][0]['message']['content'].strip()

            # FIX 5: Parse structured VERDICT line reliably
            verdict_line = ""
            reason_line = ""
            for line in res_text.splitlines():
                if line.upper().startswith("VERDICT:"):
                    verdict_line = line.split(":", 1)[-1].strip().upper()
                elif line.upper().startswith("REASON:"):
                    reason_line = line.split(":", 1)[-1].strip()

            if not verdict_line:
                # Fallback: keyword scan if model didn't follow format
                suspicious_keywords = ["RISK", "ALERT", "STEALING", "THEFT", "SUSPICIOUS", "CONCEALING", "SHOPLIFTING"]
                verdict_line = "RISK" if any(w in res_text.upper() for w in suspicious_keywords) else "SAFE"
                reason_line = res_text  # show raw response as reason

            is_alert_active = (verdict_line == "RISK")
            latest_intel = f"VERDICT: {verdict_line} | {reason_line}" if reason_line else f"VERDICT: {verdict_line}"

            print(f"--- AI VERDICT: {latest_intel} ---")

        else:
            # FIX 6: Show API errors in the UI instead of silently failing
            latest_intel = f"API Error {response.status_code}: {response.text[:120]}"
            print(f"--- GROQ ERROR: {response.status_code} {response.text} ---")

    except requests.exceptions.ConnectTimeout:
        latest_intel = "VERDICT: TIMEOUT | Could not connect to Groq API (check internet)"
        print("--- TIMEOUT: Could not connect to Groq ---")
    except requests.exceptions.ReadTimeout:
        latest_intel = "VERDICT: TIMEOUT | Groq API took too long to respond — will retry next frame"
        print("--- TIMEOUT: Groq response took too long ---")
    except requests.exceptions.ConnectionError as e:
        latest_intel = "VERDICT: OFFLINE | No internet connection detected"
        print(f"--- CONNECTION ERROR: {e} ---")
    except Exception as e:
        latest_intel = f"Analysis error: {str(e)[:100]}"
        print(f"--- EXCEPTION: {e} ---")
    finally:
        with analysis_lock:
            is_analyzing = False  # Always release lock


def gen_frames():
    global current_source, source_changed, is_alert_active, is_analyzing
    cap = cv2.VideoCapture(current_source)
    frame_count = 0

    while True:
        # FIX 7: Detect source switch and reload capture object
        if source_changed:
            cap.release()
            cap = cv2.VideoCapture(current_source)
            frame_count = 0
            is_alert_active = False
            source_changed = False

        success, frame = cap.read()

        # FIX 8: Loop video instead of breaking (for uploaded files)
        if not success:
            if isinstance(current_source, str) and os.path.isfile(current_source):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to start
                continue
            else:
                break  # Webcam disconnect — genuinely stop

        frame_count += 1

        # Local YOLO Tracking
        results = yolo_model.track(frame, persist=True, verbose=False)

        # Trigger Cloud Analysis every 90 frames (~3s at 30fps).
        # Slower cadence = smaller queue = fewer timeout errors on slow networks.
        if frame_count % 90 == 0:
            with analysis_lock:
                if not is_analyzing:
                    is_analyzing = True
                    threading.Thread(
                        target=call_groq_vision,
                        args=(frame.copy(),),
                        daemon=True
                    ).start()

        # Dashboard Rendering
        # FIX 10: Red boxes on RISK, purple on SAFE — clear visual feedback
        color = (50, 50, 247) if is_alert_active else (187, 154, 247)
        label = "!! RISK DETECTED !!" if is_alert_active else "MONITORING"

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


@app.route('/set_env', methods=['POST'])
def set_env():
    global current_environment, latest_intel, is_alert_active
    data = request.get_json()
    env = data.get('environment', 'retail')
    if env in ENVIRONMENT_PROMPTS:
        current_environment = env
        latest_intel = f"Environment switched to: {env.upper()}. Monitoring..."
        is_alert_active = False
        print(f"--- ENVIRONMENT SET TO: {env} ---")
        return jsonify({"success": True, "environment": env})
    return jsonify({"success": False, "error": "Unknown environment"}), 400


@app.route('/get_env')
def get_env():
    return jsonify({"environment": current_environment})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')   


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_intel')
def get_intel():
    return jsonify({"intel": latest_intel, "danger": is_alert_active})


@app.route('/upload', methods=['POST'])
def upload():
    global current_source, source_changed
    file = request.files['file']
    path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    file.save(path)
    current_source = path
    source_changed = True   # Signal gen_frames() to reload
    return jsonify({"success": True})


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)