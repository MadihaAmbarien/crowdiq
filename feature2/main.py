"""
CrowdIQ Backend — Gender-Aware Queue Intelligence (main1.py)
=============================================================
Feature 2: Separately monitors Male and Female queues.

Architecture: Decoupled render + detection threads (same as main.py)
  - capture_render_thread: reads video, renders overlays at TARGET_FPS — never blocked by ML
  - detection_worker:      runs YOLO inference independently, updates shared state

Queue Zones (queue12.pt):
  - Counter  — service counter
  - Male     — Queue 1, male-only lane
  - Female   — Queue 2, female-only lane
  - Queue3   — general overflow

Head counting (best.pt):
  - Heads detected inside Male zone   → Male count
  - Heads detected inside Female zone → Female count
  - Zone boundary enforces gender separation (no per-person gender classification needed)

Endpoints:
  GET  /video_feed    — MJPEG stream
  WS   /ws/analytics  — JSON analytics per detection cycle
  GET  /analytics     — REST fallback
"""

import os
import cv2
import time
import json
import threading
import asyncio
import numpy as np
from collections import deque

from ultralytics import YOLO

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)
_MODELS = os.path.join(_ROOT, "models")
_VIDEOS = os.path.join(_ROOT, "videos")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ─────────────────────────────────────────────
#  TUNABLE CONSTANTS
# ─────────────────────────────────────────────
VIDEO_PATH              = os.path.join(_VIDEOS, "queue_demo.mp4")
FRAME_WIDTH             = 1280
FRAME_HEIGHT            = 720
TARGET_FPS              = 30
MJPEG_QUALITY           = 80
SMOOTH_ALPHA            = 0.35   # EMA factor — lower = smoother, slower to react
SERVICE_TIME_PER_PERSON = 30     # seconds assumed per person served
LONG_WAIT_THRESHOLD     = 8 * 60 # 8 minutes in seconds

# ─────────────────────────────────────────────
#  RING BUFFER  — latest rendered JPEG frame
# ─────────────────────────────────────────────
class FrameRingBuffer:
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None

    def write(self, jpeg_bytes: bytes):
        with self._lock:
            self._frame = jpeg_bytes

    def read(self) -> bytes | None:
        with self._lock:
            return self._frame

frame_ring = FrameRingBuffer()

# ─────────────────────────────────────────────
#  PUB/SUB  — analytics to all WebSocket clients
# ─────────────────────────────────────────────
class PubSub:
    def __init__(self):
        self._lock = threading.Lock()
        self._queues: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=4)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        with self._lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass

    def publish(self, payload: dict):
        with self._lock:
            queues = list(self._queues)
        for q in queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(payload)
                except Exception:
                    pass

pubsub = PubSub()

# ─────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock                = threading.Lock()
        self.queue_boxes         = []
        self.head_bboxes         = []
        self.crowd_alert         = False
        self.long_wait_alert     = False
        self.queue_head_counts   = {"Counter": 0, "Male": 0, "Female": 0, "Queue3": 0}
        self.queue_service_times = {"Counter": 0.0, "Male": 0.0, "Female": 0.0, "Queue3": 0.0}
        self.queue_start_times   = {"Counter": None, "Male": None, "Female": None, "Queue3": None}
        self.smoothed_counts     = {"Counter": 0.0, "Male": 0.0, "Female": 0.0, "Queue3": 0.0}
        self.avg_wait_times      = {"Counter": 0.0, "Male": 0.0, "Female": 0.0, "Queue3": 0.0}
        self.fps                 = 0.0

state = SharedState()

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
print("[INFO] Loading models…")
yolo_queue = YOLO(os.path.join(_MODELS, "queue12.pt"))
yolo_head  = YOLO(os.path.join(_MODELS, "best.pt"))

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
CLASS_LABELS = ["Counter", "Male", "Female", "Queue3"]

COLORS = {
    "Counter": (0, 255, 255),    # Cyan
    "Male":    (255, 80,  0),    # Blue  (BGR)
    "Female":  (180, 105, 255),  # Pink  (BGR)
    "Queue3":  (0,  255,  0),    # Green
}

DISPLAY_LABELS = {
    "Counter": "Counter",
    "Male":    "Queue 1: Male",
    "Female":  "Queue 2: Female",
    "Queue3":  "Queue 3",
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def detect_crowd(hb):
    for i, (x1_1, y1_1, x2_1, y2_1) in enumerate(hb):
        hc = vc = 0
        for j, (x1_2, y1_2, x2_2, y2_2) in enumerate(hb):
            if i == j:
                continue
            if abs(y1_1 - y1_2) < 50 and x2_1 < x1_2 < x2_1 + 100:
                hc += 1
            if abs(x1_1 - x1_2) < 50 and y2_1 < y1_2 < y2_1 + 100:
                vc += 1
        if hc >= 3 and vc >= 3:
            return True
    return False


def render_overlays(frame, fps):
    with state.lock:
        queue_boxes  = list(state.queue_boxes)
        head_bboxes  = list(state.head_bboxes)
        crowd_alert  = state.crowd_alert
        long_wait    = state.long_wait_alert
        qhc          = dict(state.queue_head_counts)
        qst          = dict(state.queue_service_times)
        qawt         = dict(state.avg_wait_times)

    # Queue zone boxes
    for label, (x1, y1, x2, y2) in queue_boxes:
        col = COLORS.get(label, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, DISPLAY_LABELS.get(label, label),
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # Head detection boxes
    for (x1, y1, x2, y2) in head_bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Queue stats with black backing for readability
    y_off = 30
    alert_parts = []
    for label in CLASS_LABELS:
        cnt     = qhc[label]
        svc     = qst[label]
        awt     = qawt[label]
        awt_str = (f"{int(awt // 60)}m {int(awt % 60)}s" if awt >= 60 else f"{int(awt)}s")
        col     = COLORS.get(label, (0, 255, 0))
        text    = f"{DISPLAY_LABELS[label]} | People: {cnt} | Svc: {svc:.1f}s | Avg Wait: ~{awt_str}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.rectangle(frame, (15, y_off - th - 4), (20 + tw, y_off + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)
        y_off += 28
        if cnt > 5:
            alert_parts.append(f"{DISPLAY_LABELS[label]} FULL!")

    # Crowd alert
    if crowd_alert:
        cv2.rectangle(frame,
                      (0, FRAME_HEIGHT - 100), (FRAME_WIDTH, FRAME_HEIGHT - 60),
                      (0, 0, 180), -1)
        cv2.putText(frame, "CROWD DETECTED!",
                    (20, FRAME_HEIGHT - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No Crowd Detected",
                    (20, FRAME_HEIGHT - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

    # Long wait alert banner
    if long_wait:
        cv2.rectangle(frame,
                      (0, FRAME_HEIGHT - 145), (FRAME_WIDTH, FRAME_HEIGHT - 105),
                      (0, 140, 255), -1)
        cv2.putText(frame, "LONG WAIT — ATTENTION NEEDED",
                    (20, FRAME_HEIGHT - 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Queue overflow alerts
    if alert_parts:
        cv2.putText(frame, " | ".join(alert_parts),
                    (20, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # FPS with black backing
    fps_text = f"FPS: {fps:.1f}"
    (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (FRAME_WIDTH - fw - 20, 10), (FRAME_WIDTH - 10, 10 + fh + 10), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (FRAME_WIDTH - fw - 15, 10 + fh + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return frame


# ─────────────────────────────────────────────
#  LATEST FRAME  — shared between capture and detection threads
# ─────────────────────────────────────────────
class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = None

    def write(self, bgr, rgb):
        with self._lock:
            self._data = (bgr, rgb)

    def read(self):
        with self._lock:
            return self._data

latest_frame = LatestFrame()


# ─────────────────────────────────────────────
#  THREAD 1 — capture + render  (constant TARGET_FPS, never blocked by ML)
# ─────────────────────────────────────────────
def capture_render_thread(cap, stop_event: threading.Event):
    fps_deque  = deque(maxlen=30)
    prev_time  = time.time()
    frame_time = 1.0 / TARGET_FPS

    while not stop_event.is_set():
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        latest_frame.write(frame, rgb)

        now = time.time()
        fps_deque.append(1.0 / max(now - prev_time, 1e-6))
        prev_time   = now
        current_fps = float(np.mean(fps_deque))
        with state.lock:
            state.fps = current_fps

        rendered = render_overlays(frame.copy(), current_fps)
        ok, buf  = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if ok:
            frame_ring.write(buf.tobytes())

        elapsed = time.time() - t0
        sleep_t = frame_time - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# ─────────────────────────────────────────────
#  THREAD 2 — detection worker  (ML inference, never blocks video)
# ─────────────────────────────────────────────
def detection_worker(stop_event: threading.Event, loop: asyncio.AbstractEventLoop):
    while not stop_event.is_set():
        data = latest_frame.read()
        if data is None:
            time.sleep(0.02)
            continue

        bgr, rgb = data
        cur = time.time()

        # Detect queue zones
        q_res   = yolo_queue(rgb, verbose=False, conf=0.38, iou=0.5)
        q_boxes = []
        if q_res:
            for box in q_res[0].boxes:
                cid = int(box.cls)
                if cid < len(CLASS_LABELS):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    q_boxes.append((CLASS_LABELS[cid], (x1, y1, x2, y2)))

        # Detect heads and assign to zones
        h_res      = yolo_head(rgb, verbose=False)
        h_bboxes   = []
        new_counts = {l: 0 for l in CLASS_LABELS}
        if h_res:
            for box in h_res[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h_bboxes.append((x1, y1, x2, y2))
                # Head centre-point membership test against each zone
                hcx, hcy = (x1 + x2) // 2, (y1 + y2) // 2
                for lbl, (qx1, qy1, qx2, qy2) in q_boxes:
                    if qx1 <= hcx <= qx2 and qy1 <= hcy <= qy2:
                        new_counts[lbl] += 1
                        break

        with state.lock:
            state.queue_boxes = q_boxes
            state.head_bboxes = h_bboxes
            state.crowd_alert = detect_crowd(h_bboxes)
            for lbl in CLASS_LABELS:
                raw = new_counts[lbl]
                # EMA smoothing — hard reset when zone clears
                if raw == 0:
                    state.smoothed_counts[lbl] = 0.0
                else:
                    state.smoothed_counts[lbl] = (
                        SMOOTH_ALPHA * raw +
                        (1 - SMOOTH_ALPHA) * state.smoothed_counts[lbl]
                    )
                smooth_int = round(state.smoothed_counts[lbl])
                state.queue_head_counts[lbl] = smooth_int

                # Service-time tracking on raw count (accurate presence detection)
                if raw > 0 and state.queue_start_times[lbl] is None:
                    state.queue_start_times[lbl] = cur
                elif raw == 0:
                    state.queue_start_times[lbl]   = None
                    state.queue_service_times[lbl] = 0.0
                if state.queue_start_times[lbl]:
                    state.queue_service_times[lbl] = cur - state.queue_start_times[lbl]

                # Avg wait for a new person joining now
                state.avg_wait_times[lbl] = smooth_int * SERVICE_TIME_PER_PERSON

        state.long_wait_alert = any(
            v >= LONG_WAIT_THRESHOLD for v in state.queue_service_times.values()
        )

        payload = {
            "fps":             round(state.fps, 1),
            "crowd_alert":     bool(state.crowd_alert),
            "long_wait_alert": bool(state.long_wait_alert),
            "face_count":      0,
            "counts":          {k: int(v) for k, v in state.queue_head_counts.items()},
            "wait_times":      {k: round(float(v), 1) for k, v in state.queue_service_times.items()},
            "avg_wait_times":  {k: int(v) for k, v in state.avg_wait_times.items()},
        }
        loop.call_soon_threadsafe(pubsub.publish, payload)


# ─────────────────────────────────────────────
#  ENGINE MANAGER
# ─────────────────────────────────────────────
class Engine:
    def __init__(self):
        self._stop       = threading.Event()
        self._cap        = None
        self._cr_thread  = None
        self._det_thread = None

    def start(self, loop: asyncio.AbstractEventLoop):
        self._stop.clear()
        self._cap = cv2.VideoCapture(VIDEO_PATH)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._det_thread = threading.Thread(
            target=detection_worker,
            args=(self._stop, loop),
            daemon=True,
        )
        self._det_thread.start()

        self._cr_thread = threading.Thread(
            target=capture_render_thread,
            args=(self._cap, self._stop),
            daemon=True,
        )
        self._cr_thread.start()
        print("[ENGINE] Gender-aware mode started — render and detection threads independent.")

    def stop(self):
        self._stop.set()
        if self._cr_thread:
            self._cr_thread.join(timeout=4)
        if self._det_thread:
            self._det_thread.join(timeout=4)
        if self._cap:
            self._cap.release()
        print("[ENGINE] Stopped.")


engine = Engine()


# ─────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.start(asyncio.get_running_loop())
    yield
    engine.stop()


app = FastAPI(title="CrowdIQ Gender Queue API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  MJPEG STREAM  /video_feed
# ─────────────────────────────────────────────
async def mjpeg_generator():
    boundary = b"--frame\r\n"
    while True:
        jpeg = frame_ring.read()
        if jpeg is None:
            await asyncio.sleep(0.02)
            continue
        yield (
            boundary
            + b"Content-Type: image/jpeg\r\n"
            + b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
            + jpeg
            + b"\r\n"
        )
        await asyncio.sleep(1.0 / TARGET_FPS)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ─────────────────────────────────────────────
#  WEBSOCKET  /ws/analytics
# ─────────────────────────────────────────────
@app.websocket("/ws/analytics")
async def ws_analytics(websocket: WebSocket):
    await websocket.accept()
    q = pubsub.subscribe()
    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=5.0)
                await websocket.send_text(json.dumps(payload))
            except asyncio.TimeoutError:
                await websocket.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        print("[WS] client disconnected")
    except Exception as e:
        print(f"[WS] error: {e}")
    finally:
        pubsub.unsubscribe(q)


# ─────────────────────────────────────────────
#  REST FALLBACK  /analytics
# ─────────────────────────────────────────────
@app.get("/analytics")
async def get_analytics():
    with state.lock:
        return JSONResponse({
            "fps":             round(state.fps, 1),
            "crowd_alert":     state.crowd_alert,
            "long_wait_alert": state.long_wait_alert,
            "cleanliness":     "Clean",
            "face_count":      0,
            "counts":          dict(state.queue_head_counts),
            "wait_times":      {k: round(v, 1) for k, v in state.queue_service_times.items()},
            "avg_wait_times":  {k: int(v) for k, v in state.avg_wait_times.items()},
            "emotions":        [],
        })


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
