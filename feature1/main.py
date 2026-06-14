"""
CrowdIQ Backend — MJPEG + WebSocket Edition
============================================
Architecture:
  - Processing thread reads video, runs ML inference, draws overlays
    directly onto frames
  - Rendered frames go into a ring buffer → streamed as MJPEG over HTTP
  - Analytics numbers go to a PubSub → pushed to browser via WebSocket
  - Browser just displays the MJPEG stream (zero sync problem)
  - Sidebar stats come from WebSocket

Endpoints:
  GET  /video_feed     — MJPEG stream (plug into <img src="...">)
  WS   /ws/analytics   — JSON analytics per detection cycle
  GET  /analytics      — REST fallback

Install:
  pip install fastapi uvicorn opencv-python ultralytics keras numpy --break-system-packages
"""

import os
import cv2
import time
import json
import threading
import asyncio
import numpy as np
from collections import deque

import face_recognition
from ultralytics import YOLO

# Resolve paths relative to this file so the project runs from any directory
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_HERE)          # done_mp/
_MODELS    = os.path.join(_ROOT, "models")
_VIDEOS    = os.path.join(_ROOT, "videos")
_FACES     = os.path.join(_ROOT, "faces")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ─────────────────────────────────────────────
#  TUNABLE CONSTANTS
# ─────────────────────────────────────────────
VIDEO_PATH          = os.path.join(_VIDEOS, "queue_demo.mp4")
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720
TARGET_FPS          = 30
DETECTION_INTERVAL  = 5
MAX_FACE_BATCH      = 4
MJPEG_QUALITY       = 80   # JPEG quality 1-100 (higher = better quality, more bandwidth)

SMOOTH_ALPHA             = 0.35   # EMA factor: lower = smoother but slower to react
SERVICE_TIME_PER_PERSON  = 30     # seconds assumed per person served at counter

# ─────────────────────────────────────────────
#  RING BUFFER  — latest rendered JPEG frame
# ─────────────────────────────────────────────
class FrameRingBuffer:
    """Holds the latest JPEG-encoded frame. Writer always wins."""
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None   # bytes: JPEG

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
        """Called from the processing thread via loop.call_soon_threadsafe."""
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
#  SHARED ANALYTICS STATE  (for REST fallback)
# ─────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock                = threading.Lock()
        self.face_boxes          = []
        self.queue_boxes         = []
        self.head_bboxes         = []
        self.crowd_alert         = False
        self.long_wait_alert     = False  # Alert for 8+ minute wait times
        self.queue_head_counts   = {"Counter": 0, "Queue1": 0, "Queue2": 0, "Queue3": 0}
        self.queue_service_times = {"Counter": 0.0, "Queue1": 0.0, "Queue2": 0.0, "Queue3": 0.0}
        self.queue_start_times   = {"Counter": None, "Queue1": None, "Queue2": None, "Queue3": None}
        self.smoothed_counts     = {"Counter": 0.0, "Queue1": 0.0, "Queue2": 0.0, "Queue3": 0.0}
        self.avg_wait_times      = {"Counter": 0.0, "Queue1": 0.0, "Queue2": 0.0, "Queue3": 0.0}
        self.face_count          = 0
        self.fps                 = 0.0

state = SharedState()

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
print("[INFO] Loading models…")
yolo_queue = YOLO(os.path.join(_MODELS, "queue12.pt"))
yolo_head  = YOLO(os.path.join(_MODELS, "best.pt"))

def _load_face(path):
    img  = face_recognition.load_image_file(path)
    encs = face_recognition.face_encodings(img)
    return encs[0] if encs else None

known_face_encodings = [_load_face(os.path.join(_FACES, f"{i}.jpg")) for i in [1, 2, 3, 11, 22, 33]]
known_face_names     = ["Saif", "Momin", "Akif", "Saif", "Momin", "Akif"]
# Remove failed encodings
pairs = [(e, n) for e, n in zip(known_face_encodings, known_face_names) if e is not None]
if pairs:
    known_face_encodings, known_face_names = zip(*pairs)
    known_face_encodings = list(known_face_encodings)
    known_face_names     = list(known_face_names)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
CLASS_LABELS = ["Counter", "Queue1", "Queue2", "Queue3"]
COLORS = {
    "Counter": (255, 255,   0),
    "Queue1":  (  0, 100, 255),
    "Queue2":  (255, 105, 180),
    "Queue3":  (  0, 255, 255),
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
    """Draw all cached detection results onto the frame."""
    with state.lock:
        face_boxes  = list(state.face_boxes)
        queue_boxes = list(state.queue_boxes)
        head_bboxes = list(state.head_bboxes)
        crowd_alert = state.crowd_alert
        qhc         = dict(state.queue_head_counts)
        qst         = dict(state.queue_service_times)
        qawt        = dict(state.avg_wait_times)
        face_count  = state.face_count

    # Face recognition boxes
    for (l, t, r, b, name) in face_boxes:
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        if name:
            cv2.putText(frame, name, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Queue zone boxes
    for label, (x1, y1, x2, y2) in queue_boxes:
        col = COLORS.get(label, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # Head detection boxes
    for (x1, y1, x2, y2) in head_bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Queue stats (top-left)
    y_off = 30
    alert_parts = []
    for label in CLASS_LABELS:
        cnt = qhc[label]
        svc = qst[label]
        awt = qawt[label]
        awt_str = (f"{int(awt // 60)}m {int(awt % 60)}s" if awt >= 60 else f"{int(awt)}s")
        col = COLORS.get(label, (0, 255, 0))
        cv2.putText(frame,
            f"{label} | People: {cnt} | Svc: {svc:.1f}s | Avg Wait: ~{awt_str}",
            (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)
        y_off += 26
        if cnt > 5:
            alert_parts.append(f"{label} FULL!")

    # Face count
    cv2.putText(frame, f"Faces: {face_count}",
                (FRAME_WIDTH // 2 - 60, FRAME_HEIGHT - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (155, 75, 0), 2)

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

    # Queue overflow alerts
    if alert_parts:
        cv2.putText(frame, " | ".join(alert_parts),
                    (20, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (FRAME_WIDTH - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return frame


# ─────────────────────────────────────────────
#  LATEST FRAME  — shared between capture and detection threads
# ─────────────────────────────────────────────
class LatestFrame:
    """Thread-safe holder for the most recent decoded video frame."""
    def __init__(self):
        self._lock = threading.Lock()
        self._data = None   # (bgr_frame, rgb_frame) or None

    def write(self, bgr, rgb):
        with self._lock:
            self._data = (bgr, rgb)

    def read(self):
        with self._lock:
            return self._data

latest_frame = LatestFrame()


# ─────────────────────────────────────────────
#  THREAD 1 — capture + render  (always TARGET_FPS, never blocked by ML)
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

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Expose latest frame to detection worker
        latest_frame.write(frame, rgb)

        # FPS (rolling average, unaffected by detection cost)
        now = time.time()
        fps_deque.append(1.0 / max(now - prev_time, 1e-6))
        prev_time   = now
        current_fps = float(np.mean(fps_deque))
        with state.lock:
            state.fps = current_fps

        # Render cached detection state onto frame — zero ML cost here
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
LONG_WAIT_THRESHOLD = 8 * 60   # seconds

def detection_worker(stop_event: threading.Event, loop: asyncio.AbstractEventLoop):
    while not stop_event.is_set():
        t0 = time.time()

        data = latest_frame.read()
        if data is None:
            time.sleep(0.02)
            continue

        bgr, rgb = data
        cur = time.time()

        # Face recognition
        locs = face_recognition.face_locations(rgb, model="hog")[:MAX_FACE_BATCH]
        encs = face_recognition.face_encodings(rgb, locs)
        face_results = []
        for enc, loc in zip(encs, locs):
            matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.55)
            name = ""
            if True in matches:
                dists = face_recognition.face_distance(known_face_encodings, enc)
                name  = known_face_names[int(np.argmin(dists))]
            top, right, bottom, left = loc
            face_results.append((left, top, right, bottom, name))

        # Queue zones (YOLO)
        q_res   = yolo_queue(rgb, verbose=False)
        q_boxes = []
        if q_res:
            for box in q_res[0].boxes:
                cid = int(box.cls)
                if cid < len(CLASS_LABELS):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    q_boxes.append((CLASS_LABELS[cid], (x1, y1, x2, y2)))

        # Head detection (YOLO)
        h_res      = yolo_head(rgb, verbose=False)
        h_bboxes   = []
        new_counts = {l: 0 for l in CLASS_LABELS}
        if h_res:
            for box in h_res[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h_bboxes.append((x1, y1, x2, y2))
                for lbl, (qx1, qy1, qx2, qy2) in q_boxes:
                    if qx1 <= x1 and qy1 <= y1 and x2 <= qx2 and y2 <= qy2:
                        new_counts[lbl] += 1
                        break

        with state.lock:
            state.face_boxes  = face_results
            state.queue_boxes = q_boxes
            state.head_bboxes = h_bboxes
            state.crowd_alert = detect_crowd(h_bboxes)
            state.face_count  = len(locs)
            for lbl in CLASS_LABELS:
                raw = new_counts[lbl]
                if raw == 0:
                    state.smoothed_counts[lbl] = 0.0
                else:
                    state.smoothed_counts[lbl] = (
                        SMOOTH_ALPHA * raw +
                        (1 - SMOOTH_ALPHA) * state.smoothed_counts[lbl]
                    )
                smooth_int = round(state.smoothed_counts[lbl])
                state.queue_head_counts[lbl] = smooth_int
                if raw > 0 and state.queue_start_times[lbl] is None:
                    state.queue_start_times[lbl] = cur
                elif raw == 0:
                    state.queue_start_times[lbl]   = None
                    state.queue_service_times[lbl] = 0.0
                if state.queue_start_times[lbl]:
                    state.queue_service_times[lbl] = cur - state.queue_start_times[lbl]
                state.avg_wait_times[lbl] = smooth_int * SERVICE_TIME_PER_PERSON

        state.long_wait_alert = any(
            v >= LONG_WAIT_THRESHOLD for v in state.queue_service_times.values()
        )

        payload = {
            "fps":             round(state.fps, 1),
            "crowd_alert":     bool(state.crowd_alert),
            "long_wait_alert": bool(state.long_wait_alert),
            "face_count":      int(state.face_count),
            "counts":          {k: int(v) for k, v in state.queue_head_counts.items()},
            "wait_times":      {k: round(float(v), 1) for k, v in state.queue_service_times.items()},
            "avg_wait_times":  {k: int(v) for k, v in state.avg_wait_times.items()},
        }
        loop.call_soon_threadsafe(pubsub.publish, payload)

        # Run as fast as inference allows; no artificial throttle needed
        _ = time.time() - t0


# ─────────────────────────────────────────────
#  ENGINE MANAGER
# ─────────────────────────────────────────────
class Engine:
    def __init__(self):
        self._stop       = threading.Event()
        self._cap        = None
        self._cr_thread  = None   # capture + render
        self._det_thread = None   # detection

    def start(self, loop: asyncio.AbstractEventLoop):
        self._stop.clear()
        self._cap = cv2.VideoCapture(VIDEO_PATH)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

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
        print("[ENGINE] Started — render and detection threads running independently.")

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


app = FastAPI(title="CrowdIQ API", lifespan=lifespan)

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
    """Yields MJPEG multipart frames as fast as the ring buffer is updated."""
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
                # keep-alive ping
                await websocket.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        print("[WS] client disconnected")
    except Exception as e:
        print(f"[WS] error: {e}")
    finally:
        pubsub.unsubscribe(q)


# ────────────────────────────────────────────
#  REST FALLBACK  /analytics
# ────────────────────────────────────────────
@app.get("/analytics")
async def get_analytics():
    with state.lock:
        return JSONResponse({
            "fps":            round(state.fps, 1),
            "crowd_alert":    state.crowd_alert,
            "long_wait_alert": state.long_wait_alert,
            "cleanliness":    "Clean",
            "face_count":     state.face_count,
            "counts":         dict(state.queue_head_counts),
            "wait_times":     {k: round(v, 1) for k, v in state.queue_service_times.items()},
            "avg_wait_times": {k: int(v) for k, v in state.avg_wait_times.items()},
            "emotions":       [],
        })


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")