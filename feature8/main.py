import os, threading, time, asyncio, csv
import cv2, numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# ── Portable paths ────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
_ROOT           = os.path.dirname(_HERE)
_MODELS         = os.path.join(_ROOT, "models")
LOG_FILE        = os.path.join(_HERE, "emotion_log.csv")

CASCADE_PATH    = os.path.join(_MODELS, "haarcascade_frontalface_default.xml")
EMOTION_WEIGHTS = os.path.join(_MODELS, "emotion_model.h5")
QUEUE_MODEL     = os.path.join(_MODELS, "queue12.pt")

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_LABELS      = ["Counter", "Queue1", "Queue2", "Queue3"]
DISPLAY_LABELS    = {"Counter": "Counter", "Queue1": "Queue 1",
                     "Queue2": "Queue 2",  "Queue3": "Queue 3"}
EMOTION_DICT      = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                     3: "Happy",  4: "Neutral",   5: "Sad", 6: "Surprised"}
NEGATIVE_EMOTIONS = {"Angry", "Disgusted", "Fearful", "Sad"}

CONSEC_NEG_THRESHOLD = 3   # consecutive negative-dominant windows to flag
WINDOW_SECONDS       = 10  # analysis window
PAUSE_SECONDS      = 2     # result display pause
TARGET_FPS         = 30
YOLO_EVERY         = 30

ZONE_COLORS = {
    "Counter": (255, 255,   0),
    "Queue1":  (255,  80,   0),
    "Queue2":  (128,   0, 128),
    "Queue3":  (  0, 215, 255),
}

EMOTION_COLORS = {
    "Angry":     (0,  40, 220), "Disgusted": (0,  80, 200),
    "Fearful":   (0, 120, 210), "Sad":       (0,  60, 190),
    "Happy":     (0, 200,  80), "Neutral":   (160, 160, 160),
    "Surprised": (0, 190, 200),
}

# ── CSV ───────────────────────────────────────────────────────────────────────
def _init_csv():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "queue", "dominant_emotion",
                "neg_count", "flagged", "recommendation"
            ])

def _log_window(results: dict):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        for lbl, r in results.items():
            w.writerow([
                ts,
                DISPLAY_LABELS.get(lbl, lbl),
                r["dominant"],
                r["neg_count"],
                r["flagged"],
                "Service Quality Check Required" if r["flagged"] else "All Good",
            ])

# ── Keras emotion model ───────────────────────────────────────────────────────
def _build_emotion_model() -> Sequential:
    m = Sequential([
        Conv2D(32,  (3, 3), activation="relu", input_shape=(48, 48, 1)),
        Conv2D(64,  (3, 3), activation="relu"),
        MaxPooling2D((2, 2)), Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)), Dropout(0.25),
        Flatten(),
        Dense(1024, activation="relu"), Dropout(0.5),
        Dense(7, activation="softmax"),
    ])
    m.load_weights(EMOTION_WEIGHTS)
    return m

# ── Per-queue state ───────────────────────────────────────────────────────────
class QueueState:
    def __init__(self):
        self._lock       = threading.Lock()
        self._counts     = {e: 0 for e in EMOTION_DICT.values()}
        self.consec_neg  = 0          # consecutive windows with negative dominant
        self.flagged     = False
        self.last_dom    = "Neutral"  # dominant emotion of last closed window

    def record(self, emotion: str):
        with self._lock:
            self._counts[emotion] = self._counts.get(emotion, 0) + 1

    def evaluate_and_reset(self):
        """Close window: find dominant, update consecutive counter, reset counts."""
        with self._lock:
            total = sum(self._counts.values())
            dom   = max(self._counts, key=lambda e: self._counts.get(e, 0)) if total > 0 else "Neutral"
            neg   = sum(self._counts.get(e, 0) for e in NEGATIVE_EMOTIONS)
            self.last_dom = dom
            if dom in NEGATIVE_EMOTIONS:
                self.consec_neg += 1
            else:
                self.consec_neg = 0
            self.flagged  = (self.consec_neg >= CONSEC_NEG_THRESHOLD)
            self._counts  = {e: 0 for e in EMOTION_DICT.values()}
            return dom, neg, self.consec_neg, total

    def neg_count(self) -> int:
        with self._lock:
            return sum(self._counts.get(e, 0) for e in NEGATIVE_EMOTIONS)

# ── Phase constants ───────────────────────────────────────────────────────────
ANALYZING = "analyzing"
RESULT    = "result"

# ── Shared state with two-phase state machine ─────────────────────────────────
class SharedState:
    def __init__(self):
        self._lock        = threading.Lock()
        self.queues       = {lbl: QueueState() for lbl in CLASS_LABELS}
        self._phase       = ANALYZING
        self._phase_start = time.time()
        self.last_results: dict = {}
        self.fps          = 0.0

    def record(self, queue_label: str, emotion: str):
        with self._lock:
            if self._phase != ANALYZING:
                return          # don't collect during the 2s result pause
        if queue_label in self.queues:
            self.queues[queue_label].record(emotion)

    def tick(self):
        """Called every detection frame. Drives the phase state machine."""
        now = time.time()
        with self._lock:
            if self._phase == ANALYZING and now - self._phase_start >= WINDOW_SECONDS:
                # ── close window: evaluate → log → pause ──────────────────────
                results = {}
                for lbl, q in self.queues.items():
                    dom, neg, consec, total = q.evaluate_and_reset()
                    results[lbl] = {
                        "dominant":  dom,
                        "neg_count": neg,
                        "consec":    consec,
                        "flagged":   q.flagged,
                        "total":     total,
                    }
                self.last_results = results
                self._phase       = RESULT
                self._phase_start = now
                _log_window(results)

            elif self._phase == RESULT and now - self._phase_start >= PAUSE_SECONDS:
                # ── end of pause: start fresh analysis ────────────────────────
                self._phase       = ANALYZING
                self._phase_start = now

    def snapshot(self) -> dict:
        with self._lock:
            now   = time.time()
            phase = self._phase
            tl    = max(0.0, WINDOW_SECONDS - (now - self._phase_start)) if phase == ANALYZING else 0.0
            rtl   = max(0.0, PAUSE_SECONDS  - (now - self._phase_start)) if phase == RESULT    else 0.0
            lr    = dict(self.last_results)

        # read queue data outside lock (QueueState has its own lock)
        if phase == ANALYZING:
            q_data = {
                lbl: {"neg_count": q.neg_count(),
                      "consec":    q.consec_neg,
                      "flagged":   q.flagged}
                for lbl, q in self.queues.items()
            }
        else:
            q_data = {
                lbl: {"neg_count": lr.get(lbl, {}).get("neg_count", 0),
                      "consec":    lr.get(lbl, {}).get("consec",    0),
                      "flagged":   lr.get(lbl, {}).get("flagged",   False)}
                for lbl in CLASS_LABELS
            }
        return {
            "phase":        phase,
            "time_left":    round(tl,  1),
            "result_tl":    round(rtl, 1),
            "queue_data":   q_data,
            "last_results": lr,
        }

    def set_fps(self, v: float):
        with self._lock: self.fps = v

    def get_fps(self) -> float:
        with self._lock: return self.fps


shared = SharedState()

# ── Thread-safe stores ────────────────────────────────────────────────────────
class LatestFrame:
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None

    def set(self, f):
        with self._lock: self._frame = f.copy()

    def get(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


class LatestDetections:
    def __init__(self):
        self._lock        = threading.Lock()
        self._annotations = []
        self._zones       = {}

    def set(self, ann: list, zones: dict):
        with self._lock:
            self._annotations = list(ann)
            self._zones       = dict(zones)

    def get(self):
        with self._lock:
            return list(self._annotations), dict(self._zones)


latest_frame = LatestFrame()
latest_det   = LatestDetections()

# ── MJPEG ring buffer ─────────────────────────────────────────────────────────
class FrameRingBuffer:
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame: bytes | None = None
        self._event = threading.Event()

    def put(self, data: bytes):
        with self._lock: self._frame = data
        self._event.set()

    def get(self) -> bytes | None:
        self._event.wait(timeout=1.0)
        self._event.clear()
        with self._lock: return self._frame


ring = FrameRingBuffer()

# ── PubSub ────────────────────────────────────────────────────────────────────
class PubSub:
    def __init__(self):
        self._subs: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        async with self._lock: self._subs.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        async with self._lock:
            try: self._subs.remove(q)
            except ValueError: pass

    async def publish(self, msg: dict):
        async with self._lock:
            for q in self._subs:
                if q.full():
                    try: q.get_nowait()
                    except asyncio.QueueEmpty: pass
                await q.put(msg)


pubsub = PubSub()

# ── Zone helpers ──────────────────────────────────────────────────────────────
_zone_boxes: dict[str, tuple[int, int, int, int]] = {}

def _assign_zone(cx: int, cy: int) -> str:
    for lbl, (x1, y1, x2, y2) in _zone_boxes.items():
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return lbl
    if _zone_boxes:
        return min(_zone_boxes, key=lambda lbl: (
            (cx - (_zone_boxes[lbl][0] + _zone_boxes[lbl][2]) // 2) ** 2 +
            (cy - (_zone_boxes[lbl][1] + _zone_boxes[lbl][3]) // 2) ** 2
        ))
    return "Queue1"

# ── Thread 1: Webcam ──────────────────────────────────────────────────────────
def webcam_thread():
    # Try DirectShow first (best on Windows), then fall back. Try indices 0..3.
    cap = None
    for idx in (0, 1, 2):
        for backend, name in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")):
            c = cv2.VideoCapture(idx, backend)
            if c.isOpened():
                ok, frame = c.read()
                if ok and frame is not None:
                    print(f"[CAMERA] Opened index={idx} backend={name} size={frame.shape}")
                    cap = c
                    break
                c.release()
        if cap is not None:
            break

    if cap is None:
        print("[CAMERA] !! No webcam could be opened on indices 0/1/2 — check Windows privacy settings, close Teams/Zoom/Whatsapp, or uncover the camera.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ret, frame = cap.read()
        if ret:
            latest_frame.set(frame)

# ── Thread 2: ML detection ────────────────────────────────────────────────────
def detection_worker():
    cv2.ocl.setUseOpenCL(False)
    yolo  = YOLO(QUEUE_MODEL)
    model = _build_emotion_model()
    casc  = cv2.CascadeClassifier(CASCADE_PATH)
    fc, t0 = 0, time.time()

    while True:
        frame = latest_frame.get()
        if frame is None:
            time.sleep(0.01)
            continue

        fc += 1
        shared.tick()   # drive phase state machine

        # zone detection (infrequent — zones are stable)
        if fc % YOLO_EVERY == 1:
            res = yolo(frame, imgsz=640, verbose=False)[0]
            for box in res.boxes:
                ci = int(box.cls[0])
                if ci < len(CLASS_LABELS):
                    lbl = CLASS_LABELS[ci]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    _zone_boxes[lbl] = (x1, y1, x2, y2)

        # face + emotion detection (always runs for smooth face boxes)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = casc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        annotations: list[tuple] = []

        if len(faces):
            rois, valid = [], []
            for (x, y, w, h) in faces:
                roi = gray[max(0, y):y + h, max(0, x):x + w]
                if roi.size == 0:
                    continue
                rois.append(np.expand_dims(cv2.resize(roi, (48, 48)), -1))
                valid.append((x, y, w, h))

            if rois:
                batch = np.array(rois, dtype=np.float32) / 255.0
                preds = model.predict(batch, verbose=0)
                for (x, y, w, h), pred in zip(valid, preds):
                    emotion = EMOTION_DICT[int(np.argmax(pred))]
                    cx, cy  = x + w // 2, y + h // 2
                    qzone   = _assign_zone(cx, cy)
                    shared.record(qzone, emotion)   # no-op during RESULT phase
                    annotations.append((x, y, w, h, emotion, qzone))

        latest_det.set(annotations, dict(_zone_boxes))

        if fc % 10 == 0:
            shared.set_fps(10 / (time.time() - t0))
            t0 = time.time()

# ── Render helpers ────────────────────────────────────────────────────────────
def _draw_banner(out: np.ndarray, flagged_queues: list[str]):
    """Red banner at top when any queue is flagged."""
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 52), (0, 0, 180), -1)
    cv2.rectangle(out, (0, 0), (w, 52), (0, 0, 255), 2)
    names = ", ".join(DISPLAY_LABELS.get(q, q) for q in flagged_queues)
    txt   = f"!! SERVICE QUALITY CHECK REQUIRED  --  {names}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
    cv2.putText(out, txt, (w // 2 - tw // 2, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)


def _draw_dominant_word(out: np.ndarray, last_results: dict):
    """Top-right: dominant emotion word for the ONE queue that had detections."""
    if not last_results:
        return
    # pick the queue with the most face detections this window
    active = {lbl: r for lbl, r in last_results.items() if r.get("total", 0) > 0}
    if not active:
        return
    lbl, r  = max(active.items(), key=lambda x: x[1].get("total", 0))
    dom     = r.get("dominant", "Neutral").upper()
    color   = EMOTION_COLORS.get(r.get("dominant", "Neutral"), (200, 200, 200))
    h, w    = out.shape[:2]
    (tw, th), _ = cv2.getTextSize(dom, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
    x = w - tw - 14
    y = 70 + th
    cv2.rectangle(out, (x - 6, y - th - 6), (w - 6, y + 8), (0, 0, 0), -1)
    cv2.putText(out, dom, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)


def _draw_bottom_bar(out: np.ndarray, snap: dict):
    """Countdown bar at the bottom of the frame."""
    h, w   = out.shape[:2]
    BAR_H  = 36
    y0     = h - BAR_H
    phase  = snap["phase"]
    tl     = snap["time_left"]

    cv2.rectangle(out, (0, y0), (w, h), (20, 20, 20), -1)

    if phase == ANALYZING:
        fill      = int(w * (tl / WINDOW_SECONDS))
        any_flag  = any(v["flagged"] for v in snap["queue_data"].values())
        bar_color = (0, 50, 220) if any_flag else (0, 170, 70)
        if fill > 0:
            cv2.rectangle(out, (0, y0 + 3), (fill, h - 3), bar_color, -1)
        secs  = max(1, int(tl) + 1)
        parts = [f"Analyzing: {secs}s"]
        for lbl, v in snap["queue_data"].items():
            consec = v.get("consec", 0)
            if consec > 0:
                parts.append(f"{DISPLAY_LABELS[lbl]}: streak {consec}/{CONSEC_NEG_THRESHOLD}")
    else:
        # result phase — solid blue pulse bar
        cv2.rectangle(out, (0, y0 + 3), (w, h - 3), (100, 60, 20), -1)
        parts = ["-- RESULT DISPLAY --"]

    txt = "   |   ".join(parts)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.putText(out, txt, (max(6, w // 2 - tw // 2), y0 + BAR_H // 2 + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)


def _render(frame, annotations, zones, snap, fps) -> np.ndarray:
    out     = frame.copy()
    phase   = snap["phase"]
    q_data  = snap["queue_data"]

    flagged_queues = [lbl for lbl, v in q_data.items() if v["flagged"]]

    # ── top banner (flag alert) ───────────────────────────────────────────────
    if flagged_queues:
        _draw_banner(out, flagged_queues)

    # ── queue zone boxes ──────────────────────────────────────────────────────
    banner_offset = 54 if flagged_queues else 0
    for lbl, (x1, y1, x2, y2) in zones.items():
        color   = ZONE_COLORS.get(lbl, (255, 255, 255))
        qd      = q_data.get(lbl, {})
        flagged = qd.get("flagged", False)
        neg     = qd.get("neg_count", 0)
        consec  = qd.get("consec", 0)
        thick   = 3 if flagged else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        # show streak: "Queue 1  neg:2  streak:1/3"
        lbl_txt = f"{DISPLAY_LABELS.get(lbl, lbl)}  neg:{neg}  streak:{consec}/{CONSEC_NEG_THRESHOLD}"
        (tw, th), _ = cv2.getTextSize(lbl_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        ty = max(y1 - 5, banner_offset + th + 5)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (0, 0, 0), -1)
        cv2.putText(out, lbl_txt, (x1 + 3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (0, 0, 220) if flagged else color, 2)

    # ── face boxes ────────────────────────────────────────────────────────────
    for (x, y, w, h, emotion, qzone) in annotations:
        is_neg = emotion in NEGATIVE_EMOTIONS
        col    = (0, 0, 220) if is_neg else (0, 210, 0)
        cv2.rectangle(out, (x, y - 35), (x + w, y + h + 5), col, 2)
        tag = f"{emotion}  [{DISPLAY_LABELS.get(qzone, qzone)}]"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
        cv2.rectangle(out, (x, y - 35 - th - 4), (x + tw + 4, y - 35), (0, 0, 0), -1)
        cv2.putText(out, tag, (x + 2, y - 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2)

    # ── dominant emotion word (top-right, only during 2s pause) ─────────────────
    if phase == RESULT:
        _draw_dominant_word(out, snap["last_results"])

    # ── bottom countdown bar ──────────────────────────────────────────────────
    _draw_bottom_bar(out, snap)

    # ── FPS ───────────────────────────────────────────────────────────────────
    cv2.putText(out, f"FPS:{fps:.1f}", (8, banner_offset + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (160, 160, 160), 2)
    return out


# ── Thread 3: Render at 30 FPS ────────────────────────────────────────────────
def capture_render_thread():
    interval = 1.0 / TARGET_FPS
    while True:
        t0    = time.time()
        frame = latest_frame.get()
        if frame is None:
            time.sleep(interval)
            continue
        annotations, zones = latest_det.get()
        snap     = shared.snapshot()
        fps      = shared.get_fps()
        rendered = _render(frame, annotations, zones, snap, fps)
        ok, buf  = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            ring.put(buf.tobytes())
        dt = time.time() - t0
        if dt < interval:
            time.sleep(interval - dt)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    _init_csv()
    threading.Thread(target=webcam_thread,         daemon=True).start()
    threading.Thread(target=detection_worker,      daemon=True).start()
    threading.Thread(target=capture_render_thread, daemon=True).start()
    asyncio.create_task(_broadcast_loop())


async def _broadcast_loop():
    while True:
        await pubsub.publish({"queues": shared.snapshot(), "fps": round(shared.get_fps(), 1)})
        await asyncio.sleep(1.0)


@app.get("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = ring.get()
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/analytics")
def analytics():
    return {"data": shared.snapshot(), "fps": round(shared.get_fps(), 1)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    q = await pubsub.subscribe()
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=5.0)
                await ws.send_json(msg)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(q)
