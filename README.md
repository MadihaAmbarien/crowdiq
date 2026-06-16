# CrowdIQ — AI Crowd Intelligence System
CrowdIQ is an AI-powered crowd monitoring system that uses live CCTV feeds to track crowd density, movement, and queue formation in real time. It detects queue lines, estimates waiting times, and monitors congestion levels to keep operations running smoothly. The system also identifies priority queues for elderly and disabled individuals, ensuring faster and more accessible service. When things go wrong, instant alerts notify staff about overcrowding, long wait times, or abnormal crowd behavior. Smart staff monitoring tracks activity on the ground and optimizes counter allocation for better crowd handling. AI-powered analytics further provide insights on peak hours, service demand, and crowd trends to support smarter decision making. Altogether, CrowdIQ transforms passive surveillance into an intelligent, proactive system built to make crowded spaces safer, more efficient, and easier to manage.

## Setup (first time only)

```bash
pip install -r requirements.txt
```

> **Note on face-recognition (Feature 1 only):**
> Windows requires CMake + Visual C++ Build Tools before installing dlib/face-recognition.
> Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
> Then: `pip install cmake dlib face-recognition`

---

## Folder Structure

```text
CrowdIQ/
├── feature1/main.py        Feature 1 — Queue Management & Monitoring
├── feature2/main.py        Feature 2 — Gender-Aware Queue Intelligence
├── feature3/main.py        Feature 3 — Priority Elderly & Disabled Queue
├── feature4/main.py        Feature 4 — Smart QR Assistance & Information
├── feature5/main.py        Feature 5 — AI-Powered Voice Assistance
├── feature6/main.py        Feature 6 — Intelligent Emergency Alert & Response
├── feature7/main.py        Feature 7 — Real-Time Crowd Density & Heatmap Analytics
├── feature8/main.py        Feature 8 — Emotion & Sentiment Analysis
│   └── emotion_log.csv     Auto-generated log of every window result
├── feature9/main.py        Feature 9 — Predictive Crowd Analytics & Congestion Forecasting
├── frontend/               Next.js dashboard (connects to Feature 1 on port 8002)
├── models/
│   ├── queue12.pt                          YOLO queue zone detection
│   ├── best.pt                             YOLO head detection
│   ├── yolov8n.pt                          YOLOv8 nano (general)
│   ├── emotion_model.h5                    Keras emotion classifier (FER-2013, 7 classes)
│   └── haarcascade_frontalface_default.xml OpenCV face detector
├── videos/
│   ├── queue_demo.mp4      Demo video — Features 1 & 2
│   └── priority_demo.mp4   Demo video — Feature 3
├── faces/                  Known face images for Feature 1 recognition
├── requirements.txt
├── start_feature1.bat      Feature 1 launcher
├── start_feature2.bat      Feature 2 launcher
├── start_feature3.bat      Feature 3 launcher
├── start_feature4.bat      Feature 4 launcher
├── start_feature5.bat      Feature 5 launcher
├── start_feature6.bat      Feature 6 launcher
├── start_feature7.bat      Feature 7 launcher
├── start_feature8.bat      Feature 8 launcher (live webcam)
├── start_feature9.bat      Feature 9 launcher
└── start_dashboard.bat     Next.js dashboard launcher
```

---

## Running

### Option A — Double-click launcher (Windows)
| Launcher | Feature | Port |
|---|---|---|
| `start_feature1.bat` | Queue Management | 8002 |
| `start_feature2.bat` | Gender-Aware Queues | 8003 |
| `start_feature3.bat` | Priority Queue | 8004 |
| `start_feature4.bat` | Smart QR Assistance | 8004 (alias) |
| `start_feature5.bat` | Voice Assistance | 8005 |
| `start_feature6.bat` | Emergency Alert | 8006 |
| `start_feature7.bat` | Heatmap Analytics | 8007 |
| `start_feature8.bat` | Emotion Detection | 8008 |
| `start_feature9.bat` | Predictive Analytics | 8009 |
| `start_dashboard.bat`| Next.js Dashboard | 3000 |

### Option B — Terminal
```bash
cd feature1 && python -m uvicorn main:app --host 0.0.0.0 --port 8002
cd feature2 && python -m uvicorn main:app --host 0.0.0.0 --port 8003
cd feature3 && python -m uvicorn main:app --host 0.0.0.0 --port 8004
cd feature4 && python -m uvicorn main:app --host 0.0.0.0 --port 8004
cd feature5 && python -m uvicorn main:app --host 0.0.0.0 --port 8005
cd feature6 && python -m uvicorn main:app --host 0.0.0.0 --port 8006
cd feature7 && python -m uvicorn main:app --host 0.0.0.0 --port 8007
cd feature8 && python -m uvicorn main:app --host 0.0.0.0 --port 8008
cd feature9 && python -m uvicorn main:app --host 0.0.0.0 --port 8009
```

---

## Architecture (all features)

- **Backend:** FastAPI + Uvicorn
- **ML:** YOLOv8 (ultralytics) + Keras/TensorFlow + OpenCV
- **Streaming:** MJPEG (`/video_feed`) + WebSocket (`/ws`) + REST (`/analytics`)
- **Paths:** `__file__`-based — runs from any directory on any machine
- **Smoothness:** Decoupled render thread never blocks on ML inference
