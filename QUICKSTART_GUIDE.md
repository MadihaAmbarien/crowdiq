# 🚀 CrowdIQ: Quickstart & Setup Guide

Welcome to the **CrowdIQ** project! This is a 9-feature AI-powered crowd intelligence ecosystem. Follow this guide to get the project running on your local machine and test the API endpoints, frontend dashboard, and AI chatbot.

---

## 🛠️ Step 1: Initial Setup

Before running any features, you need to install the required Python and Node.js dependencies.

### 🐍 Python Backend Setup
1. **Open a terminal** inside the `CrowdIQ` root folder.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   > **Windows Users (Important):** Feature 1 uses Face Recognition, which requires C++ build tools. If `pip install` fails on `dlib` or `face-recognition`, download and install the [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first, then try again.

### ⚛️ Next.js Frontend Setup
1. **Open a terminal** inside the `CrowdIQ/frontend` folder.
2. Install the frontend node packages:
   ```bash
   npm install
   ```

---

## 🏃 Step 2: Running the Features

Every feature runs its own dedicated backend server (FastAPI). You can start them individually by simply double-clicking the `.bat` launcher files in the project folder!

| Feature | Launcher Script | Port | What it does |
|---------|-----------------|------|--------------|
| **1. Queue Management** | `start_feature1.bat` | **8002** | YOLO AI counting people & wait times. |
| **2. Gender-Aware Queue** | `start_feature2.bat` | **8003** | Separates queues by gender. |
| **3. Priority Assistance** | `start_feature3.bat` | **8004** | Fast-track elderly/disabled queues. |
| **4. Smart QR** | `start_feature4.bat` | **8004** | Mobile dashboard for users scanning a QR code. |
| **5. AI Voice Assist** | `start_feature5.bat` | **8005** | API that answers queue questions. |
| **6. Emergency Response** | `start_feature6.bat` | **8006** | Broadcasts evacuation alerts. |
| **7. Heatmap Analytics** | `start_feature7.bat` | **8007** | Video feed showing density heatmaps. |
| **8. Emotion Detection** | `start_feature8.bat` | **8008** | Uses webcam to detect crowd frustration. |
| **9. Predictive Analytics** | `start_feature9.bat` | **8009** | Forecasts future peak hour congestion. |
| **🖥️ Web Dashboard & Hub** | `start_dashboard.bat` | **3000** | Interactive landing hub and operation monitors! |

> **Pro Tip:** Always run `start_feature1.bat` first! It acts as the core AI engine that supplies real-time crowd data to other features (like the Smart QR View and Voice Assistant).

---

## 🌐 Step 3: Landing Hub & AI Chatbot

Once the frontend dashboard is launched (`start_dashboard.bat`):

1. **Access the Landing Page:** Open your browser to [http://localhost:3000/](http://localhost:3000/)
   - Explore descriptions, use-cases, and ports for all 9 systems.
   - Interact with live simulators for QR codes, voice commands, emergency triggers, heatmaps, and load forecasts!
   - Click **Launch Live Monitor** at the top right to go to the live operational center (`/monitor`).

2. **Setup the Groq Cloud AI Chatbot:**
   The landing hub features a persistent AI chatbot in the bottom right corner.
   - **Offline intelligent mode (Default):** Works out of the box! It uses a high-fidelity local heuristic model to answer questions about CrowdIQ setup, commands, wait times, and feature listings.
   - **Groq Cloud AI mode (Llama 3):** To activate advanced AI cognitive reasoning, set your Groq API key in your terminal before starting the dashboard:
     ```bash
     # Windows (cmd)
     set GROQ_API_KEY=your-groq-key-here
     
     # Windows (PowerShell)
     $env:GROQ_API_KEY="your-groq-key-here"
     
     # Start the dashboard with Groq enabled:
     npm run dev
     ```

---

## 📲 Step 4: Live QR Code Scanning
To test the QR assistance view on your phone:
1. Locate the **Smart QR Code** displayed on the Landing Page.
2. In a real environment, scanning this QR takes visitors to a lightweight mobile page on port `8004` (Feature 4).
3. Try clicking **Launch Phone Simulator** next to the QR code on the landing page to instantly see an interactive mobile viewport mock rendering live queue loads in real time!

---

## 📊 Testing backend APIs
Here are a few quick ways to test the backend APIs directly:

**1. Live Queue Data (Feature 1)**
Open your browser to: [http://localhost:8002/analytics](http://localhost:8002/analytics)

**2. Ask the Voice Assistant (Feature 5)**
With `start_feature5.bat` running, run this `curl` command to ask it a question:
```bash
curl -X POST "http://localhost:8005/api/voice_query" -H "Content-Type: application/json" -d "{\"text\": \"Which queue has the shortest wait?\"}"
```

**3. Trigger an Emergency (Feature 6)**
Trigger a fire drill alarm to broadcast a WebSocket emergency alert:
```bash
curl -X POST "http://localhost:8006/api/trigger_emergency" -H "Content-Type: application/json" -d "{\"zone\": \"Zone B\", \"type\": \"Fire\", \"instructions\": \"Evacuate immediately!\"}"
```

---

🎉 **Have fun exploring CrowdIQ!** 
Let us know if you run into any issues setting up the environment.
