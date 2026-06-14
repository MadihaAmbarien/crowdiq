"""
Feature 4: Smart QR Assistance & Information System
====================================================
Serves a mobile-friendly HTML page that fetches real-time 
queue data (wait time, counts, congestion) from Feature 1 (port 8002).
"""
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

app = FastAPI(title="Feature 4: Smart QR Assistance")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>CrowdIQ - Smart QR Assistance</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, sans-serif; background: #f4f4f9; margin: 0; padding: 20px; color: #333; }
        .card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h1 { font-size: 24px; text-align: center; color: #222; }
        .queue-stat { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }
        .queue-stat:last-child { border-bottom: none; }
        .alert { background: #fee2e2; color: #991b1b; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; text-align: center; }
        .success { color: #166534; font-weight: bold; }
        .danger { color: #991b1b; font-weight: bold; }
    </style>
    <script>
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                
                document.getElementById('content').innerHTML = `
                    ${data.crowd_alert ? '<div class="alert">⚠️ High Congestion Detected in Area</div>' : ''}
                    <div class="card">
                        <h3>Live Queue Status</h3>
                        ${Object.entries(data.counts).map(([q, count]) => `
                            <div class="queue-stat">
                                <span>${q}</span>
                                <span>
                                    ${count} people 
                                    (<span class="${data.wait_times[q] > 300 ? 'danger' : 'success'}">
                                        ~${Math.round(data.avg_wait_times[q]/60)} mins
                                    </span>)
                                </span>
                            </div>
                        `).join('')}
                    </div>
                `;
            } catch (err) {
                document.getElementById('content').innerHTML = '<p style="text-align:center;">Failed to load live data.</p>';
            }
        }
        setInterval(fetchData, 3000);
        window.onload = fetchData;
    </script>
</head>
<body>
    <h1>📱 Live Queue Assistance</h1>
    <div id="content">Loading live data...</div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE

@app.get("/api/data")
async def get_data():
    """Proxy live data from Feature 1"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8002/analytics", timeout=2.0)
            return response.json()
    except Exception as e:
        return {
            "error": "Cannot connect to Feature 1", 
            "counts": {"Queue1": 0, "Queue2": 0},
            "avg_wait_times": {"Queue1": 0, "Queue2": 0},
            "wait_times": {"Queue1": 0, "Queue2": 0},
            "crowd_alert": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
