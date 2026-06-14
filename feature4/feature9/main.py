"""
Feature 9: Predictive Crowd Analytics & Congestion Forecasting System
======================================================================
Uses historical queue and crowd data to predict future congestion, 
estimate peak hours, and generate preventive operational recommendations.
"""
from fastapi import FastAPI
import datetime
import random

app = FastAPI(title="Feature 9: Predictive Analytics")

@app.get("/api/forecast")
async def get_forecast():
    """
    Returns a mock 2-hour forecast based on the current time.
    In a full implementation, this would use a time-series model (like Prophet/LSTM)
    trained on historical 'emotion_log.csv' or queue wait times.
    """
    now = datetime.datetime.now()
    forecast = []
    
    # Generate 4 periods (30 min increments)
    for i in range(4):
        future_time = now + datetime.timedelta(minutes=30 * (i+1))
        
        # Simulate an upcoming evening peak
        is_evening = 17 <= future_time.hour <= 20
        base_load = random.randint(40, 70) if is_evening else random.randint(10, 30)
        
        forecast.append({
            "timestamp": future_time.strftime("%H:%M"),
            "predicted_load": base_load,
            "status": "High Congestion" if base_load > 50 else "Normal",
            "recommendation": "Open 2 additional counters" if base_load > 50 else "Maintain current staff"
        })
        
    return {
        "current_time": now.strftime("%H:%M"),
        "peak_hour_prediction": "18:00 - 20:00",
        "forecast": forecast
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
