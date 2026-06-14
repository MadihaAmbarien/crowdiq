"""
Feature 5: AI-Powered Voice Assistance System
==============================================
Provides contextual voice-based public assistance.
Uses a text API that can be easily connected to pyttsx3 or Web Speech API
on the frontend for actual voice synthesis.
"""
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import re

app = FastAPI(title="Feature 5: AI Voice Assistance")

class VoiceQuery(BaseModel):
    text: str

@app.post("/api/voice_query")
async def process_voice_query(query: VoiceQuery):
    """
    Process a natural language query and return a text response 
    that can be read aloud by a TTS engine.
    """
    text = query.text.lower()
    
    # Fetch live queue data
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8002/analytics", timeout=2.0)
            data = response.json()
    except Exception as e:
        return {"response": "I am currently unable to connect to the queue sensors. Please try again later."}

    # Intent matching
    if "shortest" in text or "fastest" in text:
        wait_times = data.get("avg_wait_times", {})
        if not wait_times:
            return {"response": "I cannot determine the wait times right now."}
        
        # Filter out queues with 0 wait time if they are closed, or keep them.
        shortest_queue = min(wait_times, key=wait_times.get)
        mins = wait_times[shortest_queue] // 60
        return {"response": f"The shortest wait is at {shortest_queue} with approximately {mins} minutes."}
        
    elif "how many" in text or "count" in text:
        match = re.search(r'queue (\d+)', text)
        q_name = f"Queue{match.group(1)}" if match else "Queue1"
        count = data.get("counts", {}).get(q_name, 0)
        return {"response": f"There are currently {count} people in {q_name}."}
        
    elif "emergency" in text or "help" in text:
        return {"response": "If this is an emergency, please contact security immediately. I am switching to alert mode."}
        
    else:
        return {"response": "I can help you find the shortest queue, check wait times, or assist with navigation. How can I help?"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
