"""
Feature 6: Intelligent Emergency Alert & Response System
=========================================================
Handles emergency incident management and broadcasts alerts.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
from typing import List

app = FastAPI(title="Feature 6: Emergency Alert & Response")

class EmergencyState:
    def __init__(self):
        self.active = False
        self.zone = None
        self.type = None
        self.instructions = None

state = EmergencyState()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send current state on connection
        await websocket.send_json(self._get_payload())

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self):
        payload = self._get_payload()
        for connection in self.active_connections:
            try:
                await connection.send_json(payload)
            except:
                pass
                
    def _get_payload(self):
        return {
            "emergency_active": state.active,
            "zone": state.zone,
            "type": state.type,
            "instructions": state.instructions
        }

manager = ConnectionManager()

class EmergencyPayload(BaseModel):
    zone: str
    type: str
    instructions: str

@app.post("/api/trigger_emergency")
async def trigger_emergency(payload: EmergencyPayload):
    """Admin triggers an emergency"""
    state.active = True
    state.zone = payload.zone
    state.type = payload.type
    state.instructions = payload.instructions
    
    await manager.broadcast()
    return {"status": "Emergency Activated", "broadcasted": True}

@app.post("/api/resolve_emergency")
async def resolve_emergency():
    """Admin resolves the emergency"""
    state.active = False
    state.zone = None
    state.type = None
    state.instructions = "All clear. Normal operations resumed."
    
    await manager.broadcast()
    return {"status": "Emergency Resolved"}

@app.websocket("/ws/alerts")
async def alert_websocket(websocket: WebSocket):
    """Clients (screens/mobile apps) connect here to listen for alerts"""
    await manager.connect(websocket)
    try:
        while True:
            # keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
