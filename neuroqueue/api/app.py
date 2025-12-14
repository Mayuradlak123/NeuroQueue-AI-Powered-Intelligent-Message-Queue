from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import time
from datetime import datetime

from neuroqueue.graph.decision_graph import decision_app
from neuroqueue.config.logger import logger
from neuroqueue.storage.mongo import mongo_client

app = FastAPI(title="NeuroQueue", version="1.0.0")

# Setup Templates
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

from typing import List, Dict

# ...

class ConnectionManager:
    def __init__(self):
        # topic -> list of websockets
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, topic: str = "general"):
        await websocket.accept()
        if topic not in self.active_connections:
            self.active_connections[topic] = []
        self.active_connections[topic].append(websocket)
        logger.info(f"Client connected to topic: {topic}")

    def disconnect(self, websocket: WebSocket, topic: str = "general"):
        if topic in self.active_connections:
            if websocket in self.active_connections[topic]:
                self.active_connections[topic].remove(websocket)

    async def broadcast(self, message: dict, topic: str):
        if topic in self.active_connections:
            for connection in self.active_connections[topic]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")

manager = ConnectionManager()

class MessageInput(BaseModel):
    content: str
    topic: str = "general"
    metadata: dict = {}

async def _core_process_message(content: str, topic: str = "general", metadata: dict = {}):
    """Core logic decoupled from API transport."""
    msg_id = str(uuid.uuid4())
    logger.info(f"Processing message {msg_id} [Topic: {topic}]: {content}")
    
    initial_state = {
        "message_id": msg_id,
        "content": content,
        "metadata": metadata,
        "embedding": None,
        "llm_reasoning": "",
        "intent": "",
        "priority": 0.0,
        "failure_risk": 0.0,
        "routing_decision": "",
        "explanation": ""
    }
    
    result = await decision_app.ainvoke(initial_state)
    
    # Save structured log for UI consumption
    mongo_client.db.processed_history.insert_one({
        "message_id": result['message_id'],
        "content": result['content'],
        "topic": topic,
        "intent": result.get('intent', 'unknown'),
        "priority": result.get('priority', 0.0),
        "risk": result.get('failure_risk', 0.0),
        "decision": result.get('routing_decision', 'UNKNOWN'),
        "timestamp": datetime.now().isoformat()
    })
    
    response_payload = {
        "message_id": msg_id,
        "content": content,
        "topic": topic,
        "routing": {
            "decision": result.get('routing_decision', 'UNKNOWN'),
            "explanation": result.get('explanation', '')
        },
        "analysis": {
            "intent": result.get('intent', 'unknown'),
            "priority": result.get('priority', 0.0),
            "risk": result.get('failure_risk', 0.0)
        }
    }
    
    # Broadcast to subscribers of this topic
    await manager.broadcast(response_payload, topic)
    
    return response_payload

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_message(msg: MessageInput):
    try:
        return await _core_process_message(msg.content, msg.topic, msg.metadata)
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Initial connection handling
    # We expect the client to send a "subscribe" or "publish" message after connecting
    # But for simplicity, we treat raw connection as "general" subscriber unless specified?
    # Better: Wait for first message to decide role.
    
    await websocket.accept()
    current_topic = "general" # Default
    
    # Register connection
    if current_topic not in manager.active_connections:
        manager.active_connections[current_topic] = []
    manager.active_connections[current_topic].append(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            action = data.get("action", "publish")
            topic = data.get("topic", "general")
            
            if action == "subscribe":
                # Switch topic subscription
                manager.disconnect(websocket, current_topic)
                current_topic = topic
                if current_topic not in manager.active_connections:
                    manager.active_connections[current_topic] = []
                manager.active_connections[current_topic].append(websocket)
                await websocket.send_json({"status": "subscribed", "topic": topic})
                
            elif action == "publish":
                content = data.get("content", "")
                # Process and it will auto-broadcast to topic subscribers
                result = await _core_process_message(content, topic)
                # Do not send directly; reliance on broadcast prevents duplicates
                # if the user is subscribed to the topic (which UI enforces).
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, current_topic)
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.get("/history")
def get_history():
    """Fetch last 20 processed messages for the consumer feed."""
    try:
        cursor = mongo_client.db.processed_history.find().sort("timestamp", -1).limit(20)
        messages = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"]) # ObjectId not serializable
            messages.append(doc)
        return messages
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        return []

@app.get("/search")
def search_messages(q: str):
    """Semantic search using ChromaDB."""
    from neuroqueue.storage.chroma import chroma_client
    try:
        results = chroma_client.retrieve_similar_messages(q)
        return {"query": q, "results": results}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}
