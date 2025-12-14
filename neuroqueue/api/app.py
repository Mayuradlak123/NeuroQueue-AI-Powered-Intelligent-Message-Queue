from fastapi import FastAPI, HTTPException, Request
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

class MessageInput(BaseModel):
    content: str
    metadata: dict = {}

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_message(msg: MessageInput):
    msg_id = str(uuid.uuid4())
    logger.info(f"Received message {msg_id}: {msg.content}")
    
    initial_state = {
        "message_id": msg_id,
        "content": msg.content,
        "metadata": msg.metadata,
        "embedding": None,
        "intent": "",
        "priority": 0.0,
        "failure_risk": 0.0,
        "routing_decision": "",
        "explanation": ""
    }
    
    try:
        result = decision_app.invoke(initial_state)
        
        # Save structured log for UI consumption
        # Using mongo client directly to ensure it appears in specific collection for history
        mongo_client.db.processed_history.insert_one({
            "message_id": result['message_id'],
            "content": result['content'],
            "intent": result['intent'],
            "priority": result['priority'],
            "risk": result['failure_risk'],
            "decision": result['routing_decision'],
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "message_id": msg_id,
            "routing": {
                "decision": result['routing_decision'],
                "explanation": result['explanation']
            },
            "analysis": {
                "intent": result['intent'],
                "priority": result['priority'],
                "risk": result['failure_risk']
            }
        }
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
