import os
from datetime import datetime
from pymongo import MongoClient
from neuroqueue.config.logger import logger
from neuroqueue.config.settings import settings

class MongoCustomClient:
    def __init__(self):
        try:
            self.client = MongoClient(settings.MONGO_URI)
            self.db = self.client[settings.MONGO_DB]
            logger.info(f"Connected to MongoDB at {settings.MONGO_URI}, DB: {settings.MONGO_DB}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise e

    def log_audit(self, message_id: str, data: dict):
        """Logs an audit record for a message with timestamp."""
        try:
            self.db.audit_logs.insert_one({
                "message_id": message_id,
                "created_at": datetime.utcnow().isoformat(),
                **data
            })
            logger.info(f"Audit log saved for message {message_id}")
        except Exception as e:
            logger.error(f"Failed to log audit for {message_id}: {e}")

# Global instance
mongo_client = MongoCustomClient()
