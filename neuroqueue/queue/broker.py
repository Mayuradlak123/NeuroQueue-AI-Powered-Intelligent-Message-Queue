import os
import json
import time
import uuid
from neuroqueue.config.logger import logger

class LocalBroker:
    """
    A local file-based message broker to replace Redis.
    Uses a simple JSONL file to simulate a stream.
    """
    def __init__(self, data_dir="neuroqueue_data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.queue_file = os.path.join(self.data_dir, "queue.jsonl")
        logger.info(f"Initialized LocalBroker backed by {self.queue_file}")

    def publish(self, stream_key: str, message: dict):
        """Simulates publishing to a stream by appending to a file."""
        try:
            msg_id = f"{int(time.time()*1000)}-0" # Mimic Redis ID format
            entry = {
                "stream": stream_key,
                "id": msg_id,
                "data": message,
                "timestamp": time.time()
            }
            
            with open(self.queue_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            
            logger.info(f"Published message {msg_id} to local stream {stream_key}")
            return msg_id
        except Exception as e:
            logger.error(f"Failed to publish to {stream_key}: {e}")
            return None

    def consume(self, stream_key: str, group_name: str, consumer_name: str, count=1):
        """
        Simulates consuming.
        For simplicity in this local version, we just read the last N messages
        that match the stream_key.
        NOTE: This is a simplified usage for demo purposes.
        """
        try:
            messages = []
            if not os.path.exists(self.queue_file):
                return []
                
            with open(self.queue_file, "r") as f:
                lines = f.readlines()
                
            # Filter for stream
            stream_msgs = []
            for line in lines:
                try:
                    entry = json.loads(line)
                    if entry.get("stream") == stream_key:
                        stream_msgs.append(entry)
                except:
                    continue
            
            # Return last 'count' messages mimicking Redis XREADGROUP structure
            # Redis returns: [[stream_name, [(msg_id, data)]]]
            # We return similar structure for compatibility if needed.
            
            recent = stream_msgs[-count:] if stream_msgs else []
            formatted_msgs = [(m['id'], m['data']) for m in recent]
            
            if not formatted_msgs:
                return []
                
            return [[stream_key, formatted_msgs]]
            
        except Exception as e:
            logger.error(f"Failed to consume from {stream_key}: {e}")
            return []

    def ack(self, stream_key: str, group_name: str, msg_id: str):
        """No-op for local simple file broker."""
        pass

# Global broker instance
broker = LocalBroker()
