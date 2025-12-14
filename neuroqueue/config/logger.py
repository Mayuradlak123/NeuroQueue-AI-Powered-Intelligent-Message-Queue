import logging
from logging.config import dictConfig
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

class RelativePathFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.project_root = os.getcwd()
        self.path_cache = {}

    def filter(self, record):
        if hasattr(record, 'pathname'):
            if record.pathname not in self.path_cache:
                self.path_cache[record.pathname] = os.path.relpath(record.pathname, self.project_root)
            record.relpath = self.path_cache[record.pathname]
        else:
            record.relpath = "unknown"
        return True

log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "relpath": {
            "()": RelativePathFilter,
        }
    },
    "formatters": {
        "console": {
            "format": "%(levelname)s: %(asctime)s - [%(relpath)s:%(lineno)d] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "file": { 
            "format": "%(levelname)s: %(asctime)s - [%(relpath)s:%(lineno)d] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "formatter": "console",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "filters": ["relpath"]
        },
        "file": {
            "formatter": "file",
            "class": "logging.FileHandler",
            "filename": "logs/neuroqueue.log",
            "mode": "a",
            "encoding": "utf-8",
            "filters": ["relpath"]
        },
    },
    "loggers": {
        "neuroqueue": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    },
}

dictConfig(log_config)
logger = logging.getLogger("neuroqueue")
