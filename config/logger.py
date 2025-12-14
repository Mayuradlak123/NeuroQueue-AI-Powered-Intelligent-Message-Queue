import logging
from logging.config import dictConfig
import os
from pydantic import BaseModel
os.makedirs("logs", exist_ok=True)

log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        # "console": {
        #     "format": "%(levelname)s: %(asctime)s - [%(pathname)s:%(lineno)d] - %(message)s",
        #     "datefmt": "%Y-%m-%d %H:%M:%S",
        # },
        "file": { 
            "format": "%(levelname)s: %(asctime)s - [%(pathname)s:%(lineno)d] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        # "console": {
        #     "formatter": "console",
        #     "class": "logging.StreamHandler",
        #     "stream": "ext://sys.stderr",
        # },
        "file": {
            "formatter": "file",
            "class": "logging.FileHandler",
            "filename": "logs/neuroqueue.log",
            "mode": "a",
            "encoding": "utf-8"
        },
    },
    "loggers": {
        "neuroqueue": {
            # "handlers": ["console", "file"],
            "handlers": ["file"],
            "level": "INFO",
        },
    },
}

dictConfig(log_config)
logger = logging.getLogger("neuroqueue")