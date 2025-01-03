import logging
from logging.handlers import RotatingFileHandler
import os
from config import DOWN_DATA_PATH, LOG_LEVEL, CONSOLE_LOG, CONSOLE_LOG_LEVEL

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d | %(levelname)s | %(funcName)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# Create file handler
file_handler = RotatingFileHandler(
    os.path.join(DOWN_DATA_PATH, "log"), maxBytes=100000000, backupCount=10
)
file_handler.setLevel(LOG_LEVEL)

# Set formats and add the handlers to the logger
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create stream handler if CONSOLE_LOG is True
if CONSOLE_LOG == "True":
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(CONSOLE_LOG_LEVEL)

    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)