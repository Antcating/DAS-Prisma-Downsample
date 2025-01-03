import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Paths
RAW_DATA_PATH = config.get("PATHS", "RAW_DATA_PATH")
DOWN_DATA_PATH = config.get("PATHS", "DOWN_DATA_PATH")
TMP_PATH = config.get("PATHS", "TMP_PATH")

# DAS system parameters
PACKET_SIZE = int(config.get("SYSTEM", "PACKET_SIZE"))
NUM_CHANNELS = int(config.get("SYSTEM", "RAW_NUM_CHANNELS"))
RAW_SPS = float(config.get("SYSTEM", "RAW_SPS"))
RAW_DX = float(config.get("SYSTEM", "RAW_DX"))

# Downsample parameters
CHUNK_SIZE = int(config.get("DOWNSAMPLING", "CHUNK_SIZE"))
CHUNK_OVERLAP = int(config.get("DOWNSAMPLING", "CHUNK_OVERLAP"))
SPS = int(config.get("DOWNSAMPLING", "SPS"))
DX = float(config.get("DOWNSAMPLING", "DX"))
NUM_THREADS = int(config.get("DOWNSAMPLING", "NUM_THREADS"))
FACTORS_TIME = [int(x) for x in config.get("DOWNSAMPLING", "FACTORS_TIME").split(",")]
FACTORS_SPACE = [int(x) for x in config.get("DOWNSAMPLING", "FACTORS_SPACE").split(",")]

# Logging parameters
LOG_LEVEL = config.get("LOGGING", "LOG_LEVEL")
CONSOLE_LOG = config.get("LOGGING", "CONSOLE_LOG")
CONSOLE_LOG_LEVEL = config.get("LOGGING", "CONSOLE_LOG_LEVEL")
