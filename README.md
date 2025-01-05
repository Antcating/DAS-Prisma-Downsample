# DAS-Prisma-Downsample

## Overview

DAS-Prisma-Downsample is a Python-based project designed to downsample raw DAS (Distributed Acoustic Sensing) data. The project reads raw SEG-Y files, processes them, and writes the downsampled data to HDF5 files. The downsampling process is optimized for real-time performance using multithreading.

## Installation

To install the necessary dependencies for this project, follow these steps:

1. Clone the repository:

```shell
git clone https://github.com/Antcating/DAS-Prisma-Downsample.git
cd DAS-Prisma-Downsample
```

2. Create a virtual environment:

```shell
python3 -m venv .venv
```

3. Activate the virtual environment:

On macOS and Linux:

```shell
source .venv/bin/activate
```

On Windows:

```shell
.venv\Scripts\activate
```

4. Install the dependencies:

```shell
pip install -r requirements.txt
```

5. Configure the downsampler

Make sure to configure the config.ini file with the appropriate paths and settings before running the downsampling process.

## Usage

### Single run

To run the downsampling process, execute the entry point of the program:

```shell
python3 src/main.py
```

### Scheduling

For scheduling on Windows, there is a `downsample.bat` file in the root of the project. To work properly, you have to enter the absolute path to the program directory on line 2 of the `downsample.bat` file. 

## Configuration

To configure the project for a particular setup, the project uses a configuration file `config.ini` to set various parameters:

```
[PATHS]
; Absolute path to the raw data
RAW_DATA_PATH = absolute_path_to_raw_data
; Absolute path to the processed data
DOWN_DATA_PATH = absolute_path_to_downsampled_data
; Absolute path to the temporary data (e.g. for status files)
TMP_PATH = absolute_path_to_temporary_data

; DAS configuration (set by the provider of the system)
[SYSTEM]
; Packet size in seconds
PACKET_SIZE = 10
; Sampling rate in Hz before downsampling
RAW_SPS = 150
; Number of channels in the raw data before downsampling
RAW_NUM_CHANNELS = 1000
; Spatial sampling rate in 1/m before downsampling
RAW_DX = 10

; Configuration for the downsampling
[DOWNSAMPLING]
; Sampling rate in Hz after downsampling
SPS = 10
; Number of channels in the raw data after downsampling
DX = 10
; Size of the downsampling window in seconds
CHUNK_SIZE = 60
; Overlap of the downsampling window in seconds
CHUNK_OVERLAP = 1
; Factors for the downsampling in time and space
FACTORS_TIME = 3,5
; TO BE IMPLEMENTED!!!
FACTORS_SPACE = 1
; Number of threads for the downsampling
NUM_THREADS = 4

; Configuration for logging
[LOGGING]
; Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = INFO
; Console logging (True/False)
CONSOLE_LOG = True
; Console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
CONSOLE_LOG_LEVEL = INFO
```
