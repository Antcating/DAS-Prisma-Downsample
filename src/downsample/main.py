import os
from datetime import datetime

import h5py
import pytz
import numpy as np
from utils.multithread import decimate, multithreaded_mean
from log.main_logger import logger as log
from config import (
    RAW_DATA_PATH,
    DOWN_DATA_PATH,
    TMP_PATH,
    PACKET_SIZE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SPS,
    DX,
    RAW_SPS,
    RAW_DX,
    FACTORS_TIME,
    FACTORS_SPACE,
    NUM_THREADS,
)

class Downsampler:
    def __init__(
            self, 
            raw_data_path=RAW_DATA_PATH,
            down_data_path=DOWN_DATA_PATH,
            tmp_path=TMP_PATH,
            packet_size=PACKET_SIZE,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            sps=SPS,
            dx=DX,
            raw_sps=RAW_SPS,
            raw_dx=RAW_DX,
            factors_time=FACTORS_TIME,
            factors_space=FACTORS_SPACE,
            num_threads=NUM_THREADS,
    ):
        self.raw_data_path = raw_data_path
        self.down_data_path = down_data_path
        self.tmp_path = tmp_path
        self.packet_size = packet_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sps = sps
        self.dx = dx
        self.raw_sps = raw_sps
        self.raw_dx = raw_dx
        self.factors_time = factors_time
        self.factors_space = factors_space
        
        self.num_threads = num_threads

        self.num_channels = None
        self.raw_data_array: np.ndarray = None
        self.last_processed_file, self.last_file_offset = None, None

        self.time_delay_threshold = 0.05
        self.version = "1.0.0"
        self.last_updated = "2024-03-01"

    def _get_file_timestamp(self, file):
        """
        Get the timestamp of a file based on its name.

        Parameters:
        -----------
        file : str
            The file name.

        Returns:
        --------
        float
            The timestamp of the file.
        """
        file_datetime = datetime.strptime(file.split(".")[0], "%Y-%m-%dT%H-%M-%S-%f")
        file_datetime = pytz.timezone("Asia/Jerusalem").localize(file_datetime)
        file_datetime_utc = file_datetime.astimezone(pytz.UTC)
        file_timestamp = file_datetime_utc.timestamp()
        return file_timestamp
    
    def _preallocate_raw_data_array(self):
        """
        Preallocate the raw data array.

        Returns:
        --------
        numpy.ndarray
            The preallocated raw data array.
        """
        self.raw_data_array = np.zeros((self.num_channels, int(self.raw_sps * self.chunk_size + 2 * self.raw_sps * self.chunk_overlap)), dtype=np.float32)
    
    def _read_last_file_status(self):
        """
        Get the last processed file data.

        Returns:
        --------
        tuple
            The last processed file and its offset.
        """
        try:
            with open(os.path.join(self.tmp_path, "status"), "r") as f:
                last_processed_file, last_chunk_offset = f.read().split(",")
                return last_processed_file, int(last_chunk_offset)
        except FileNotFoundError:
            return "", 0

    def _list_raw_files(self):
        """
        Fetch the list of raw files to be processed.

        Returns:
        --------
        list
            A list of tuples containing directory path and file name.
        """
        raw_files_list = []

        if self.last_processed_file == "":
            last_data_time = 0
        else:
            last_data_time = self._get_file_timestamp(self.last_processed_file)
        today = datetime.now(tz=pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        dirs = [
            dir for dir in os.listdir(self.raw_data_path)
            if os.path.isdir(os.path.join(self.raw_data_path, dir))
            and os.path.getmtime(os.path.join(self.raw_data_path, dir)) < today.timestamp()
        ]

        for dir_path in sorted(dirs, key=lambda x: os.path.getmtime(os.path.join(self.raw_data_path, x))):
            for _, dirs, files in os.walk(os.path.join(self.raw_data_path, dir_path)):
                # Filter out non-SEG-Y files (e.g. metadata files)
                files = [file for file in files if file.endswith(".segy")]
                for file in sorted(files, key=lambda x: self._get_file_timestamp(x)):
                    file_timestamp = self._get_file_timestamp(file)
                    # Only process files that are newer than the last processed file
                    # But also include the last processed file because it might have been cut in the middle
                    if file.endswith(".segy") and file_timestamp >= last_data_time:
                        raw_files_list.append([dir_path, file])
        # Reverse the list to process the oldest files first (FIFO)
        raw_files_list = raw_files_list[::-1]
        return raw_files_list
    
    def _load_segy_data(self, file_path):
        """
        Load SEG-Y data from a file.

        Parameters:
        -----------
        file_path : str
            The path to the SEG-Y file.

        Returns:
        --------
        numpy.ndarray
            The loaded data.
        """
        with open(os.path.join(self.raw_data_path, file_path), "rb") as f:
            f.seek(3714)
            traces = np.frombuffer(f.read(2), dtype=np.int16)[0]
        mmap_dtype = np.dtype([("headers", np.void, 240), ("data", "f4", traces)])
        segy_data = np.memmap(os.path.join(self.raw_data_path, file_path), dtype=mmap_dtype, mode="r", offset=3600)
        data = segy_data["data"]
        return data

    def _downsample_raw_data(self):
        """
        Downsample the raw data array.

        Returns:
        --------
        numpy.ndarray
            The downsampled data.
        """
        down_data = self._downsample_array(self.raw_data_array, factors=self.factors_time, type="scipy")
        down_data = down_data[:, self.chunk_overlap * self.sps : -self.chunk_overlap * self.sps]
        return down_data

    def _validate_downsampled_data(self, down_data):
        """
        Validate the downsampled data.

        Parameters:
        -----------
        down_data : numpy.ndarray
            The downsampled data to validate.

        Raises:
        -------
        ValueError
            If the downsampled data is invalid.
        """
        if down_data.shape[1] != self.chunk_size * self.sps:
            raise ValueError(f"Downsampled data shape is {down_data.shape}, expected {(self.num_channels, self.chunk_size * self.sps)}")
        if np.isnan(down_data).any():
            raise ValueError("Downsampled data contains NaN values")
        if np.isinf(down_data).any():
            raise ValueError("Downsampled data contains infinite values")

    def _downsample_array(self, arr, factor=-1, factors=[], type="scipy") -> np.ndarray:
        """
        Downsample an array using the specified method.

        Parameters:
        -----------
        arr : numpy.ndarray
            The input array to downsample.
        factor : int, optional
            The downsampling factor (default is -1).
        factors : list, optional
            A list of downsampling factors (default is []).
        type : str, optional
            The downsampling method (default is "scipy").

        Returns:
        --------
        numpy.ndarray
            The downsampled array.
        """
        if type == "scipy":
            if factor != -1:
                return decimate(arr, factor, axis=-1)
            else:
                # We have to downsample raw data by factor 15 in time.
                # On the old system we used mean downsampling by factor 15 directly, 
                # but it was creating aliasing artifacts at high frequencies.
                # To avoid this, we switched to scipy.signal.decimate function. 
                # But as it turned out, it is too slow to run in real-time, 
                # so I have rewritten it to use multithreading and also we decided to do downsampling in two steps:
                # 1. Decimate downsampling by factor 3 in time.
                # 2. Decimate downsampling by factor 5 in time.
                # This way we avoid aliasing and keep the system real-time.
                arr = decimate(arr, factors[0], axis=1, num_thread=self.num_threads)
                return decimate(arr, factors[1], axis=1, num_thread=self.num_threads)
        elif type == "mean":
            if factors != []:
                raise NotImplementedError("Mean downsampling does not support multiple factors")
            arr = arr.reshape(arr.shape[0], -1, factor)
            return multithreaded_mean(arr, factor)
        
    def _determine_chunk_bounds(self) -> tuple:
        # Check if the last processed file is not set (first run ever)
        # or the data is missing (last file offset is set to -1)
        if self.last_processed_file == "" or self.last_file_offset == -1:
            # Special case when there is insufficient information 
            # to determine the start and end times of the chunk
            return 0, float("inf")
        last_data_time = self._get_file_timestamp(self.last_processed_file) + self.last_file_offset / self.raw_sps

        # Borrow 2 CHUNK_OVERLAP seconds from the *previous* chunk with reference to the offset pointer
        # Because the filter is applied in frequency domain:
        # it is important to cut-off the edges.
        # Data will be taken from the middle of the augmented chunk:
        # |---CHUNK_OVERLAP---|---CHUNK_SIZE---|---CHUNK_OVERLAP---|

        # Explanation why we need to subtract 2 * CHUNK_OVERLAP from the last data time:
        # Offset pointer shows the position in the array up to which the data was taken from the last file
        # Because of the structure of the array, the offset pointer points to the position after overlap, 
        # so we need to subtract it (and the overlap) from the last data time for current chunk.
        start_time_overlap = last_data_time - 2 * self.chunk_overlap
        end_time_overlap = last_data_time + self.chunk_size
        return start_time_overlap, end_time_overlap

    def _determine_first_chunk_bounds(self, data_start_ts: float) -> tuple:
        start_time_overlap = data_start_ts
        
        # Borrow 2 CHUNK_OVERLAP seconds from the *next* chunk
        
        # Because the filter is applied in frequency domain:
        # it is important to cut-off the edges.
        # Data will be taken from the middle of the augmented chunk:
        # |---CHUNK_OVERLAP---|---CHUNK_SIZE---|---CHUNK_OVERLAP---|
        end_time_overlap = data_start_ts + self.chunk_size + 2 * self.chunk_overlap
        return start_time_overlap, end_time_overlap

    def _load_chuck_raw_data(self, raw_files_list):
        """
        Load raw data from a list of raw files and process it into a contiguous array.

        Parameters:
        -----------
        raw_files_list : list of tuples
            A list of tuples where each tuple contains the directory path and filename of a raw file.

        Returns:
        --------
        tuple
            A tuple containing the start time overlap and end time overlap of the processed data.

        Raises:
        -------
        ValueError
            If the last file does not fill the array completely.
        """
        # Initialize the offset pointer to 0
        # This pointer will be used to keep track of the position in the array
        offset_pointer = 0

        # Determine the start and end time of the chunk (with overlaps on both sides)
        start_time_overlap, end_time_overlap = self._determine_chunk_bounds()
        current_time = start_time_overlap
        while len(raw_files_list) > 0:
            # Load the last file in the list (FIFO)
            dir_path, file = raw_files_list[-1]
            # Get the timestamp of the file
            data_start_ts = self._get_file_timestamp(file)
            
            # Check if the start time overlap was not determined (equal to 0)
            # 1. It is the first run ever without any processed files
            # 2. The data is missing and the last file offset is set to -1 -> start time overlap is set to 0
            if start_time_overlap == 0:
                start_time_overlap, end_time_overlap = self._determine_first_chunk_bounds(data_start_ts)
                current_time = start_time_overlap

            # Check if the data is missing
            if np.abs(current_time - data_start_ts) > self.time_delay_threshold:
                log.warning(f"Data is missing between {current_time} and {data_start_ts}")
                # Set the last processed file to the current file (to skip previous file on next iteration)
                self.last_processed_file = file
                # Set the last file offset to -1 to indicate that the data is missing
                self.last_file_offset = -1
                return -1, -1
            
            # Load the data
            packet_data = self._load_segy_data(os.path.join(self.raw_data_path, dir_path, file))

            # Check if the number of channels in the data is consistent with the expected number of channels
            if packet_data.shape[0] != self.num_channels:
                log.warning(f"Number of channels in the data is inconsistent with the expected number of channels")
                log.warning(f"Expected: {self.num_channels}, Actual: {packet_data.shape[0]}")
                # Set the number of channels to the actual number of channels
                self.num_channels = packet_data.shape[0]
                # Reallocate the raw data array with the correct number of channels
                self._preallocate_raw_data_array()
                return -1, -1

            if data_start_ts < start_time_overlap:
                # Cut the data if it starts before the start time overlap
                packet_data = packet_data[:, int(round(start_time_overlap - data_start_ts, 1) * self.raw_sps) :]
                offset_pointer = int(round(start_time_overlap - data_start_ts, 1) * self.raw_sps)
                self.raw_data_array[:, :packet_data.shape[1]] = packet_data
                # Update the current time (used for checking for missing data)
                current_time += packet_data.shape[1] / self.raw_sps
            elif start_time_overlap + self.packet_size + (offset_pointer / self.raw_sps) >= end_time_overlap:
                # Cut the data if it ends after the end time overlap
                packet_data = packet_data[:, : int(round(end_time_overlap - data_start_ts, 1) * self.raw_sps)]
                self.raw_data_array[:, offset_pointer:] = packet_data
                # Set the last processed file and offset
                self.last_processed_file = file
                # Set the last file offset to the length of the cut data
                self.last_file_offset = packet_data.shape[1]
                offset_pointer += packet_data.shape[1]
                current_time += packet_data.shape[1] / self.raw_sps

                # Exit the loop if the end time overlap is reached (no need to load more files)
                break
            else:
                self.raw_data_array[:, offset_pointer : offset_pointer + packet_data.shape[1]] = packet_data
                offset_pointer += packet_data.shape[1]
                current_time += packet_data.shape[1] / self.raw_sps

            # Remove processed file from the list
            # **Note**: If file is last in the current chunk, it **will not** be removed,
            # because it will be used in the next chunk (if it was cut in the middle)
            raw_files_list.pop(-1)

        if offset_pointer != self.raw_data_array.shape[1]:
            # raise ValueError(f"Last file {file} does not fill the array completely")
            return -1, -1
        return start_time_overlap, end_time_overlap
    
    def _write_downsampled_output(self, down_data, start_time_overlap, end_time_overlap):
        """
        Write the downsampled data to an HDF5 file and store it in a directory structure based on the date.

        Parameters:
        -----------
        down_data : numpy.ndarray
            The downsampled data to be written to the file.
        start_time_overlap : float
            The start time of the overlap period in seconds since the epoch.
        end_time_overlap : float
            The end time of the overlap period in seconds since the epoch.
        """
        start_time = start_time_overlap + self.chunk_overlap
        year, month, day = datetime.fromtimestamp(start_time, tz=pytz.UTC).strftime("%Y-%m-%d").split("-")

        file_name = f"{start_time:.2f}.h5"
        if not os.path.exists(os.path.join(self.down_data_path, year, f"{year}{month}{day}")):
            os.makedirs(os.path.join(self.down_data_path, year, f"{year}{month}{day}"))
        file_path = os.path.join(self.down_data_path, year, f"{year}{month}{day}", file_name)
        with h5py.File(file_path, "w") as f:
            f.create_dataset("data_down", data=down_data)
            f.attrs["DX_down"] = self.dx
            f.attrs["SPS_down"] = self.sps
            f.attrs["down_factor_space"] = 1
            f.attrs["down_factor_time"] = 15

            f.attrs["_downsampler_version"] = self.version
            f.attrs["_downsampler_last_updated"] = self.last_updated
        log.info(f"Downsampled chunk starting at {start_time} written to {file_name}")
        
    def _update_last_file_status(self):
        """
        Update the status of the last processed file.
        """
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        with open(os.path.join(self.tmp_path, "status"), "w") as f:
            f.write(f"{self.last_processed_file},{self.last_file_offset}")
        
    def run(self):
        """
        Run the downsampling process.
        """
        log.info("Starting downsampling process")
        # Load the last processed file and offset
        self.last_processed_file, self.last_file_offset = self._read_last_file_status()
        # Fetch the list of raw files to be processed
        raw_files_list = self._list_raw_files()
        if not raw_files_list:
            log.warning("No raw files found")
            return
        # Read first file to number of traces
        first_file = raw_files_list[-1]
        first_file_path = os.path.join(self.raw_data_path, first_file[0], first_file[1])
        self.num_channels = self._load_segy_data(first_file_path).shape[0]
        self._preallocate_raw_data_array()

        while raw_files_list:
            log.info(f"Processing {len(raw_files_list)} files")
            start_time_overlap, end_time_overlap = self._load_chuck_raw_data(raw_files_list)
            if start_time_overlap == -1 and end_time_overlap == -1:
                log.warning(f"Problem with a chunk, skipping downsampling")
                continue
            down_data = self._downsample_raw_data()
            self._validate_downsampled_data(down_data)
            self._write_downsampled_output(down_data, start_time_overlap, end_time_overlap)
            self._update_last_file_status()
        log.info("Downsampling process completed")