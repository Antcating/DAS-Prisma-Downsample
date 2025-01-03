import os
from datetime import datetime, timedelta

import pytz
import numpy as np
from scipy.signal import decimate

import h5py

from concurrent.futures import ThreadPoolExecutor

RAW_DATA_PATH = "/Users/achating-macpro/Work/DAS-Systems/DAS-Prisma/DAS-Prisma-Downsample/input"
DOWN_DATA_PATH = "output"
TMP_PATH = "tmp"
PACKET_SIZE = 20 # in seconds
CHUNK_SIZE = 60 # in seconds
CHUNK_OVERLAP = 1 # in seconds
SPS = 100 # samples per second
DX = 9.8 # meters
RAW_SPS = 1500 # samples per second
NUM_CHANNELS = 3746

def multithreaded_mean(arr, num_thread):
    """
    Compute the mean of an array in a multithreaded manner.

    Parameters:
    -----------
    arr : numpy.ndarray
        The input array to compute the mean.
    num_thread : int
        The number of threads to use.

    Returns:
    --------
    numpy.ndarray
        The mean of the input array.
    """
    def mean_chunk(chunk):
        return np.mean(chunk, axis=-1, dtype=np.float32)
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        chunks = np.array_split(arr, num_thread, axis=1)
        results = executor.map(mean_chunk, chunks)
    result = np.hstack(list(results))
    return result

class Downsampler:
    def __init__(self):
        self.raw_data_array = np.zeros((NUM_CHANNELS, RAW_SPS * CHUNK_SIZE + 2 * RAW_SPS * CHUNK_OVERLAP))
        self.last_processed_file, self.last_file_offset = self._read_last_file_status()

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
    
    def _read_last_file_status(self):
        """
        Get the last processed file data.

        Returns:
        --------
        tuple
            The last processed file and its offset.
        """
        try:
            with open(os.path.join(TMP_PATH, "status"), "r") as f:
                last_processed_file, last_chunk_offset = f.read().split(",")
                return last_processed_file, int(last_chunk_offset)
        except FileNotFoundError:
            return "", 0

    def list_raw_files(self):
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
            dir for dir in os.listdir(RAW_DATA_PATH)
            if os.path.isdir(os.path.join(RAW_DATA_PATH, dir))
        ]

        for dir_path in sorted(dirs, key=lambda x: os.path.getmtime(os.path.join(RAW_DATA_PATH, x))):
            for root, dirs, files in os.walk(os.path.join(RAW_DATA_PATH, dir_path)):
                files = [file for file in files if file.endswith(".segy")]
                for file in sorted(files, key=lambda x: self._get_file_timestamp(x)):
                    file_timestamp = self._get_file_timestamp(file)
                    if file.endswith(".segy") and file_timestamp >= last_data_time:
                        raw_files_list.append([dir_path, file])
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
        with open(os.path.join(RAW_DATA_PATH, file_path), "rb") as f:
            f.seek(3714)
            traces = np.frombuffer(f.read(2), dtype=np.int16)[0]
        mmap_dtype = np.dtype([("headers", np.void, 240), ("data", "f4", traces)])
        segy_data = np.memmap(os.path.join(RAW_DATA_PATH, file_path), dtype=mmap_dtype, mode="r", offset=3600)
        data = segy_data["data"]
        return data

    def downsample_raw_data(self):
        """
        Downsample the raw data array.

        Returns:
        --------
        numpy.ndarray
            The downsampled data.
        """
        down_data = self._downsample_array(self.raw_data_array, factors=[3, 5], type="scipy")
        down_data = down_data[:, CHUNK_OVERLAP * SPS : -CHUNK_OVERLAP * SPS]
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
        if down_data.shape[1] != CHUNK_SIZE * SPS:
            raise ValueError(f"Downsampled data shape is {down_data.shape}, expected {(NUM_CHANNELS, CHUNK_SIZE * SPS)}")
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
                arr = arr.reshape(arr.shape[0], -1, factors[0])
                arr = multithreaded_mean(arr, factors[0])
                return decimate(arr, factors[1], axis=-1)
        elif type == "mean":
            if factors != []:
                raise NotImplementedError("Mean downsampling does not support multiple factors")
            arr = arr.reshape(arr.shape[0], -1, factor)
            return multithreaded_mean(arr, factor)
        
    def load_chuck_raw_data(self, raw_files_list):
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
        offset_pointer = 0

        if self.last_processed_file == "" or self.last_file_offset == -1:
            start_time_overlap = 0
            end_time_overlap = float("inf")
        else:
            last_data_time = self._get_file_timestamp(self.last_processed_file) + self.last_file_offset / RAW_SPS
            start_time_overlap = last_data_time - 2 * CHUNK_OVERLAP
            end_time_overlap = last_data_time + CHUNK_SIZE
            current_time = start_time_overlap
        print(f"Start time overlap: {start_time_overlap}")
        while len(raw_files_list) > 1:
            dir_path, file = raw_files_list[-1]
            data_start_ts = self._get_file_timestamp(file)
            
            if start_time_overlap == 0:
                start_time_overlap = data_start_ts
                end_time_overlap = data_start_ts + CHUNK_SIZE + 2 * CHUNK_OVERLAP            
                current_time = start_time_overlap

            if np.abs(current_time - data_start_ts) > self.time_delay_threshold:
                self.last_processed_file = file
                self.last_file_offset = -1
                return -1, -1
                raise ValueError(f'Data from {current_time} to {data_start_ts} is missing!')
            
            data = self._load_segy_data(os.path.join(RAW_DATA_PATH, dir_path, file))

            if data_start_ts < start_time_overlap:
                data = data[:, int(round(start_time_overlap - data_start_ts, 1) * RAW_SPS) :]
                offset_pointer = int(round(start_time_overlap - data_start_ts, 1) * RAW_SPS)
                self.raw_data_array[:, :data.shape[1]] = data
                current_time += data.shape[1] / RAW_SPS
            elif start_time_overlap + PACKET_SIZE + (offset_pointer / RAW_SPS) >= end_time_overlap:
                data = data[:, : int(round(end_time_overlap - data_start_ts, 1) * RAW_SPS)]
                self.raw_data_array[:, offset_pointer:] = data
                self.last_processed_file = file
                self.last_file_offset = data.shape[1]
                offset_pointer += data.shape[1]
                current_time += data.shape[1] / RAW_SPS
                break
            else:
                self.raw_data_array[:, offset_pointer : offset_pointer + data.shape[1]] = data
                offset_pointer += data.shape[1]
                current_time += data.shape[1] / RAW_SPS
            print(f"Loaded {file}")
            print(f"Data shape: {self.raw_data_array.shape}")
            print(f"Offset pointer: {offset_pointer}")
            print(f"Current time: {current_time}")
            raw_files_list.pop(-1)
        else:
            dir_path, file = raw_files_list[-1]
            data_start_ts = self._get_file_timestamp(file)
            
            if start_time_overlap == 0:
                start_time_overlap = data_start_ts
                end_time_overlap = data_start_ts + CHUNK_SIZE + 2 * CHUNK_OVERLAP
            
            current_time = start_time_overlap
            data = self._load_segy_data(os.path.join(RAW_DATA_PATH, dir_path, file))
            if data_start_ts < start_time_overlap:
                data = data[:, int(round(start_time_overlap - data_start_ts, 1) * RAW_SPS) :]
                offset_pointer = int(round(start_time_overlap - data_start_ts, 1) * RAW_SPS)
                self.raw_data_array[:, :data.shape[1]] = data
                current_time += data.shape[1] / RAW_SPS
            elif start_time_overlap + PACKET_SIZE + (offset_pointer / RAW_SPS) >= end_time_overlap:
                data = data[:, : int(round(end_time_overlap - data_start_ts, 1) * RAW_SPS)]
                self.raw_data_array[:, offset_pointer:] = data
                self.last_processed_file = file
                self.last_file_offset = data.shape[1]
                offset_pointer += data.shape[1]
                current_time += data.shape[1] / RAW_SPS
            else:
                self.raw_data_array[:, offset_pointer : offset_pointer + data.shape[1]] = data
                offset_pointer += data.shape[1]
                current_time += data.shape[1] / RAW_SPS
            print(f"Loaded {file}")
            print(f"Data shape: {self.raw_data_array.shape}")
            print(f"Offset pointer: {offset_pointer}")

            if offset_pointer != self.raw_data_array.shape[1]:
                raise ValueError(f"Last file has not filled the array. Offset pointer: {offset_pointer}, array shape: {self.raw_data_array.shape}")

        return start_time_overlap, end_time_overlap
    
    def write_downsampled_output(self, down_data, start_time_overlap, end_time_overlap):
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
        start_time = start_time_overlap + CHUNK_OVERLAP
        year, month, day = datetime.fromtimestamp(start_time, tz=pytz.UTC).strftime("%Y-%m-%d").split("-")

        file_name = f"{start_time:.2f}.h5"
        if not os.path.exists(os.path.join(DOWN_DATA_PATH, year, f"{year}{month}{day}")):
            os.makedirs(os.path.join(DOWN_DATA_PATH, year, f"{year}{month}{day}"))
        file_path = os.path.join(DOWN_DATA_PATH, year, f"{year}{month}{day}", file_name)
        with h5py.File(file_path, "w") as f:
            f.create_dataset("data_down", data=down_data)
            f.attrs["DX_down"] = DX
            f.attrs["SPS_down"] = SPS
            f.attrs["down_factor_space"] = 1
            f.attrs["down_factor_time"] = 15

            f.attrs["_downsampler_version"] = self.version
            f.attrs["_downsampler_last_updated"] = self.last_updated
        
    def update_last_file_status(self):
        """
        Update the status of the last processed file.
        """
        if not os.path.exists(TMP_PATH):
            os.makedirs(TMP_PATH)
        with open(os.path.join(TMP_PATH, "status"), "w") as f:
            f.write(f"{self.last_processed_file},{self.last_file_offset}")
        
    def run(self):
        """
        Run the downsampling process.
        """
        start_time = datetime.now(tz=pytz.UTC)
        raw_files_list = self.list_raw_files()
        while True:
            start_time_overlap, end_time_overlap = self.load_chuck_raw_data(raw_files_list)
            if start_time_overlap == -1 and end_time_overlap == -1:
                print(f"Data is missing!")
                continue
            down_data = self.downsample_raw_data()
            self._validate_downsampled_data(down_data)
            self.write_downsampled_output(down_data, start_time_overlap, end_time_overlap)
            self.update_last_file_status()
            self.raw_data_array = np.zeros((NUM_CHANNELS, RAW_SPS * CHUNK_SIZE + 2 * RAW_SPS * CHUNK_OVERLAP))
            print(f"Processed data from {start_time_overlap} to {end_time_overlap}")

if __name__ == "__main__":
    downsampler = Downsampler()
    downsampler.run()
