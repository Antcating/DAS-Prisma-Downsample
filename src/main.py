from downsample.main import Downsampler

import time

def main():
    start = time.time()
    ds = Downsampler(num_threads=8)
    ds.run()
    end = time.time()
    print(f"Time taken: {end - start} seconds")

if __name__ == '__main__':
    main()