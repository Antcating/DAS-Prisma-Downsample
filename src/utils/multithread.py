import numpy as np
from scipy.signal import cheby1, filtfilt, firwin, lfilter, resample_poly, sosfilt, sosfiltfilt, upfirdn, zpk2sos
from concurrent.futures import ThreadPoolExecutor
import operator

def decimate(x, q, n=None, ftype='iir', axis=-1, zero_phase=True, num_thread=4):
    """
    Downsample the signal after applying an anti-aliasing filter.

    This is a modified version of SciPy's `decimate` function for use in 
    multithreaded environments. The original function can be found at: 
    https://github.com/scipy/scipy/blob/v1.14.1/scipy/signal/_signaltools.py#L4497-L4655

    By default, an order 8 Chebyshev type I filter is used. A 30 point FIR
    filter with Hamming window is used if `ftype` is 'fir'.

    Parameters
    ----------
    x : array_like
        The signal to be downsampled, as an N-dimensional array.
    q : int
        The downsampling factor. When using IIR downsampling, it is recommended
        to call `decimate` multiple times for downsampling factors higher than
        13.
    n : int, optional
        The order of the filter (1 less than the length for 'fir'). Defaults to
        8 for 'iir' and 20 times the downsampling factor for 'fir'.
    ftype : str {'iir', 'fir'} or ``dlti`` instance, optional
        If 'iir' or 'fir', specifies the type of lowpass filter. If an instance
        of an `dlti` object, uses that object to filter before downsampling.
    axis : int, optional
        The axis along which to decimate.
    zero_phase : bool, optional
        Prevent phase shift by filtering with `filtfilt` instead of `lfilter`
        when using an IIR filter, and shifting the outputs back by the filter's
        group delay when using an FIR filter. The default value of ``True`` is
        recommended, since a phase shift is generally not desired.

        .. versionadded:: 0.18.0

    Returns
    -------
    y : ndarray
        The down-sampled signal.

    See Also
    --------
    resample : Resample up or down using the FFT method.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    The ``zero_phase`` keyword was added in 0.18.0.
    The possibility to use instances of ``dlti`` as ``ftype`` was added in
    0.18.0.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    Define wave parameters.

    >>> wave_duration = 3
    >>> sample_rate = 100
    >>> freq = 2
    >>> q = 5

    Calculate number of samples.

    >>> samples = wave_duration*sample_rate
    >>> samples_decimated = int(samples/q)

    Create cosine wave.

    >>> x = np.linspace(0, wave_duration, samples, endpoint=False)
    >>> y = np.cos(x*np.pi*freq*2)

    Decimate cosine wave.

    >>> ydem = signal.decimate(y, q)
    >>> xnew = np.linspace(0, wave_duration, samples_decimated, endpoint=False)

    Plot original and decimated waves.

    >>> plt.plot(x, y, '.-', xnew, ydem, 'o-')
    >>> plt.xlabel('Time, Seconds')
    >>> plt.legend(['data', 'decimated'], loc='best')
    >>> plt.show()

    """

    x = np.asarray(x)
    q = operator.index(q)

    if n is not None:
        n = operator.index(n)

    result_type = x.dtype
    if not np.issubdtype(result_type, np.inexact) \
       or result_type.type == np.float16:
        # upcast integers and float16 to float64
        result_type = np.float64

    if ftype == 'fir':
        if n is None:
            half_len = 10 * q  # reasonable cutoff for our sinc-like function
            n = 2 * half_len
        b, a = firwin(n+1, 1. / q, window='hamming'), 1.
        b = np.asarray(b, dtype=result_type)
        a = np.asarray(a, dtype=result_type)
    elif ftype == 'iir':
        iir_use_sos = True
        if n is None:
            n = 8
        sos = cheby1(n, 0.05, 0.8 / q, output='sos')
        sos = np.asarray(sos, dtype=result_type)
    else:
        raise ValueError('invalid ftype')

    sl = [slice(None)] * x.ndim

    def process_chunk(chunk):
        if ftype == 'fir':
            b_chunk = b / a
            if zero_phase:
                return resample_poly(chunk, 1, q, axis=axis, window=b_chunk)
            else:
                n_out = chunk.shape[axis] // q + bool(chunk.shape[axis] % q)
                y_chunk = upfirdn(b_chunk, chunk, up=1, down=q, axis=axis)
                sl[axis] = slice(None, n_out, None)
                return y_chunk[tuple(sl)]
        else:  # IIR case
            if zero_phase:
                if iir_use_sos:
                    return sosfiltfilt(sos, chunk, axis=axis)
                else:
                    return filtfilt(b, a, chunk, axis=axis)
            else:
                if iir_use_sos:
                    return sosfilt(sos, chunk, axis=axis)
                else:
                    return lfilter(b, a, chunk, axis=axis)
    # Split the array into chunks along the specified axis
    chunks = np.array_split(x, num_thread, axis=0)
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        results = list(executor.map(process_chunk, chunks))

    # Concatenate the results along the specified axis
    y = np.concatenate(results, axis=0)

    sl[axis] = slice(None, None, q)
    return y[tuple(sl)]

def multithreaded_mean(arr, axis=-1, num_thread=4):
    """
    Compute the mean of an array in a multithreaded manner.

    This function splits the input array into chunks and computes the mean of 
    each chunk in parallel using multiple threads.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array to compute the mean.
    axis : int, optional
        The axis along which to compute the mean. Defaults to -1.
    num_thread : int
        The number of threads to use.

    Returns
    -------
    numpy.ndarray
        The mean of the input array.
    """
    def mean_chunk(chunk):
        return np.mean(chunk, axis=axis, dtype=np.float32)
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        chunks = np.array_split(arr, num_thread, axis=1)
        results = executor.map(mean_chunk, chunks)
    result = np.hstack(list(results))
    return result
