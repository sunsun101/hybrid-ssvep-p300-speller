"""
Utils for classification module

Author: Juan Jesús Torre, Ludovic Darmet, Giuseppe Ferraro
Mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr, Ludovic.DARMET@isae-supaero.fr, Giuseppe.FERRARO@isae-supaero.fr
"""

import numpy as np
import scipy.signal as scp


def filterbank(data, sfreq, idx_fb, peaks):
    """
    Filter bank design for decomposing EEG data into sub-band components [1]

    Parameters
    ----------

    data: np.array, shape (trials, channels, samples) or (channels, samples)
        EEG data to be processed

    sfreq: int
        Sampling frequency of the data.

    idx_fb: int
        Index of filters in filter bank analysis

    peaks : list of len (n_classes)
        Frequencies corresponding to the SSVEP components.

    Returns
    -------

    y: np.array, shape (trials, channels, samples)
        Sub-band components decomposed by a filter bank

    Reference:
      [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
          "Enhancing detection of SSVEPs for a high-speed brain speller using
           task-related component analysis",
          IEEE Trans. Biomed. Eng, 65(1):104-112, 2018.

    Code based on the Matlab implementation from authors of [1]
    (https://github.com/mnakanishi/TRCA-SSVEP).
    """

    # Calibration data comes in batches of trials
    if data.ndim == 3:
        num_chans = data.shape[1]
        num_trials = data.shape[0]

    # Testdata come with only one trial at the time
    elif data.ndim == 2:
        num_chans = data.shape[0]
        num_trials = 1

    sfreq = sfreq / 2

    min_freq = np.min(peaks)
    max_freq = np.max(peaks)
    if max_freq < 40:
        top = 90
    else:
        top = 115
    diff = min_freq
    # Lowcut frequencies for the pass band (depends on the frequencies of SSVEP)
    # No more than 3dB loss in the passband
    passband = [min_freq + x * diff for x in range(7)]

    # At least 40db attenuation in the stopband
    if min_freq - 4 > 0:
        stopband = [
            min_freq - 4 + x * (diff - 2) if x < 3 else min_freq - 4 + x * diff
            for x in range(7)
        ]
    else:
        stopband = [2 + x * (diff - 2) if x < 3 else 2 + x * diff for x in range(7)]

    Wp = [passband[idx_fb] / sfreq, top / sfreq]
    Ws = [stopband[idx_fb] / sfreq, (top + 5) / sfreq]
    N, Wn = scp.cheb1ord(Wp, Ws, 3, 40)  # Chebyshev type I filter order selection.

    B, A = scp.cheby1(N, 0.5, Wn, btype="bandpass")  #  Chebyshev type I filter design

    y = np.zeros(data.shape)
    if num_trials == 1:  # For testdata
        for ch_i in range(num_chans):
            try:
                # The arguments 'axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1)' correspond
                # to Matlab filtfilt (https://dsp.stackexchange.com/a/47945)
                y[ch_i, :] = scp.filtfilt(
                    B,
                    A,
                    data[ch_i, :],
                    axis=0,
                    padtype="odd",
                    padlen=3 * (max(len(B), len(A)) - 1),
                )
            except Exception as e:
                print(e)
                print(f'Error on channel {ch_i}')
    else:
        for trial_i in range(num_trials):  # Filter each trial sequentially
            for ch_i in range(num_chans):  # Filter each channel sequentially
                y[trial_i, ch_i, :] = scp.filtfilt(
                    B,
                    A,
                    data[trial_i, ch_i, :],
                    axis=0,
                    padtype="odd",
                    padlen=3 * (max(len(B), len(A)) - 1),
                )
    return y