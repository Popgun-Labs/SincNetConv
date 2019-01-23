import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def sinc(band, t_right):
    """
    :param band: (n_filt)
    :param t_right: K = filt_dim // 2 - 1
    :return: (n_filt, filt_dim)
    """
    n_filt = band.size(0)
    band = band[:, None]  # (n_filt, 1)
    t_right = t_right[None, :]  # (1, K)
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)  # (n_filt, K)
    y_left = torch.flip(y_right, [1])
    y = torch.cat([y_left, torch.ones([n_filt, 1], device=band.device), y_right], dim=1)  # (n_filt, filt_dim)
    return y


def get_mel_points(fs, n_filt, fmin=80):
    """
    Return `n_filt` points linearly spaced in the mel-scale (in Hz)
    with an upper frequency of fs / 2
    :param fs: the sample rate in Hz
    :return: np.array (n_filt)
    """
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))
    mel_points = np.linspace(fmin, high_freq_mel, n_filt)  # equally spaced in mel scale
    f_cos = (700 * (10 ** (mel_points / 2595) - 1))
    return f_cos


def get_bands(f_cos, fs):
    """
    :param f_cos: vector of mel-scaled frequency (n_filt)
    :param fs: audio sample rate
    :return (b1, b1)
        b1: vector of lower cutoffs (n_filt)
        b2: vector of upper cutoffs
    """
    b1 = np.roll(f_cos, 1)
    b2 = np.roll(f_cos, -1)
    b1[0] = 30
    b2[-1] = (fs / 2) - 100
    return b1, b2


class SincConv(nn.Module):
    def __init__(self, n_filt, filt_dim, fs):
        """
        :param n_filt: number of filters
        :param filt_dim: filter width
        :param fs: audio sample rate in Hz
        """
        super(SincConv, self).__init__()
        self.n_filt = n_filt
        self.filt_dim = filt_dim
        self.fs = fs

        # set minimum cutoff and bandwidth
        self.min_freq = 50.0
        self.min_band = 50.0

        # calculate the band params
        f_cos = get_mel_points(fs, n_filt)
        b1, b2 = get_bands(f_cos, fs)

        # learnable params
        filt_b1 = torch.from_numpy(b1 / self.fs).float()
        filt_band = torch.from_numpy((b2 - b1) / self.fs).float()
        self.filt_b1 = nn.Parameter(filt_b1)
        self.filt_band = nn.Parameter(filt_band)

        # define the window function
        hamming_window = torch.from_numpy(np.hamming(filt_dim)).float()
        self.register_buffer('hamming_window', hamming_window)

        # define the linspace for the sinc function here
        t_right = torch.linspace(1, (filt_dim - 1) / 2, steps=int((filt_dim - 1) / 2)) / fs
        self.register_buffer('t_right', t_right.float())

    def forward(self, x):
        """
        :param x: (batch, 1, length)
        :return: (batch, n_filt, length)
        """

        # convert bandwidth to upper cutoff and enforce they are >= min values
        filt_beg_freq = torch.abs(self.filt_b1) + self.min_freq / self.fs
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + self.min_band / self.fs)

        # construct the filter bank
        low_pass1 = 2 * filt_beg_freq[:, None] * sinc(filt_beg_freq * self.fs, self.t_right)
        low_pass2 = 2 * filt_end_freq[:, None] * sinc(filt_end_freq * self.fs, self.t_right)
        band_pass = (low_pass2 - low_pass1)
        max_band, _ = torch.max(band_pass, dim=1, keepdim=True)
        band_pass = band_pass / max_band  # (n_filt, filt_dim)
        filters = band_pass * self.hamming_window[None, ]  # (n_filt, filt_dim)

        # apply padding to preserve length dimension and convolve
        padding = (self.filt_dim - 1) // 2
        out = F.conv1d(x, filters.view(self.n_filt, 1, self.filt_dim), padding=padding)

        return out
