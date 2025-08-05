from scipy.signal import firwin, lfilter, filtfilt, butter
import numpy as np
from scipy.linalg import sqrtm
import mne
# 功能主要包括 带通滤波、归一化、空-均协方差对齐预处理

def mne_apply(func, raw, verbose="WARNING"):
    """
    Apply function to data of `mne.io.RawArray`.

    Parameters
    ----------
    func: function
        Should accept 2d-array (channels x time) and return modified 2d-array
    raw: `mne.io.RawArray`
    verbose: bool
        Whether to log creation of new `mne.io.RawArray`.

    Returns
    -------
    transformed_set: Copy of `raw` with data transformed by given function.

    """
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=verbose)

# 使用了 Blackman窗的FIR带通滤波器，截止频率为 [4, 38] Hz（典型的 MI 频段）。
# zero_phase=True 时会采用 filtfilt 实现 零相位滤波（前向后向双向滤波），否则使用 lfilter
def bandpass_cnt(data, low_cut_hz, high_cut_hz, fs, filt_order=200, zero_phase=False):
    # nyq_freq = 0.5 * fs
    # low = low_cut_hz / nyq_freq
    # high = high_cut_hz / nyq_freq

    # win = firwin(filt_order, [low, high], window='blackman', ass_zero='bandpass')
    win = firwin(filt_order, [low_cut_hz, high_cut_hz], window='blackman', fs=fs, pass_zero='bandpass')

    data_bandpassed = lfilter(win, 1, data)
    if zero_phase:
        data_bandpassed = filtfilt(win, 1, data)
    return data_bandpassed


def data_norm(data):
    """
    对数据进行归一化
    :param data:   ndarray ,shape[N,channel,samples]
    :return:
    每个 trial（data[i]）除以其绝对最大值，实现 逐 trial 最大值归一化，将每段 EEG 控制在 [-1, 1] 范围。
    """
    data_copy = np.copy(data)
    for i in range(len(data)):
        data_copy[i] = data_copy[i] / np.max(abs(data[i]))

    return data_copy


def prepare_data(data):
    # [-1,1]
    """
    归一化 → 空间对齐 → 在第1维插入channel维度（变成[N, 1, C, T]）以匹配网络输入格式。
    """

    data_preprocss = data_norm(data)
    data_ea = preprocess_ea(data_preprocss)

    data_pre = np.expand_dims(data_ea, axis=1)

    return data_pre

# Euclidean Alignment 空间对齐
def preprocess_ea(data):
    """
        计算所有 trial 的协方差矩阵的平均值 R̄（样本协方差估计）。
        使用欧几里得对齐方法（EA）将每个 trial 的通道数据左乘 R̄^(-1/2)，对齐不同 trial 的协方差结构。
        本质是对 EEG 信号在空间结构上做标准化，提升跨 session / subject 迁移性能。
    """
    R_bar = np.zeros((data.shape[1], data.shape[1]))
    for i in range(len(data)):
        R_bar += np.dot(data[i], data[i].T)
    R_bar_mean = R_bar / len(data)
    # assert (R_bar_mean >= 0 ).all(), 'Before squr,all element must >=0'

    for i in range(len(data)):
        data[i] = np.dot(np.linalg.inv(sqrtm(R_bar_mean)), data[i])
    return data


def preprocess4mi(data):  # data: ndarray with shape, nums, chans, samples
    # data_filter = bandpass_cnt(data,  low_cut_hz=4, high_cut_hz=38, fs=250)
    data_preprocessed = prepare_data(data)
    return data_preprocessed