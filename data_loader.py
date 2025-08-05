import numpy as np  # 用于数值计算，如数组操作
import mne  # 强大的 EEG/MEG 数据处理库
from scipy.io import loadmat  # 用于读取 .mat 标签文件
import os  # 用于路径拼接等文件操作

'''
该脚本用于加载和预处理 BCI Competition IV 2a 的 EEG 数据（GDF格式），
包括通道重命名、坏值修复、标签提取等操作
'''


class BCICompetition4Set2A:
    def __init__(self, filename, labels_filename=None):
        # 初始化数据文件路径及其对应的标签文件路径（.mat）
        self.filename = filename
        self.labels_filename = labels_filename

    def load(self):
        cnt = self.extract_data()  # 提取 EEG 原始数据
        events, artifact_trial_mask = self.extract_events(cnt)  # 获取事件和伪迹掩码
        return cnt, events, artifact_trial_mask

    # def load(self):
    #     '''
    #     载入完整预处理后的数据，包括事件和伪迹遮罩信息
    #     '''
    #     cnt = self.extract_data()  # 提取、清洗 EEG 原始数据
    #     events, artifact_trial_mask = self.extract_events(cnt)  # 提取事件信息及坏试次掩码
    #     cnt.info["events"] = events
    #     cnt.info["artifact_trial_mask"] = artifact_trial_mask
    #     return cnt  # 返回预处理完的数据对象（Raw）

    def extract_data(self):
        '''
        读取 GDF 文件，重命名通道，修复异常值（如最小值标记），输出标准化 EEG 数据
        '''
        # 读取 GDF 数据，自动识别事件通道，同时排除眼电通道
        raw_gdf = mne.io.read_raw_gdf(self.filename, stim_channel="auto", verbose='ERROR',
                                      exclude=(["EOG-left", "EOG-central", "EOG-right"]))

        # 将原始 EEG 通道名重命名为标准名称（用于后续通道对齐）
        raw_gdf.rename_channels({
            'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
            'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4',
            'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
            'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'
        })

        raw_gdf.load_data()  # 加载原始数据至内存

        data = raw_gdf.get_data()  # 获取原始 EEG 数组（形状为 channels × timepoints）

        for i_chan in range(data.shape[0]):
            # 对每个通道：
            this_chan = data[i_chan]
            # 将通道中最小值（可能是标记的错误值）替换为 NaN
            data[i_chan] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
            # 创建 NaN 掩码
            mask = np.isnan(data[i_chan])
            # 计算非 NaN 均值
            chan_mean = np.nanmean(data[i_chan])
            # 用均值替换 NaN（填补坏点）
            data[i_chan, mask] = chan_mean

        # 从原始数据中提取事件信息（基于标注）
        gdf_events = mne.events_from_annotations(raw_gdf)
        # 用清洗后的数据重新构造 RawArray
        raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="ERROR")
        # 保存事件字典
        # raw_gdf.info["gdf_events"] = gdf_events
        if 'temp' not in raw_gdf.info:
            raw_gdf.info['temp'] = {}
        raw_gdf.info['temp']['gdf_events'] = gdf_events

        return raw_gdf

    def extract_events(self, raw_gdf):
        '''
        提取 trial 的开始时间和对应类别标签（包括处理伪迹标记 trial）
        '''
        # 获取事件和标签映射表
        # events, name_to_code = raw_gdf.info["gdf_events"]
        events, name_to_code = raw_gdf.info["temp"]["gdf_events"]

        # 判断是否为训练集（包含运动想象任务标签 769–772）
        if "769" and "770" and "771" and "772" in name_to_code:
            train_set = True
        else:
            train_set = False
            # 测试集必须至少包含标签 "783"
            assert "783" in name_to_code

        # 根据是否为训练集选择事件编码
        if train_set:
            if self.filename[-8:] == 'A04T.gdf':  # 特殊 case：A04 编号的数据稍有不同
                trial_codes = [5, 6, 7, 8]
            else:
                trial_codes = [7, 8, 9, 10]
        else:
            trial_codes = [7]  # 测试集仅包含未知标签

        # 创建 trial mask，筛选出合法事件
        trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
        trial_events = events[trial_mask]

        # 确保每个 session 都有 288 条有效 trial
        assert len(trial_events) == 288, f"Got {len(trial_events)} markers"

        # 将事件编号转换为标准标签 1-4（左手、右手、脚、舌头）
        if train_set and self.filename[-8:-4] == 'A04T':
            trial_events[:, 2] = trial_events[:, 2] - 4
        else:
            trial_events[:, 2] = trial_events[:, 2] - 6

        # 如果存在标签文件（.mat），则使用其 label 替换
        if self.labels_filename is not None:
            classes = loadmat(self.labels_filename)["classlabel"].squeeze()
            if train_set:
                # 验证两个标签一致（mat 标签和 gdf 标签）
                np.testing.assert_array_equal(trial_events[:, 2], classes)
            trial_events[:, 2] = classes  # 最终使用 mat 文件中的标签

        # 确保标签为 1, 2, 3, 4 四类
        unique_classes = np.unique(trial_events[:, 2])
        assert np.array_equal([1, 2, 3, 4], unique_classes), \
            f"Expect 1,2,3,4 as class labels, got {str(unique_classes)}"

        # 构建 artifact trial 的掩码（是否为伪迹）
        if train_set and self.filename[-8:-5] == 'A04':
            trial_start_events = events[events[:, 2] == 4]
        else:
            trial_start_events = events[events[:, 2] == 6]
        assert len(trial_start_events) == len(trial_events)

        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        artifact_events = events[events[:, 2] == 1]  # 标记为 artifact 的事件（噪声等）

        for artifact_time in artifact_events[:, 0]:
            i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
            artifact_trial_mask[i_trial] = 1  # 标记该 trial 为伪迹

        return trial_events, artifact_trial_mask


# def extract_segment_trial(raw_gdb, baseline=(-0.5, 0), duration=4):
#     '''
#     将原始 EEG 切分为以 cue 为中心的 trial 数据段
#     baseline: (前置时间, 后延时间)，单位为秒
#     duration: 运动想象时长（单位秒）
#     '''
#     events = raw_gdb.info['events']  # cue 时刻及对应标签
#     raw_data = raw_gdb.get_data()  # 获取 EEG 原始矩阵
#     freqs = raw_gdb.info['sfreq']  # 采样率

#     # 计算采样点数（将时间转换为样本点）
#     mi_duration = int(freqs * duration)
#     duration_before_mi = int(freqs * baseline[0])
#     duration_after_mi = int(freqs * baseline[1])

#     labels = np.array(events[:, 2])  # 每个 trial 的标签

#     trial_data = []
#     for i_event in events:
#         # 根据事件时间索引提取信号片段（含 baseline）
#         segmented_data = raw_data[:, int(i_event[0]) + duration_before_mi:
#                                      int(i_event[0]) + mi_duration + duration_after_mi]
#         # 验证 trial 长度一致
#         assert segmented_data.shape[-1] == mi_duration - duration_before_mi + duration_after_mi
#         trial_data.append(segmented_data)
#     trial_data = np.stack(trial_data, 0)  # 转为三维数组 trials × channels × timepoints

#     return trial_data, labels

def extract_segment_trial(raw_data, events, sfreq, baseline=(-0.5, 0), duration=4):
    """
    将原始 EEG 切分为以 cue 为中心的 trial 数据段
    参数:
        raw_data: ndarray, shape=(channels, timepoints)，原始EEG数据
        events: ndarray, shape=(n_events, 3)，每一行包含 [时间点, 0, 标签]
        sfreq: float，采样率
        baseline: tuple，baseline时间窗（单位：秒）
        duration: float，运动想象时间（单位：秒）
    返回:
        trial_data: ndarray, shape=(trials, channels, timepoints)
        labels: ndarray, shape=(trials,)
    """
    # 计算采样点数（将时间转换为样本点）
    mi_duration = int(sfreq * duration)
    duration_before_mi = int(sfreq * baseline[0])
    duration_after_mi = int(sfreq * baseline[1])

    labels = np.array(events[:, 2])  # 每个 trial 的标签

    trial_data = []
    for i_event in events:
        start = int(i_event[0]) + duration_before_mi
        end = int(i_event[0]) + mi_duration + duration_after_mi
        segmented_data = raw_data[:, start:end]
        # 验证 trial 长度一致
        assert segmented_data.shape[-1] == mi_duration - duration_before_mi + duration_after_mi
        trial_data.append(segmented_data)
    trial_data = np.stack(trial_data, 0)  # 转为三维数组 trials × channels × timepoints

    return trial_data, labels


# 测试代码：用于调试加载特定 subject 的训练和测试数据
if __name__ == '__main__':
    subject_id = 4
    data_path = "/root/autodl-tmp/LMDA-Code-main/bci2a_data/"  # 数据根目录

    # 构建训练/测试文件路径
    train_filename = f"A{subject_id:02d}T.gdf"
    test_filename = f"A{subject_id:02d}E.gdf"
    train_filepath = os.path.join(data_path, train_filename)
    test_filepath = os.path.join(data_path, test_filename)
    train_label_filepath = train_filepath.replace(".gdf", ".mat")
    test_label_filepath = test_filepath.replace(".gdf", ".mat")

    # 实例化加载器，加载原始数据
    train_loader = BCICompetition4Set2A(train_filepath, labels_filename=train_label_filepath)
    test_loader = BCICompetition4Set2A(test_filepath, labels_filename=test_label_filepath)

    # 加载数据并清洗
    # train_cnt = train_loader.load()
    # test_cnt = test_loader.load()
    train_cnt, train_events, train_artifact_mask = train_loader.load()
    test_cnt, test_events, test_artifact_mask = test_loader.load()

    # 再次手动删除眼电通道
    train_cnt = train_cnt.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
    test_cnt = test_cnt.drop_channels(["EOG-left", "EOG-central", "EOG-right"])

    # 打印 shape 验证是否加载成功（channels × timepoints）
    print(train_cnt.get_data().shape)
    print(test_cnt.get_data().shape)
