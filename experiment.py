from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import logging
import torch.nn as nn
import matplotlib.pyplot as plt


def setup_seed(seed=521):  # 固定随机种子函数
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class EEGDataLoader(Dataset):  # 封装 EEG 数据为 PyTorch Dataset 格式，以便后续使用 DataLoader 迭代训练。
    def __init__(self, x, y):
        self.data = torch.from_numpy(x)
        self.labels = torch.from_numpy(y)  # label without one-hot coding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = self.data[idx]
        label_tensor = self.labels[idx]
        return data_tensor, label_tensor


class Measurement(object):  # 测试评估指标封装类
    '''
    用于评估模型测试集精度：包括最大精度、后10轮平均精度、标准差、偏移量等指标
    '''

    def __init__(self, test_df, classes):
        '''
        :param test_df: DataFrame of test, 将其转化为ndarray似乎更方便;
        :return: None
        '''
        self.test_df = test_df.values
        self.classes = classes

    def max_acc(self):  # using kappa for MI, AUC for P300
        return self.test_df.max()

    def last10_acc(self):
        return self.test_df[-10:].mean()

    def last10_std(self):
        return self.test_df[-10:].std()

    def max_mean_offset(self):
        """
        计算最大精度与平均精度的偏差值（衡量模型在不同被试间精度差异的指标）
        """
        # 最大值与均值的偏差（绝对差）
        return abs(self.max_acc() - self.last10_acc())

    def index_equation(self):
        # 综合评分指标：0.4*max + 0.4*mean - 0.2*offset
        max_val = self.max_acc()
        mean_val = self.last10_acc()
        offset = self.max_mean_offset()
        return 0.4 * max_val + 0.4 * mean_val - 0.2 * offset


class Experiment(object):
    # 该类封装了训练过程的全部逻辑，含训练、验证、测试、日志、绘图等功能
    def __init__(self, model, optimizer, train_dl, test_dl, val_dl, fig_path, device='cuda:0',
                 step_one=300, model_constraint=None,
                 p300_flag=False, imbalanced_class=1, classes=2,
                 ):
        self.model = model  # 传入的模型
        self.optim4model = optimizer  # 优化器
        self.model_constraint = model_constraint  # 权重限制器（如正则）
        self.best_test = float("-inf")  # 当前最优 test_acc
        self.step_one_epoch = step_one  # 训练轮数上限
        self.device = device  # 使用的设备（GPU 或 CPU）
        # for imbalanced classed 处理类别不均衡（例如ERN或P300场景）：
        self.classes = classes
        class_weight = [1.0] * classes
        class_weight[0] *= imbalanced_class  # kaggle ERN, 0:1=3:7, 因此imbalanced_weight=3/7
        class_weight = torch.FloatTensor(class_weight).to(device)
        self.nllloss = nn.CrossEntropyLoss(weight=class_weight)
        # 封装为字典
        self.datasets = OrderedDict((("train", train_dl), ("valid", val_dl), ("test", test_dl)))
        if val_dl is None:
            self.datasets.pop("valid")
        if test_dl is None:
            self.datasets.pop("test")
        self.fig_path = fig_path
        # initialize epoch dataFrame instead of loss and acc for train and test
        self.train_df = pd.DataFrame()  # 训练日志
        self.val_df = pd.DataFrame()  # 验证日志
        self.epoch_df = pd.DataFrame()  # 总日志
        # p300 # 是否是P300任务（用于切换评估指标）
        self.p300_flag = p300_flag

    def train_epoch(self):  # 训练一个 epoch
        '''
        遍历训练集；
        前向传播；
        计算 loss（交叉熵）；
        根据任务类型选择精度或 AUC；
        反向传播优化；
        记录训练过程
        '''
        self.model.train()
        batch_size, cls_loss, train_acc, all_labels = [], [], [], []
        for i, (train_input, train_target) in enumerate(self.datasets['train']):
            train_input = train_input.to(self.device).float()
            train_label = train_target.to(self.device).long()
            source_softmax = self.model(train_input)  # 前向传播
            nll_loss = self.nllloss(source_softmax, train_label)  # 交叉熵损失

            batch_size.append(len(train_input))
            if self.p300_flag:  # AUC instead of acc
                # get AUC
                source_softmax = nn.Softmax(dim=0)(source_softmax)
                batch_acc = source_softmax.cpu().detach().numpy()[:, 1]
                all_labels.append(train_label.cpu().detach().numpy())
            else:  # MI任务，计算准确率
                _, predicted = torch.max(source_softmax.data, 1)
                batch_acc = np.equal(predicted.cpu().detach().numpy(), train_label.cpu().detach().numpy()).sum() / len(
                    train_label)
            train_acc.append(batch_acc)
            cls_loss_np = nll_loss.cpu().detach().numpy()
            cls_loss.append(cls_loss_np)
            # 反向传播与优化
            self.optim4model.zero_grad()
            nll_loss.backward()
            self.optim4model.step()
            # 是否添加约束，如BN限制
            if self.model_constraint is not None:
                self.model_constraint.apply(self.model)
        # 整体评估指标计算
        if self.p300_flag:
            train_acc = np.concatenate(train_acc, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            epoch_acc = roc_auc_score(all_labels, train_acc)
        else:
            epoch_acc = sum(train_acc) / len(train_acc) * 100
        epoch_loss = sum(cls_loss) / len(cls_loss)
        # 保存该 epoch 的训练指标
        train_dicts_per_epoch = OrderedDict()

        cls_loss = {'train_loss': epoch_loss}
        train_acc = {'train_acc': epoch_acc}
        train_dicts_per_epoch.update(cls_loss)
        train_dicts_per_epoch.update(train_acc)

        # self.train_df = self.train_df.append(train_dicts_per_epoch, ignore_index=True)

        self.train_df = pd.concat([self.train_df, pd.DataFrame([train_dicts_per_epoch])], ignore_index=True)
        self.train_df = self.train_df[list(train_dicts_per_epoch.keys())]  # 让epoch_df中的顺序和row_dict中的一致

    def test_batch(self, input, target):  # **评估单个 batch（在 test/val 中调用）**
        self.model.eval()
        with torch.no_grad():
            val_input = input.to(self.device).float()
            val_target = target.to(self.device).long()
            val_fc1 = self.model(val_input)  # 前向传播
            loss = self.nllloss(val_fc1, val_target)  # 计算损失
            if self.p300_flag:
                source_softmax = nn.Softmax(dim=0)(val_fc1)
                preds = source_softmax.cpu().detach().numpy()[:, 1]
            else:
                _, preds = torch.max(val_fc1.data, 1)
                preds = preds.cpu().detach().numpy()
            loss = loss.cpu().detach().numpy()
        return preds, loss

    def monitor_epoch(self, datasets):
        # 强制使用训练时定义的测试集，即确保进行测试集评估
        datasets['test'] = self.datasets['test']

        # 用于存储每个评估集（这里只是 test）的损失与准确率
        result_dicts_per_monitor = OrderedDict()

        # 遍历传入的数据集（这里只包括 test）
        for setname in datasets:
            # 不能是训练集（monitor 不做训练集评估）
            assert setname != 'train', 'dataset without train set'
            # 只能是 test（或者未来可能扩展到 valid）
            assert setname in ["test"]

            dataset = datasets[setname]  # 取出具体的数据加载器
            batch_size, epoch_loss, epoch_acc, test_labels = [], [], [], []

            # 遍历测试集每一个 batch
            for i, (input, target) in enumerate(dataset):
                # 调用 test_batch 得到模型输出的预测结果和对应的损失
                pred, loss = self.test_batch(input, target)

                # 累加 loss，用于后面求平均损失
                epoch_loss.append(loss)
                # 保存当前 batch 的样本数
                batch_size.append(len(target))

                if self.p300_flag:  # 如果是 P300 类任务，使用 AUC 而非准确率
                    epoch_acc.append(pred)  # 保存预测概率（用于 AUC）
                    test_labels.append(target.cpu().detach().numpy())  # 保存真实标签
                else:
                    # 计算当前 batch 的准确率（预测值与真实值一致的个数）
                    epoch_acc.append(np.equal(pred, target.numpy()).sum())

                    # 如果是 P300，使用 AUC 作为准确性指标
            if self.p300_flag:
                epoch_acc = np.concatenate(epoch_acc, axis=0)  # 拼接所有预测结果
                test_labels = np.concatenate(test_labels, axis=0)  # 拼接所有真实标签
                epoch_acc = roc_auc_score(test_labels, epoch_acc)  # 计算 AUC 得分
            else:
                # 否则使用常规的准确率，= 正确分类总数 / 样本总数
                epoch_acc = sum(epoch_acc) / sum(batch_size) * 100

                # 计算整个测试集上的平均 loss
            epoch_loss = sum(epoch_loss) / len(epoch_loss)

            # 定义结果的键名，例如：'test_loss'、'test_acc'
            key_loss = setname + '_loss'
            key_acc = setname + '_acc'

            # 将损失和准确率封装为字典
            loss = {key_loss: epoch_loss}
            acc = {key_acc: epoch_acc}

            # 将结果添加到汇总字典中
            result_dicts_per_monitor.update(loss)
            result_dicts_per_monitor.update(acc)

            # 如果当前测试集准确率超过历史最佳，更新 best_test 并记录日志
            if epoch_acc > self.best_test:
                self.best_test = epoch_acc
                logging.info("New best test acc %5f", epoch_acc)

        # 将本轮评估结果加入 val_df（保存每轮测试指标）
        # self.val_df = self.val_df.append(result_dicts_per_monitor, ignore_index=True)
        self.val_df = pd.concat([self.val_df, pd.DataFrame([result_dicts_per_monitor])], ignore_index=True)

        # 确保列顺序和 result_dicts_per_monitor 中的键顺序一致
        self.val_df = self.val_df[list(result_dicts_per_monitor.keys())]

    def train_step_one(self):  # 执行 step one 阶段训练

        logging.info("****** Run until first stop... ********")
        logging.info("Train size: %d", len(self.datasets['train']))
        logging.info("Test size: %d", len(self.datasets['test']))

        epoch = 0
        while epoch < self.step_one_epoch:  # 设置训练终止条件
            self.train_epoch()
            self.monitor_epoch({})
            self.epoch_df = pd.concat([self.train_df, self.val_df], axis=1)

            if epoch % 20 == 0 or epoch > self.step_one_epoch - 50:
                self.log_epoch()
            epoch += 1

    def log_epoch(self):
        # -1 due to doing one monitor at start of training
        i_epoch = len(self.epoch_df) - 1
        logging.info("Epoch {:d}".format(i_epoch))
        last_row = self.epoch_df.iloc[-1]
        for key, val in last_row.items():
            logging.info("%s       %.5f", key, val)
        logging.info("")

    def run(self):
        # 开始第一阶段的训练，直到达到设定的 step_one_epoch 轮数
        self.train_step_one()

        # 创建一个 Measurement 实例，用于对测试集指标进行分析
        self.measure = Measurement(self.epoch_df['test_acc'], classes=self.classes)

        # 判断是分类准确率（acc）还是 AUC，用于兼容 P300 等特殊任务
        if self.epoch_df['test_acc'].max() < 1:  # 如果最大准确率 < 1，说明用的是 AUC（最大值不会是100%）
            # 打印 P300 中使用的 AUC 评估指标
            logging.info('Best test AUC %.5f ', self.measure.max_acc())  # 最大 AUC
            logging.info('mean AUC of last 10 epochs %.5f ', self.measure.last10_acc())  # 最后10轮的平均 AUC
            logging.info('std of last 10 epochs %.5f ', self.measure.last10_std())  # 最后10轮的标准差
            logging.info('offset of max and mean AUC (percentage) %.5f ', self.measure.max_mean_offset())  # 最大值与平均值的差距
        else:
            # 如果是准确率任务，打印分类准确率相关指标
            logging.info('Best test acc %.5f ', self.measure.max_acc())  # 最大准确率
            logging.info('mean acc of last 10 epochs %.5f ', self.measure.last10_acc())  # 最后10轮的平均准确率
            logging.info('std of last 10 epochs %.5f ', self.measure.last10_std())  # 最后10轮的标准差
            logging.info('offset: max SUB mean acc %.5f ', self.measure.max_mean_offset())  # 最大值与平均的偏差

        # 打印分隔线
        logging.info('* ' * 20)

        # 计算综合得分（指标函数），例如：0.4 * max + 0.4 * mean - 0.2 * offset
        logging.info('Index score(0.4*max+0.4*mean-0.2*offset): %.5f', self.measure.index_equation())

        # 保存测试准确率变化图
        self.save_acc_loss_fig()

    def save_acc_loss_fig(self):

        test_acc = self.epoch_df['test_acc'].values.tolist()
        plt.figure()
        plt.plot(range(len(test_acc)), test_acc, label='test acc', linewidth=0.7)  # 画出每一轮的测试准确率
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='lower right')
        plt.savefig(self.fig_path + 'mean{:.3f}max{:.3f}.png'.format(self.measure.last10_acc(), self.measure.max_acc()))


