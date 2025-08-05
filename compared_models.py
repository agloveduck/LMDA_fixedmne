from torchsummary import summary
import torch
import torch.nn as nn


# 模型权重初始化函数，对不同模块进行初始化 通用初始化方法，提高训练稳定性
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)  # 用 Xavier 均匀分布初始化
        # nn.init.constant(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):  # 权重设为 1，偏置设为 0
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # 用 Xavier 均匀分布初始化
        nn.init.constant_(m.bias, 0)


class MaxNormDefaultConstraint(object):  # 对网络的每一层施加 最大 L2 范数约束（限制权重的最大范数），提高泛化能力。
    """
    Applies max L2 norm 2 to the weights until the final layer and L2 norm 0.5
    to the weights of the final layer as done in [1]_.中间层权重限制在 max norm = 2 最后一层限制为 max norm = 0.5

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730

    """

    def apply(self, model):
        last_weight = None
        for name, module in list(model.named_children()):
            if hasattr(module, "weight") and (
                    not module.__class__.__name__.startswith("BatchNorm")
            ):
                module.weight.data = torch.renorm(
                    module.weight.data, 2, 0, maxnorm=2
                )
                last_weight = module.weight
        if last_weight is not None:
            last_weight.data = torch.renorm(last_weight.data, 2, 0, maxnorm=0.5)


class SeparableConv2D(nn.Module):  # 这是一个 Depthwise + Pointwise 卷积模块，适用于轻量级 EEGNet 中的模块设计
    # Depthwise Conv -> ELU -> Pointwise Conv -> BN -> ELU
    def __init__(self, in_channels, out_channels, kernel1_size, **kw):
        super(SeparableConv2D, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel1_size, **kw),
            # nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            # pw
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), **kw),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.depth_conv(x)


def square_activation(x):  # 返回输入的平方
    return torch.square(x)


def safe_log(x):  # 避免出现 log(0) 的数值不稳定问题（clip）
    return torch.clip(torch.log(x), min=1e-7, max=1e7)


class ShallowConvNet(nn.Module):  # 输入 [batch, 1, chans, samples]
    '''
          Conv2d(1, 40, (1, 25))              # 时间卷积
          Conv2d(40, 40, (chans, 1))         # 空间卷积（通道融合）
          BatchNorm2d(40)
          square_activation
          AvgPool2d((1, 75), stride=(1, 15))  # 时间压缩
          safe_log
          Dropout
          Linear -> num_classes

    '''

    def __init__(self, num_classes, chans, samples=1125):
        super(ShallowConvNet, self).__init__()
        self.conv_nums = 40
        self.features = nn.Sequential(
            nn.Conv2d(1, self.conv_nums, (1, 25)),
            nn.Conv2d(self.conv_nums, self.conv_nums, (chans, 1), bias=False),
            nn.BatchNorm2d(self.conv_nums)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout()
        out = torch.ones((1, 1, chans, samples))
        out = self.features(out)
        out = self.avgpool(out)
        n_out_time = out.cpu().data.numpy().shape
        self.classifier = nn.Linear(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = square_activation(x)
        x = self.avgpool(x)
        x = safe_log(x)
        x = self.dropout(x)
        features = torch.flatten(x, 1)  # 使用卷积网络代替全连接层进行分类, 因此需要返回x和卷积层个数
        cls = self.classifier(features)
        return cls


class EEGNet(nn.Module):
    '''
        Conv2d(1, F1, (1, kernel_length))         # 时间卷积
        Depthwise Conv: Conv2d(F1, F1, (chans, 1), groups=F1)
        BatchNorm + ELU
        AvgPool2d((1, 4))
        Dropout

        Depthwise Conv: Conv2d(F1, F1, (1, 16), groups=F1)
        Pointwise Conv: Conv2d(F1, F2, (1, 1))
        BatchNorm + ELU
        AvgPool2d((1, 8))
        Dropout

        Linear -> num_classes

    '''

    def __init__(self, num_classes, chans, samples=1125, dropout_rate=0.5, kernel_length=64, F1=8,
                 F2=16, ):
        super(EEGNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1, kernel_size=(chans, 1), groups=F1, bias=False),  # groups=F1 for depthWiseConv
            nn.BatchNorm2d(F1),
            nn.ELU(inplace=True),
            # nn.ReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
            # for SeparableCon2D
            # SeparableConv2D(F1, F2, kernel1_size=(1, 16), bias=False),
            nn.Conv2d(F1, F1, kernel_size=(1, 16), groups=F1, bias=False),  # groups=F1 for depthWiseConv
            nn.BatchNorm2d(F1),
            nn.ELU(inplace=True),
            # nn.ReLU(),
            nn.Conv2d(F1, F2, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn
            nn.BatchNorm2d(F2),
            # nn.ReLU(),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=dropout_rate),
        )
        out = torch.ones((1, 1, chans, samples))
        out = self.features(out)
        n_out_time = out.cpu().data.numpy().shape
        self.classifier = nn.Linear(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes)

    def forward(self, x):
        conv_features = self.features(x)
        features = torch.flatten(conv_features, 1)
        cls = self.classifier(features)
        return cls


if __name__ == '__main__':
    model = ShallowConvNet(num_classes=4, chans=22, samples=1125).cuda()
    a = torch.randn(12, 1, 3, 875).cuda().float()
    l2 = model(a)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    summary(model, show_input=True)

    print(l2.shape)

