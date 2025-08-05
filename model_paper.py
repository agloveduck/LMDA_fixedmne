from torchsummary import summary
import torch
import torch.nn as nn

class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    输入格式：[batch, 1, channels, samples]
    """

    def __init__(self, chans=22, samples=1125, num_classes=4, depth=9, kernel=75,
                 channel_depth1=24, channel_depth2=9, ave_depth=1, avepool=5):
        super(LMDA, self).__init__()
        self.ave_depth = ave_depth

        # 可学习的通道权重参数：[depth, 1, chans]
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)  # Xavier 初始化

        # 时间卷积模块（类似时域特征提取）
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),  # 升维
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel), groups=channel_depth1, bias=False),  # 深度卷积（每个通道单独卷积）
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),  # GELU激活
        )

        # 通道卷积模块（空间域特征提取）
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),  # 降维
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),  # 深度卷积，跨通道
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        # 池化 + dropout（用于归一化和防过拟合）
        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        # 预先推理一次网络，用于确定最后全连接层的输入维度
        out = torch.ones((1, 1, chans, samples))  # 模拟输入
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)  # 应用通道加权
        out = self.time_conv(out)
        out = self.chanel_conv(out)
        out = self.norm(out)

        # 得到输出维度（[batch, channel, 1, time]）
        n_out_time = out.cpu().data.numpy().shape
        print('In ShallowNet, n_out_time shape: ', n_out_time)

        # 构建最终分类器，全连接层输入维度 = 输出张量展平后维度
        self.classifier = nn.Linear(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes)

    def EEGDepthAttention(self, x):
        """
        通道注意力机制（Depth Attention）
        x: [batch, channels, height, width]
        返回值：对x进行加权的结果
        """

        N, C, H, W = x.size()
        k = 7  # 卷积核大小
        adaptive_pool = nn.AdaptiveAvgPool2d((1, W))  # 平均池化，压缩空间维度
        conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k//2, 0), bias=True).to(x.device)  # 注意力生成卷积
        softmax = nn.Softmax(dim=-2)  # 沿 height 方向归一化

        x_pool = adaptive_pool(x)  # -> [N, C, 1, W]
        x_transpose = x_pool.transpose(-2, -3)  # -> [N, C, W, 1]

        y = conv(x_transpose)  # 卷积生成注意力分布
        y = softmax(y)  # softmax 归一化
        y = y.transpose(-2, -3)  # 转回原形状 [N, C, 1, W]

        return y * C * x  # 将注意力权重应用于原始特征

    def forward(self, x):
        """
        前向传播过程
        输入：x [batch, 1, chans, samples]
        """

        # 通道加权：将通道维 (chans) 与 channel_weight 相乘
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)

        # 时间卷积（提取时间特征）
        x_time = self.time_conv(x)

        # 深度注意力机制
        x_time = self.EEGDepthAttention(x_time)

        # 空间卷积（提取通道特征）
        x = self.chanel_conv(x_time)

        # 池化 + dropout
        x = self.norm(x)

        # 展平特征图
        features = torch.flatten(x, 1)

        # 全连接分类
        cls = self.classifier(features)

        return cls
