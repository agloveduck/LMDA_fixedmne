from torchsummary import summary
import torch
import torch.nn as nn


class EEGDepthAttention(nn.Module):
    """
    Build EEG Depth Attention module.
    :arg
    C: num of channels
    W: num of time samples
    k: learnable kernel size
    """
    """
        EEG信号的深度注意力模块。通过自适应平均池化 + 1D卷积 + Softmax 来给通道加权。
        """
    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        # 自适应池化，把 channel × time 特征降成 1 × W，聚焦时间维度
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        # 卷积核尺寸为 (k × 1)，对 channel 维进行注意力建模
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        # softmax 激活，用于归一化注意力权重
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        """
        前向传播
        输入：x shape = [batch, channels, 1, time]
        返回：与原始特征相乘后的加权输出
        """
        x_pool = self.adaptive_pool(x)  # shape: [B, C, 1, W]
        x_transpose = x_pool.transpose(-2, -3)  # shape: [B, C, W, 1]
        y = self.conv(x_transpose)  # 卷积后 shape: [B, 1, W, 1]
        y = self.softmax(y)  # 归一化注意力权重
        y = y.transpose(-2, -3)  # shape: [B, 1, 1, W]
        return y * self.C * x  # 注意力加权（可缩放）


class LMDA(nn.Module):
    """
    LMDA-Net for the paper  输入格式：[batch, 1, channels, samples]
    """

    def __init__(self,
                 chans=22,  # EEG 通道数（如22个导联）
                 samples=1125,  # 每段 EEG 样本点（时间长度）
                 num_classes=4,  # 分类类别数（如运动想象任务通常为4类）
                 depth=9,  # 深度权重维度（等价于 attention head 数）
                 kernel=75,  # 时间卷积核长度
                 channel_depth1=24,  # 时间卷积后的通道数
                 channel_depth2=9,  # 空间卷积后的通道数
                 ave_depth=1,
                 avepool=5):  # 时间维上的平均池化大小
        super(LMDA, self).__init__()
        self.ave_depth = ave_depth
        # 可学习导联权重矩阵 表示：多个注意力头对 EEG 导联通道的权重学习；作用：在特征提取前，对输入的 EEG 导联进行学习性加权（形状为 [depth, 1, chans]），用于增强通道选择能力。
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)
        # nn.init.kaiming_normal_(self.channel_weight.data, nonlinearity='relu')
        # nn.init.normal_(self.channel_weight.data)
        # nn.init.constant_(self.channel_weight.data, val=1/chans)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        # self.avgPool1 = nn.AvgPool2d((1, 24))
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            # nn.AdaptiveAvgPool3d((9, 1, 35)),
            nn.Dropout(p=0.65),
        )

        # 定义自动填充模块
        out = torch.ones((1, 1, chans, samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        # out = self.avgPool1(out)
        N, C, H, W = out.size()

        self.depthAttention = EEGDepthAttention(W, C, k=7)

        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape
        print('In ShallowNet, n_out_time shape: ', n_out_time)
        self.classifier = nn.Linear(n_out_time[-1]*n_out_time[-2]*n_out_time[-3], num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)  # 导联权重筛选

        x_time = self.time_conv(x)  # batch, depth1, channel, samples_
        x_time = self.depthAttention(x_time)  # DA1

        x = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        x = self.norm(x)

        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        return cls


if __name__ == '__main__':
    model = LMDA(num_classes=2, chans=3, samples=875, channel_depth1=24, channel_depth2=7).cuda()
    a = torch.randn(12, 1, 3, 875).cuda().float()
    l2 = model(a)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    summary(model, show_input=True)
    print(l2.shape)

