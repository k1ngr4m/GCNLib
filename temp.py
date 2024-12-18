import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeAttention(nn.Module):
    def __init__(self, N, F_in, T):
        super(TimeAttention, self).__init__()
        self.U1 = nn.Parameter(torch.randn(F_in, N))
        self.U2 = nn.Parameter(torch.randn(N, T))
        self.U3 = nn.Parameter(torch.randn(T, F_in))
        self.Ve = nn.Parameter(torch.randn(T, 1))
        self.be = nn.Parameter(torch.randn(T))

    def forward(self, x):
        """
        前向传播方法，计算时间注意力分数

        参数:
            x (torch.tensor): 输入特征，形状为 (batch_size, N, F_in, T)

        返回:
            torch.tensor: 时间注意力分数，形状为 (B, T, T)
        """
        batch_size, N, F_in, T = x.shape

        # 计算左侧特征，x的维度从(B, N, F_in, T)变为(B, F_in, N, T)，再变为(B, T, N)
        lhs = torch.matmul(torch.matmul(x.permute(0, 2, 1, 3).reshape(batch_size, F_in, -1), self.U1), self.U2)

        # 计算右侧特征，x的维度从(B, N, F_in, T)变为(B, N, T)
        rhs = torch.matmul(self.U3, x.reshape(batch_size, N * F_in, T)).permute(0, 2, 1)

        # 计算特征乘积，得到时间注意力分数的原始值
        product = torch.bmm(lhs, rhs)

        # 应用sigmoid激活函数并计算最终的时间注意力分数
        e = torch.matmul(torch.sigmoid(product + self.be.unsqueeze(0)), self.Ve).squeeze(-1)

        # 对时间注意力分数进行归一化
        e_normalized = F.softmax(e, dim=1)

        return e_normalized  # 返回时间注意力分数


# 示例使用
if __name__ == "__main__":
    batch_size, N, F_in, T = 4, 32, 64, 12
    x = torch.randn(batch_size, N, F_in, T)
    model = TimeAttention(N, F_in, T)
    output = model(x)
    print(output.shape)  # 输出应为 (batch_size, T, T)