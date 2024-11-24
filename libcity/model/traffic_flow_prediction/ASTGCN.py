import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from scipy.sparse.linalg import eigs


# 定义计算缩放拉普拉斯矩阵的函数
def scaled_laplacian(weight):
    """
    计算缩放拉普拉斯矩阵 ~L
    L = D - A
    ~L = 2L/lambda_max - I

    参数:
        weight(np.ndarray): 邻接矩阵，形状为 (N, N)，N 是顶点的数量

    返回:
        np.ndarray: 缩放拉普拉斯矩阵 ~L，形状为 (N, N)
    """
    assert weight.shape[0] == weight.shape[1]  # 确保邻接矩阵是方阵
    n = weight.shape[0]  # 获取顶点数量
    diag = np.diag(np.sum(weight, axis=1))  # 计算度矩阵
    lap = diag - weight  # 计算拉普拉斯矩阵
    for i in range(n):
        for j in range(n):
            if diag[i, i] > 0 and diag[j, j] > 0:
                lap[i, j] /= np.sqrt(diag[i, i] * diag[j, j])  # 对拉普拉斯矩阵进行归一化
    lambda_max = eigs(lap, k=1, which='LR')[0].real  # 计算最大特征值
    return (2 * lap) / lambda_max - np.identity(weight.shape[0])  # 返回缩放拉普拉斯矩阵


# 定义计算切比雪夫多项式的函数
def cheb_polynomial(l_tilde, k):
    """
    计算从 T_0 到 T_{K-1} 的一系列切比雪夫多项式

    参数:
        l_tilde(np.ndarray): 缩放拉普拉斯矩阵，形状为 (N, N)
        k(int): 切比雪夫多项式的最大阶数

    返回:
        list(np.ndarray): 切比雪夫多项式列表，长度为 K，从 T_0 到 T_{K-1}
    """
    num = l_tilde.shape[0]  # 获取顶点数量
    cheb_polynomials = [np.identity(num), l_tilde.copy()]  # 初始化多项式列表
    for i in range(2, k):
        cheb_polynomials.append(np.matmul(2 * l_tilde, cheb_polynomials[i - 1]) - cheb_polynomials[i - 2])  # 递归计算多项式
    return cheb_polynomials  # 返回多项式列表


# 定义空间注意力层类
class SpatialAttentionLayer(nn.Module):
    """
    计算空间注意力分数
    """

    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(SpatialAttentionLayer, self).__init__()  # 初始化父类
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(device))  # 时间权重参数
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(device))  # 特征权重参数
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))  # 特征权重参数
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(device))  # 空间偏置参数
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(device))  # 空间变换参数

    def forward(self, x):
        """
        前向传播

        参数:
            x(torch.tensor): 输入特征，形状为 (B, N, F_in, T)

        返回:
            torch.tensor: 空间注意力分数，形状为 (B, N, N)
        """
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # 计算左侧特征
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # 计算右侧特征并转置
        product = torch.matmul(lhs, rhs)  # 计算特征乘积
        s = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # 应用激活函数并计算空间注意力
        s_normalized = F.softmax(s, dim=1)  # 归一化空间注意力分数
        return s_normalized  # 返回空间注意力分数


# 定义带空间注意力的切比雪夫图卷积类
class ChebConvWithSAt(nn.Module):
    """
    K阶切比雪夫图卷积
    """

    def __init__(self, k, cheb_polynomials, in_channels, out_channels):
        """
        初始化切比雪夫图卷积层

        参数:
            k(int): 阶数
            cheb_polynomials: 切比雪夫多项式列表
            in_channels(int): 输入通道数
            out_channels(int): 输出通道数
        """
        super(ChebConvWithSAt, self).__init__()  # 初始化父类
        self.K = k  # 阶数
        self.cheb_polynomials = cheb_polynomials  # 切比雪夫多项式列表
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.DEVICE = cheb_polynomials[0].device  # 设备
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(k)])  # 权重参数列表

    def forward(self, x, spatial_attention):
        """
        前向传播

        参数:
            x: 输入特征，形状为 (batch_size, N, F_in, T)
            spatial_attention: 空间注意力分数，形状为 (batch_size, N, N)

        返回:
            torch.tensor: 输出特征，形状为 (batch_size, N, F_out, T)
        """
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape  # 获取输入特征形状
        outputs = []  # 初始化输出列表
        for time_step in range(num_of_timesteps):  # 遍历时间步
            graph_signal = x[:, :, :, time_step]  # 获取时间步的特征
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # 初始化输出特征
            for k in range(self.K):  # 遍历多项式阶数
                t_k = self.cheb_polynomials[k]  # 获取切比雪夫多项式
                t_k_with_at = t_k.mul(spatial_attention)  # 应用空间注意力
                theta_k = self.Theta[k]  # 获取权重参数
                rhs = t_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # 计算右侧特征
                output = output + rhs.matmul(theta_k)  # 计算输出特征
            outputs.append(output.unsqueeze(-1))  # 添加输出特征
        return F.relu(torch.cat(outputs, dim=-1))  # 返回激活后的输出特征


class TemporalAttentionLayer(nn.Module):
    # 时间注意力层类，用于计算时间注意力分数
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(TemporalAttentionLayer, self).__init__()
        # 初始化参数
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(device))  # 时间注意力参数U1
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(device))  # 时间注意力参数U2
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))  # 时间注意力参数U3
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(device))  # 时间注意力偏置be
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(device))  # 时间注意力变换Ve

    def forward(self, x):
        """
        前向传播方法，计算时间注意力分数

        参数:
            x (torch.tensor): 输入特征，形状为 (batch_size, N, F_in, T)

        返回:
            torch.tensor: 时间注意力分数，形状为 (B, T, T)
        """
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # 计算左侧特征，x的维度从(B, N, F_in, T)变为(B, T, F_in, N)，再变为(B, T, N)
        rhs = torch.matmul(self.U3, x)  # 计算右侧特征，x的维度从(B, N, F_in, T)变为(B, N, T)
        product = torch.matmul(lhs, rhs)  # 计算特征乘积，得到时间注意力分数的原始值
        e = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # 应用sigmoid激活函数并计算最终的时间注意力分数
        e_normalized = F.softmax(e, dim=1)  # 对时间注意力分数进行归一化
        return e_normalized  # 返回时间注意力分数


class ASTGCNBlock(nn.Module):
    # ASTGCN块类，结合时间和空间注意力的图卷积网络块
    def __init__(self, device, in_channels, k, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(ASTGCNBlock, self).__init__()
        self.TAt = TemporalAttentionLayer(device, in_channels, num_of_vertices, num_of_timesteps)  # 时间注意力层
        self.SAt = SpatialAttentionLayer(device, in_channels, num_of_vertices, num_of_timesteps)  # 空间注意力层
        self.cheb_conv_SAt = ChebConvWithSAt(k, cheb_polynomials, in_channels, nb_chev_filter)  # 带空间注意力的切比雪夫图卷积
        # 时间卷积: 输入时间长度 = num_of_timesteps = time_strides * output_window
        # 输入必须是输出output_window的固定倍数！
        # ker=3, pad=2, stride=time_strides
        # 输出时间长度 = (time_strides * output_window + 2 * pad - ker) / time_strides + 1 = output_window
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3),
                                   stride=(1, time_strides), padding=(0, 1))
        # 时间维度上卷积: 输入时间长度 = num_of_timesteps = time_strides * output_window
        # ker=1, stride=time_strides
        # 输出时间长度 = (time_strides * output_window - ker) / time_strides + 1 = output_window
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1),
                                       stride=(1, time_strides))  # 残差卷积
        self.ln = nn.LayerNorm(nb_time_filter)  # 需要将channel放到最后一个维度上

    def forward(self, x):
        """
        前向传播方法

        参数:
            x (torch.tensor): 输入特征，形状为 (batch_size, N, F_in, T)

        返回:
            torch.tensor: 输出特征，形状为 (batch_size, N, nb_time_filter, output_window)
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # TAt
        temporal_at = self.TAt(x)  # (B, T, T)

        x_tat = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_at) \
            .reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        # 结合时间注意力：(B, N*F_in, T) * (B, T, T) -> (B, N*F_in, T) -> (B, N, F_in, T)

        # SAt
        spatial_at = self.SAt(x_tat)  # (B, N, N)

        # 结合空间注意力的图卷积 cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_at)  # (B, N, F_out, T), F_out = nb_chev_filter

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        # (B, N, F_out, T) -> (B, F_out, N, T) 用(1,3)的卷积核去做->(B, F_out', N, T') F_out'=nb_time_filter

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        # (B, N, F_in, T) -> (B, F_in, N, T) 用(1,1)的卷积核去做->(B, F_out', N, T') F_out'=nb_time_filter

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (B, F_out', N, T') -> (B, T', N, F_out') -ln -> (B, T', N, F_out') -> (B, N, F_out', T')

        return x_residual


class FusionLayer(nn.Module):
    # 融合层类，用于特征融合
    def __init__(self, n, h, w, device):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(1, n, h, w).to(device))  # 可训练的融合权重参数

    def forward(self, x):
        # 前向传播方法，进行特征融合
        x = x * self.weights  # 元素级乘法融合特征
        return x  # 返回融合后的特征


class ASTGCNSubmodule(nn.Module):
    # ASTGCN子模块类，包含多个ASTGCN块
    def __init__(self, device, nb_block, in_channels, k, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, output_window, output_dim, num_of_vertices):
        super(ASTGCNSubmodule, self).__init__()

        self.BlockList = nn.ModuleList([ASTGCNBlock(device, in_channels, k, nb_chev_filter,
                                                    nb_time_filter, time_strides, cheb_polynomials,
                                                    num_of_vertices, time_strides * output_window)])

        self.BlockList.extend([ASTGCNBlock(device, nb_time_filter, k, nb_chev_filter,
                                           nb_time_filter, 1, cheb_polynomials,
                                           num_of_vertices, output_window)
                               for _ in range(nb_block - 1)])

        self.final_conv = nn.Conv2d(output_window, output_window,
                                    kernel_size=(1, nb_time_filter - output_dim + 1))  # 最终卷积
        self.fusionlayer = FusionLayer(output_window, num_of_vertices, output_dim, device)  # 融合层

    def forward(self, x):
        """
        前向传播方法

        参数:
            x (torch.tensor): 输入特征，形状为 (B, T_in, N_nodes, F_in)

        返回:
            torch.tensor: 输出特征，形状为 (B, T_out, N_nodes, out_dim)
        """
        x = x.permute(0, 2, 3, 1)  # 调整输入特征的维度
        for block in self.BlockList:
            x = block(x)  # 通过每个ASTGCN块
        output = self.final_conv(x.permute(0, 3, 1, 2))  # 最终卷积
        output = self.fusionlayer(output)  # 融合层
        return output  # 返回输出特征


class ASTGCN(AbstractTrafficStateModel):
    # ASTGCN类，继承自AbstractTrafficStateModel，用于交通状态预测
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)  # 调用父类的构造函数

        # 从data_feature中获取图的节点数，默认为1
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        # 从data_feature中获取特征维度，默认为1
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        # 从data_feature中获取周期长度，默认为0
        self.len_period = self.data_feature.get('len_period', 0)
        # 从data_feature中获取趋势长度，默认为0
        self.len_trend = self.data_feature.get('len_trend', 0)
        # 从data_feature中获取接近度长度，默认为0
        self.len_closeness = self.data_feature.get('len_closeness', 0)
        # 确保至少有一个时间维度长度不为零
        if self.len_period == 0 and self.len_trend == 0 and self.len_closeness == 0:
            raise ValueError('Num of days/weeks/hours are all zero! Set at least one of them not zero!')
        # 从data_feature中获取输出维度，默认为1
        self.output_dim = self.data_feature.get('output_dim', 1)

        # 从config中获取输出窗口大小，默认为1
        self.output_window = config.get('output_window', 1)
        # 从config中获取设备，默认为cpu
        self.device = config.get('device', torch.device('cpu'))
        # 从config中获取块的数量，默认为2
        self.nb_block = config.get('nb_block', 2)
        # 从config中获取切比雪夫多项式的阶数，默认为3
        self.K = config.get('K', 3)
        # 从config中获取切比雪夫滤波器的数量，默认为64
        self.nb_chev_filter = config.get('nb_chev_filter', 64)
        # 从config中获取时间滤波器的数量，默认为64
        self.nb_time_filter = config.get('nb_time_filter', 64)

        # 从data_feature中获取邻接矩阵
        adj_mx = self.data_feature.get('adj_mx')
        # 计算缩放拉普拉斯矩阵
        l_tilde = scaled_laplacian(adj_mx)
        # 计算切比雪夫多项式
        self.cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device)
                                 for i in cheb_polynomial(l_tilde, self.K)]
        # 获取日志记录器
        self._logger = getLogger()
        # 从data_feature中获取归一化缩放器
        self._scaler = self.data_feature.get('scaler')

        # 根据时间维度长度创建ASTGCN子模块
        if self.len_closeness > 0:
            self.hours_ASTGCN_submodule = \
                ASTGCNSubmodule(self.device, self.nb_block, self.feature_dim,
                                self.K, self.nb_chev_filter, self.nb_time_filter,
                                self.len_closeness // self.output_window, self.cheb_polynomials,
                                self.output_window, self.output_dim, self.num_nodes)
        if self.len_period > 0:
            self.days_ASTGCN_submodule = \
                ASTGCNSubmodule(self.device, self.nb_block, self.feature_dim,
                                self.K, self.nb_chev_filter, self.nb_time_filter,
                                self.len_period // self.output_window, self.cheb_polynomials,
                                self.output_window, self.output_dim, self.num_nodes)
        if self.len_trend > 0:
            self.weeks_ASTGCN_submodule = \
                ASTGCNSubmodule(self.device, self.nb_block, self.feature_dim,
                                self.K, self.nb_chev_filter, self.nb_time_filter,
                                self.len_trend // self.output_window, self.cheb_polynomials,
                                self.output_window, self.output_dim, self.num_nodes)
        self._init_parameters()  # 初始化模型参数

    def _init_parameters(self):
        # 初始化模型参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 对多维参数使用Xavier均匀初始化
            else:
                nn.init.uniform_(p)  # 对一维参数使用均匀初始化

    def forward(self, batch):
        # 前向传播函数
        x = batch['X']  # 获取输入数据 (B, Tw+Td+Th, N_nodes, F_in)
        output = 0  # 初始化输出
        # 根据时间维度长度处理不同的输入部分
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            output_hours = self.hours_ASTGCN_submodule(x[:, begin_index:end_index, :, :])
            output += output_hours
        if self.len_period > 0:
            begin_index = self.len_closeness
            end_index = begin_index + self.len_period
            output_days = self.days_ASTGCN_submodule(x[:, begin_index:end_index, :, :])
            output += output_days
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            end_index = begin_index + self.len_trend
            output_weeks = self.weeks_ASTGCN_submodule(x[:, begin_index:end_index, :, :])
            output += output_weeks
        return output  # 返回输出 (B, Tp, N_nodes, F_out)

    def calculate_loss(self, batch):
        # 计算损失函数
        y_true = batch['y']  # 真实值
        y_predicted = self.predict(batch)  # 预测值
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])  # 逆变换真实值
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])  # 逆变换预测值
        return loss.masked_mse_torch(y_predicted, y_true)  # 计算masked MSE损失

    def predict(self, batch):
        # 预测函数
        return self.forward(batch)  # 调用前向传播函数进行预测
