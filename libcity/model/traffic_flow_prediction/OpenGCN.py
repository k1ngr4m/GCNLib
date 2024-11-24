# OpenGCN.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from scipy.sparse.linalg import eigs

# from ASTGCN import scaled_laplacian, cheb_polynomial, ASTGCNBlock
# from OpenCity import STEncoderBlock, FeedForward, PatchEmbedding_flow, PositionalEncoding



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


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(2)].unsqueeze(1).expand_as(x)


class PatchEmbedding_flow(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, his):
        super(PatchEmbedding_flow, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        # do patching
        x = x.squeeze(-1).permute(0, 2, 1)
        if self.his == x.shape[-1]:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            gap = self.his // x.shape[-1]
            x = x.unfold(dimension=-1, size=self.patch_len // gap, step=self.stride // gap)
            x = F.pad(x, (0, (self.patch_len - self.patch_len // gap)))
        x = self.value_embedding(x)
        x = x + self.position_encoding(x)
        x = x.permute(0, 2, 1, 3)
        return x


class PatchEmbedding_time(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, his):
        super(PatchEmbedding_time, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his
        self.minute_size = 1440 + 1
        self.daytime_embedding = nn.Embedding(self.minute_size, d_model // 2)
        weekday_size = 7 + 1
        self.weekday_embedding = nn.Embedding(weekday_size, d_model // 2)

    def forward(self, x):
        # do patching
        bs, ts, nn, dim = x.size()
        x = x.permute(0, 2, 3, 1).reshape(bs, -1, ts)
        if self.his == x.shape[-1]:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            gap = self.his // x.shape[-1]
            x = x.unfold(dimension=-1, size=self.patch_len // gap, step=self.stride // gap)
        num_patch = x.shape[-2]
        x = x.reshape(bs, nn, dim, num_patch, -1).transpose(1, 3)
        x_tdh = x[:, :, 0, :, 0]
        x_dwh = x[:, :, 1, :, 0]
        x_tdp = x[:, :, 2, :, 0]
        x_dwp = x[:, :, 3, :, 0]

        x_tdh = self.daytime_embedding(x_tdh)
        x_dwh = self.weekday_embedding(x_dwh)
        x_tdp = self.daytime_embedding(x_tdp)
        x_dwp = self.weekday_embedding(x_dwp)
        x_th = torch.cat([x_tdh, x_dwh], dim=-1)
        x_tp = torch.cat([x_tdp, x_dwp], dim=-1)

        return x_th, x_tp


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, prob_drop, alpha):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
        self.mlp = nn.Linear(out_dim, out_dim)
        self.dropout = prob_drop
        self.alpha = alpha

    def forward(self, x, adj):
        d = adj.sum(1)
        h = x
        a = adj / d.view(-1, 1)
        gcn_out = self.fc1(torch.einsum('bdkt,nk->bdnt', h, a))
        out = self.alpha * x + (1 - self.alpha) * gcn_out
        ho = self.mlp(out)
        return ho


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()

        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TemporalSelfAttention(nn.Module):
    def __init__(
            self, dim, t_attn_size, t_num_heads=6, tc_num_heads=6, qkv_bias=False,
            attn_drop=0., proj_drop=0., device=torch.device('cpu'),
    ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.tc_num_heads = tc_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.t_attn_size = t_attn_size

        self.t_q_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_k_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_v_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.norm_tatt1 = LlamaRMSNorm(dim)
        self.norm_tatt2 = LlamaRMSNorm(dim)

        self.tc_q_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.tc_k_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.tc_v_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.tc_attn_drop = nn.Dropout(attn_drop)

        self.GCN = GCN(dim, dim, proj_drop, alpha=0.05)
        self.act = nn.GELU()

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_k, x_v, TH, TP, adj, geo_mask=None, sem_mask=None, trg_mask=False):
        B, T_q, N, D = x_q.shape
        T_k, T_v = x_k.shape[1], x_v.shape[1]

        tc_q = self.tc_q_conv(TP).transpose(1, 2)
        tc_k = self.tc_k_conv(TH).transpose(1, 2)
        tc_v = self.tc_v_conv(x_q).transpose(1, 2)
        tc_q = tc_q.reshape(B, N, T_q, self.tc_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        tc_k = tc_k.reshape(B, N, T_k, self.tc_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        tc_v = tc_v.reshape(B, N, T_v, self.tc_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        tc_attn = (tc_q @ tc_k.transpose(-2, -1)) * self.scale
        if trg_mask:
            ones = torch.ones_like(tc_attn).to(self.device)
            dec_mask = torch.triu(ones, diagonal=1)
            tc_attn = tc_attn.masked_fill(dec_mask == 1, -1e9)
        tc_attn = tc_attn.softmax(dim=-1)
        tc_attn = self.tc_attn_drop(tc_attn)
        tc_x = (tc_attn @ tc_v).transpose(2, 3).reshape(B, N, T_q, D).transpose(1, 2)

        tc_x = self.norm_tatt1(tc_x + x_q)

        t_q = self.t_q_conv(tc_x).transpose(1, 2)
        t_k = self.t_k_conv(tc_x).transpose(1, 2)
        t_v = self.t_v_conv(tc_x).transpose(1, 2)
        t_q = t_q.reshape(B, N, T_q, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T_k, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T_v, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        if trg_mask:
            ones = torch.ones_like(t_attn).to(self.device)
            dec_mask = torch.triu(ones, diagonal=1)
            t_attn = t_attn.masked_fill(dec_mask == 1, -1e9)
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T_q, D).transpose(1, 2)

        t_x = self.norm_tatt2(t_x + tc_x)
        gcn_out = self.GCN(t_x, adj)
        x = self.proj_drop(gcn_out)
        return x


class STEncoderBlock(nn.Module):
    def __init__(
            self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=4, tc_num_heads=4, t_num_heads=4,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, device=torch.device('cpu'), type_ln="pre", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = LlamaRMSNorm(dim)
        self.norm2 = LlamaRMSNorm(dim)
        self.st_attn = TemporalSelfAttention(dim, t_attn_size, t_num_heads=t_num_heads, tc_num_heads=tc_num_heads,
                                             qkv_bias=qkv_bias,
                                             attn_drop=attn_drop, proj_drop=drop, device=device)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(hidden_size=dim, intermediate_size=mlp_hidden_dim)

    def forward(self, x, dec_in, enc_out, TH, TP, adj, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':
            x_nor1 = self.norm1(x)
            x = x + self.drop_path(
                self.st_attn(x_nor1, x_nor1, x_nor1, TH, TP, adj, geo_mask=geo_mask, sem_mask=sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(
                (x + self.drop_path(self.st_attn(x, x, x, TH, TP, adj, geo_mask=geo_mask, sem_mask=sem_mask))))
            x = self.norm2((x + self.drop_path(self.mlp(x))))
        else:
            x = x + self.drop_path(self.st_attn(x, x, x, TH, TP, adj, geo_mask=geo_mask, sem_mask=sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x





class OpenGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # ASTGCN相关参数
        self.K = config.get('K', 3)  # 切比雪夫多项式的阶数
        self.nb_chev_filter = config.get('nb_chev_filter', 64)  # 切比雪夫滤波器的数量
        self.nb_time_filter = config.get('nb_time_filter', 64)  # 时间滤波器的数量

        # OpenCity相关参数
        self.hidden_size = config.get('hidden_size', 64)  # 隐藏层尺寸
        self.num_layers = config.get('num_layers', 1)  # 网络层数
        self.dropout = config.get('dropout', 0)  # Dropout率

        # 数据特征参数
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)

        # 模型结构
        self.astgcn_layers = self.build_astgcn_layers()
        self.opencity_layers = self.build_opencity_layers()
        self.fusion_layer = self.build_fusion_layer()

    def build_astgcn_layers(self):
        # 构建ASTGCN层
        device = self.device
        adj_mx = self.data_feature.get('adj_mx')
        l_tilde = scaled_laplacian(adj_mx)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in
                            cheb_polynomial(l_tilde, self.K)]

        self.astgcn_blocks = nn.ModuleList([
            ASTGCNBlock(device, self.feature_dim, self.K, self.nb_chev_filter, self.nb_time_filter, 1, cheb_polynomials,
                        self.num_nodes, 1)
            for _ in range(self.nb_block)
        ])
        return self.astgcn_blocks

    def build_opencity_layers(self):
        # 构建OpenCity层
        self.patch_embedding = PatchEmbedding_flow(self.hidden_size, patch_len=3, stride=1, padding=0,
                                                   his=self.input_window)
        self.positional_encoding = PositionalEncoding(self.hidden_size)
        self.st_encoder_blocks = nn.ModuleList([
            STEncoderBlock(self.hidden_size, s_attn_size=7, t_attn_size=7, geo_num_heads=4, sem_num_heads=4,
                           tc_num_heads=4, t_num_heads=4)
            for _ in range(self.num_layers)
        ])
        self.feed_forward = FeedForward(self.hidden_size, intermediate_size=self.hidden_size * 4)
        return self.st_encoder_blocks

    def build_fusion_layer(self):
        # 构建融合层
        fusion_layer = nn.Linear(self.nb_time_filter + self.hidden_size, self.output_dim)
        return fusion_layer

    def forward(self, batch):
        # 获取输入数据
        x = batch['X']

        # ASTGCN前向传播
        astgcn_output = self.forward_astgcn(x)

        # OpenCity前向传播
        opencity_output = self.forward_opencity(x)

        # 融合两个模型的输出
        fused_output = torch.cat((astgcn_output, opencity_output), dim=-1)
        output = self.fusion_layer(fused_output)

        return output

    def forward_astgcn(self, x):
        for block in self.astgcn_blocks:
            x = block(x)
        return x

    def forward_opencity(self, x):
        x = self.patch_embedding(x)
        x = x + self.positional_encoding(x)
        for block in self.st_encoder_blocks:
            x = block(x, None, None, None, None)
        x = self.feed_forward(x)
        return x

    def calculate_loss(self, batch):
        # 计算损失函数
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
