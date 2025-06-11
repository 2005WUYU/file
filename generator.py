"""
GAT-UNet生成器模型实现
- 编码器：GAT + 池化
- 解码器：GAT + 上采样
- 跳跃连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TopKPooling
from typing import List, Tuple, Optional
import logging


class GATConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 heads: int = 4,
                 dropout: float = 0.2):
        """
        GAT卷积块
        Args:
            in_channels: 输入特征维度
            out_channels: 每个头的输出维度（总输出维度将是 out_channels * heads）
            heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels // heads,  # 注意这里除以heads
            heads=heads,
            dropout=dropout,
            concat=True
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.gat(x, edge_index, edge_attr)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pool_ratio: float = 0.5,
                 heads: int = 4):
        """
        编码器块：GAT + 池化
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            pool_ratio: 池化保留节点比例
            heads: GAT注意力头数
        """
        super().__init__()
        self.conv = GATConvBlock(in_channels, out_channels, heads)
        self.pool = TopKPooling(out_channels, ratio=pool_ratio)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # GAT卷积
        x = self.conv(x, edge_index, edge_attr)

        # 池化
        x, edge_index, edge_attr, batch, perm, score = self.pool(
            x, edge_index, edge_attr, batch
        )

        return x, edge_index, edge_attr, batch, perm, score


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 heads: int = 4):
        """
        解码器块：上采样 + GAT
        Args:
            in_channels: 上采样特征的通道数
            skip_channels: 跳跃连接特征的通道数
            out_channels: 输出通道数（必须是heads的倍数）
            heads: GAT注意力头数
        """
        super().__init__()
        total_in_channels = in_channels + skip_channels
        self.conv = GATConvBlock(total_in_channels, out_channels, heads)

    def forward(self, x, x_skip, edge_index, edge_attr=None):
        x = torch.cat([x, x_skip], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class Generator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int] = [32, 64, 128, 256],
                 pool_ratios: List[float] = [0.8, 0.6, 0.4],
                 heads: int = 4):
        """
        GAT-UNet生成器
        Args:
            in_channels: 输入特征维度
            hidden_channels: 各层隐藏特征维度
            pool_ratios: 各层池化比例
            heads: GAT注意力头数
        """
        super().__init__()

        # 确保所有通道数都是heads的倍数
        self.hidden_channels = [
            (h + heads - 1) // heads * heads for h in hidden_channels
        ]

        # 确保维度匹配
        assert len(pool_ratios) == len(self.hidden_channels) - 1

        # 编码器层
        self.encoder_blocks = nn.ModuleList()
        curr_channels = in_channels
        for h_dim, p_ratio in zip(self.hidden_channels[:-1], pool_ratios):
            self.encoder_blocks.append(
                EncoderBlock(curr_channels, h_dim, p_ratio, heads)
            )
            curr_channels = h_dim

        # 瓶颈层
        self.bottleneck = GATConvBlock(
            self.hidden_channels[-2],
            self.hidden_channels[-1],
            heads
        )

        # 解码器层
        self.decoder_blocks = nn.ModuleList()
        reversed_channels = list(reversed(self.hidden_channels))
        skip_channels = list(reversed(self.hidden_channels[:-1]))

        for i in range(len(self.hidden_channels) - 1):
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=reversed_channels[i],
                    skip_channels=skip_channels[i],
                    out_channels=reversed_channels[i+1],
                    heads=heads
                )
            )

        # 输出层
        # 确保最终输出通道数正确
        self.final_out_channels = (in_channels + heads - 1) // heads * heads
        self.output_layer = GATConv(
            self.hidden_channels[0],
            self.final_out_channels // heads,
            heads=heads,
            concat=True
        )

    def encode(self, x, edge_index, edge_attr=None, batch=None):
        """编码过程"""
        encoded_features = []
        curr_x = x
        curr_edge_index = edge_index
        curr_edge_attr = edge_attr
        curr_batch = batch

        for encoder in self.encoder_blocks:
            # 保存特征用于跳跃连接
            encoded_features.append((curr_x, curr_edge_index, curr_edge_attr))

            # 编码器前向传播
            curr_x, curr_edge_index, curr_edge_attr, curr_batch, perm, _ = encoder(
                curr_x, curr_edge_index, curr_edge_attr, curr_batch
            )

            # 保存perm用于unpooling
            encoded_features[-1] = (*encoded_features[-1], perm)

        # 瓶颈层
        bottleneck = self.bottleneck(curr_x, curr_edge_index, curr_edge_attr)
        return bottleneck, encoded_features

    def decode(self, x, encoded_features):
        """解码过程"""
        curr_x = x

        for decoder, (skip_x, skip_edge_index, skip_edge_attr, perm) in zip(
                self.decoder_blocks, reversed(encoded_features)
        ):
            # Unpooling: 还原节点数
            x_upsampled = torch.zeros(
                (skip_x.size(0), curr_x.size(1)),
                device=curr_x.device,
                dtype=curr_x.dtype
            )
            x_upsampled[perm] = curr_x

            # 解码器前向传播
            curr_x = decoder(
                x_upsampled, skip_x, skip_edge_index, skip_edge_attr
            )

        return curr_x

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        前向传播
        Args:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
            edge_attr: 边特征 [E, edge_features]
            batch: 批处理索引 [N]
        Returns:
            输出特征 [N, in_channels]
        """
        # 编码
        x, encoded_features = self.encode(x, edge_index, edge_attr, batch)

        # 解码
        x = self.decode(x, encoded_features)

        # 输出层
        x = self.output_layer(x, edge_index)

        return x[:, :self.in_channels]  # 如果需要，截断到原始通道数