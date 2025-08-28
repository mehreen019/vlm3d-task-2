#!/usr/bin/env python3
"""
Advanced Attention Mechanisms for Medical Image Analysis
Implements CBAM, SE-Net, and custom medical attention modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(attention_map)
        attention = self.sigmoid(attention_map)
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MedicalAttention(nn.Module):
    """Custom attention mechanism designed for medical images"""
    
    def __init__(self, in_channels: int, num_regions: int = 4):
        super().__init__()
        self.num_regions = num_regions
        self.in_channels = in_channels
        
        # Region-aware attention
        self.region_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, 1, 1),
                nn.Sigmoid()
            ) for _ in range(num_regions)
        ])
        
        # Multi-scale feature aggregation
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels)
            for k in [1, 3, 5, 7]
        ])
        
        self.fusion = nn.Conv2d(in_channels * 4, in_channels, 1)
        
        # Pathology-aware weights
        self.pathology_weights = nn.Parameter(torch.ones(num_regions))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for conv in self.multi_scale:
            multi_scale_features.append(conv(x))
        
        multi_scale_x = torch.cat(multi_scale_features, dim=1)
        multi_scale_x = self.fusion(multi_scale_x)
        
        # Region-based attention
        region_height = h // 2
        region_width = w // 2
        
        regions = [
            x[:, :, :region_height, :region_width],  # Top-left
            x[:, :, :region_height, region_width:],  # Top-right
            x[:, :, region_height:, :region_width],  # Bottom-left
            x[:, :, region_height:, region_width:]   # Bottom-right
        ]
        
        attended_regions = []
        for i, (region, attention_module) in enumerate(zip(regions, self.region_attention)):
            attention_map = attention_module(region)
            attended_region = region * attention_map * self.pathology_weights[i]
            attended_regions.append(attended_region)
        
        # Reconstruct full image
        top = torch.cat([attended_regions[0], attended_regions[1]], dim=3)
        bottom = torch.cat([attended_regions[2], attended_regions[3]], dim=3)
        attended_x = torch.cat([top, bottom], dim=2)
        
        return multi_scale_x + attended_x


class DualAttention(nn.Module):
    """Dual Attention combining position and channel attention"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.position_attention = PositionAttention(in_channels)
        self.channel_attention = ChannelAttentionModule(in_channels)
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_att = self.position_attention(x)
        chan_att = self.channel_attention(x)
        
        out = self.alpha * pos_att + self.beta * chan_att + x
        return out


class PositionAttention(nn.Module):
    """Position Attention Module"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        proj_query = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, w * h)
        proj_value = self.value_conv(x).view(b, -1, w * h)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        
        out = self.gamma * out + x
        return out


class ChannelAttentionModule(nn.Module):
    """Channel Attention Module for Dual Attention"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        proj_query = x.view(b, c, -1)
        proj_key = x.view(b, c, -1).permute(0, 2, 1)
        proj_value = x.view(b, c, -1)
        
        energy = torch.bmm(proj_query, proj_key)
        max_energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy - energy
        
        attention = self.softmax(energy_new)
        out = torch.bmm(attention, proj_value)
        out = out.view(b, c, h, w)
        
        out = self.gamma * out + x
        return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention for feature aggregation"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
    
    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return torch.matmul(attention_weights, v)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output = self.scaled_dot_product_attention(q, k, v)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output


class AttentionAggregator(nn.Module):
    """Attention-based aggregation for multiple slices per volume"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slice_features: [batch_size, num_slices, feature_dim]
        Returns:
            aggregated_features: [batch_size, feature_dim]
        """
        # Transform features
        transformed_features = self.feature_transform(slice_features)
        
        # Compute attention weights
        attention_scores = self.attention(transformed_features)  # [batch_size, num_slices, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_slices, 1]
        
        # Weighted aggregation
        aggregated = torch.sum(transformed_features * attention_weights, dim=1)  # [batch_size, feature_dim]
        
        return aggregated


def create_attention_module(attention_type: str, in_channels: int, **kwargs) -> nn.Module:
    """Factory function to create attention modules"""
    
    attention_modules = {
        'cbam': lambda: CBAM(in_channels, **kwargs),
        'se': lambda: SEBlock(in_channels, **kwargs),
        'medical': lambda: MedicalAttention(in_channels, **kwargs),
        'dual': lambda: DualAttention(in_channels),
        'channel': lambda: ChannelAttention(in_channels, **kwargs),
        'spatial': lambda: SpatialAttention(**kwargs)
    }
    
    if attention_type not in attention_modules:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    return attention_modules[attention_type]()
