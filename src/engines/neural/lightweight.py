import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HSwish(nn.Module):
    """
    Hard-Swish activation function, optimized for mobile NPUs.
    Formula: x * ReLU6(x+3) / 6
    """
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0

class GhostModule(nn.Module):
    """
    Ghost Module from GhostNet: reduces parameters by generating "ghost" features 
    using cheap depthwise convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            HSwish() if relu else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            HSwish() if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class SwinWindowAttention(nn.Module):
    """
    Lightweight Swin-Attention with window-based attention and Relative Position Bias.
    """
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        # Get relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        # x: [B*num_windows, Wh*Ww, C]
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class GhostResidualBlock(nn.Module):
    """
    MobileNetV3-style Inverted Residual Block using GhostModules.
    Expand -> Depthwise -> Squeeze
    """
    def __init__(self, in_chs, out_chs, mid_chs, stride=1):
        super().__init__()
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
        self.conv_dw = nn.Sequential(
            nn.Conv2d(mid_chs, mid_chs, 3, stride, 1, groups=mid_chs, bias=False),
            nn.BatchNorm2d(mid_chs),
            HSwish(),
        )
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_chs),
        ) if (stride != 1 or in_chs != out_chs) else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.ghost1(x)
        x = self.conv_dw(x)
        x = self.ghost2(x)
        return x + res

class GlobalContextGating(nn.Module):
    """
    Scales feature channels based on global context (similar to SE block).
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SovereignQuantizer(nn.Module):
    """
    Placeholder for SovereignQuantizer logic.
    Discrete bottleneck for compressibility.
    """
    def forward(self, x):
        # Rounding for discrete representation
        # Use straight-through estimator for gradients
        return x + (torch.round(x) - x).detach()

class DetailEngine(nn.Module):
    """
    The Mobile Specialist: Neural Engine for high-detail areas.
    GhostNet + Swin-Attention + QAT.
    """
    def __init__(self, in_channels=3, latent_channels=192):
        super().__init__()
        # Encoder
        self.enc_stage1 = GhostResidualBlock(in_channels, 64, 128, stride=2)   # 128 -> 64
        self.enc_stage2 = GhostResidualBlock(64, 128, 256, stride=2)          # 64 -> 32
        self.enc_stage3 = GhostResidualBlock(128, latent_channels, 512, stride=2) # 32 -> 16
        
        self.attn = SwinWindowAttention(latent_channels)
        self.quantizer = SovereignQuantizer()
        
        # Decoder
        self.dec_stage1 = GhostResidualBlock(latent_channels, 128, 512, stride=1)
        self.dec_stage2 = GhostResidualBlock(128, 64, 256, stride=1)
        self.dec_stage3 = GhostResidualBlock(64, 32, 128, stride=1)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, in_channels, 3, 1, 1),
            GlobalContextGating(in_channels)
        )

    def encode(self, x):
        x = self.enc_stage1(x)
        x = self.enc_stage2(x)
        x = self.enc_stage3(x)
        
        # Attention on bottleneck (16x16 grid)
        # Partition into windows of size 8x8
        B, C, H, W = x.shape
        window_size = 8
        
        # (B, C, 2, 8, 2, 8) -> (B, 2, 2, 8, 8, C) -> (B*4, 64, C)
        x_windows = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
        
        # Apply Attention
        x_windows = self.attn(x_windows)
        
        # Merge windows back
        x = x_windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        
        latent = self.quantizer(x)
        return latent

    def decode(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) # 16 -> 32
        x = self.dec_stage1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) # 32 -> 64
        x = self.dec_stage2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) # 64 -> 128
        x = self.dec_stage3(x)
        
        out = self.final_conv(x)
        return out

    def forward(self, x):
        latent = self.encode(x)
        out = self.decode(latent)
        return out
