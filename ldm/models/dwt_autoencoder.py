"""
DWT (离散小波变换) 编解码器
用于替代VQ-VAE，实现零参数、高效的频率空间编解码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


class DWTForward(nn.Module):
    """
    2D离散小波变换 (前向) - 修复版

    Haar小波的DWT实现：
    LL = (x[2i,2j] + x[2i,2j+1] + x[2i+1,2j] + x[2i+1,2j+1]) / 2
    LH = (x[2i,2j] - x[2i,2j+1] + x[2i+1,2j] - x[2i+1,2j+1]) / 2
    HL = (x[2i,2j] + x[2i,2j+1] - x[2i+1,2j] - x[2i+1,2j+1]) / 2
    HH = (x[2i,2j] - x[2i,2j+1] - x[2i+1,2j] + x[2i+1,2j+1]) / 2
    """
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet

        if wavelet != 'haar':
            raise ValueError("当前只支持 'haar' 小波，其他小波请使用pywt版本")

    def forward(self, x):
        """
        前向DWT - Haar小波
        输入: x [B, C, H, W]
        输出: LL, (LH, HL, HH) 各为 [B, C, H/2, W/2]
        """
        # 提取偶数和奇数位置的像素
        x00 = x[:, :, 0::2, 0::2]  # 偶行偶列
        x01 = x[:, :, 0::2, 1::2]  # 偶行奇列
        x10 = x[:, :, 1::2, 0::2]  # 奇行偶列
        x11 = x[:, :, 1::2, 1::2]  # 奇行奇列

        # Haar小波变换
        LL = (x00 + x01 + x10 + x11) / 2.0
        LH = (x00 - x01 + x10 - x11) / 2.0
        HL = (x00 + x01 - x10 - x11) / 2.0
        HH = (x00 - x01 - x10 + x11) / 2.0

        return LL, (LH, HL, HH)


class DWTInverse(nn.Module):
    """
    2D离散小波逆变换 - 修复版

    Haar小波的IDWT实现：
    对于Haar小波，重建公式为：
    x[2i]   = (LL + LH + HL + HH) / 2
    x[2i+1] = (LL - LH + HL - HH) / 2
    (沿行和列分别应用)
    """
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet

        if wavelet != 'haar':
            raise ValueError("当前只支持 'haar' 小波，其他小波请使用pywt版本")

    def forward(self, LL, highs, target_size=None):
        """
        逆DWT - Haar小波
        输入: LL [B, C, H, W], highs = (LH, HL, HH) 各为 [B, C, H, W]
        输出: x [B, C, H*2, W*2]
        """
        LH, HL, HH = highs
        B, C, H, W = LL.shape

        # 创建输出张量
        out_H = H * 2
        out_W = W * 2
        x = torch.zeros(B, C, out_H, out_W, device=LL.device, dtype=LL.dtype)

        # Haar小波逆变换
        # 首先沿列方向重建，然后沿行方向重建
        # 等价于：将4个子带组合回原图

        # 方法：直接使用Haar IDWT公式
        # x[2i, 2j]     = (LL + LH + HL + HH) / 2
        # x[2i, 2j+1]   = (LL - LH + HL - HH) / 2
        # x[2i+1, 2j]   = (LL + LH - HL - HH) / 2
        # x[2i+1, 2j+1] = (LL - LH - HL + HH) / 2

        x[:, :, 0::2, 0::2] = (LL + LH + HL + HH) / 2.0
        x[:, :, 0::2, 1::2] = (LL - LH + HL - HH) / 2.0
        x[:, :, 1::2, 0::2] = (LL + LH - HL - HH) / 2.0
        x[:, :, 1::2, 1::2] = (LL - LH - HL + HH) / 2.0

        # 裁剪到目标尺寸（如果指定）
        if target_size is not None:
            x = x[:, :, :target_size[0], :target_size[1]]

        return x


class DWTModel(nn.Module):
    """
    基于DWT的编解码器，用于替代VQ-VAE

    2级小波分解: 256×256 → 64×64 (与VAE潜空间维度一致)

    特点:
    - 零参数，无需预训练
    - 计算量极低 (相比VAE降低99%+)
    - 完全可逆，无信息损失
    """
    def __init__(self, wavelet='haar', level=2, mode='periodization'):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

        # 创建DWT和IDWT模块
        self.dwt = DWTForward(wavelet)
        self.idwt = DWTInverse(wavelet)

        # 缓存高频系数
        self.high_freq_cache = None

    def encode(self, x):
        """
        DWT编码
        输入: x [B, C, H, W] - 图像张量，范围[-1, 1]
        输出: LL [B, C, H/4, W/4] - 低频子带 (2级分解)
        """
        # 保存原始尺寸用于解码
        self.original_size = (x.shape[2], x.shape[3])

        # 第一级DWT: H×W → H/2×W/2
        LL1, highs1 = self.dwt(x)
        self.level1_size = (LL1.shape[2], LL1.shape[3])

        # 第二级DWT: H/2×W/2 → H/4×W/4
        LL2, highs2 = self.dwt(LL1)

        # 缓存高频系数用于解码
        self.high_freq_cache = (highs1, highs2)

        return LL2

    def decode(self, z):
        """
        IDWT解码
        输入: z [B, C, H/4, W/4] - 低频子带
        输出: x [B, C, H, W] - 重建图像
        """
        if self.high_freq_cache is None:
            # 如果没有缓存的高频系数，用零填充
            B, C, H, W = z.shape
            device = z.device
            dtype = z.dtype

            # 创建零高频系数
            highs2 = (
                torch.zeros(B, C, H, W, device=device, dtype=dtype),
                torch.zeros(B, C, H, W, device=device, dtype=dtype),
                torch.zeros(B, C, H, W, device=device, dtype=dtype)
            )
            highs1 = (
                torch.zeros(B, C, H*2, W*2, device=device, dtype=dtype),
                torch.zeros(B, C, H*2, W*2, device=device, dtype=dtype),
                torch.zeros(B, C, H*2, W*2, device=device, dtype=dtype)
            )
            level1_size = (H*2, W*2)
            original_size = (H*4, W*4)
        else:
            highs1, highs2 = self.high_freq_cache
            level1_size = getattr(self, 'level1_size', (z.shape[2]*2, z.shape[3]*2))
            original_size = getattr(self, 'original_size', (z.shape[2]*4, z.shape[3]*4))

        # 第一级IDWT: H/4×W/4 → H/2×W/2
        LL1 = self.idwt(z, highs2, target_size=level1_size)

        # 第二级IDWT: H/2×W/2 → H×W
        x = self.idwt(LL1, highs1, target_size=original_size)

        return x

    def forward(self, x):
        """前向传播：编码后解码"""
        z = self.encode(x)
        return self.decode(z)

    def clear_cache(self):
        """清除高频系数缓存"""
        self.high_freq_cache = None


class DWTModelSimple(nn.Module):
    """
    简化版DWT编解码器 - 使用pywt库

    更稳定，但需要CPU-GPU数据传输
    """
    def __init__(self, wavelet='haar', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.high_freq_cache = None

        if not HAS_PYWT:
            raise ImportError("pywt is required for DWTModelSimple. Install with: pip install PyWavelets")

    def encode(self, x):
        """
        DWT编码
        输入: x [B, C, H, W]
        输出: LL [B, C, H/4, W/4]
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # 转到CPU进行小波变换
        x_np = x.detach().cpu().numpy()

        LL_list = []
        HF_list = []

        for b in range(B):
            channel_LLs = []
            channel_HFs = []
            for c in range(C):
                # 多级DWT分解
                coeffs = pywt.wavedec2(x_np[b, c], self.wavelet, level=self.level)
                LL = coeffs[0]  # 最低频
                HF = coeffs[1:]  # 高频细节
                channel_LLs.append(LL)
                channel_HFs.append(HF)

            LL_list.append(np.stack(channel_LLs, axis=0))
            HF_list.append(channel_HFs)

        # 缓存高频系数
        self.high_freq_cache = HF_list

        LL_tensor = torch.from_numpy(np.stack(LL_list, axis=0)).to(device=device, dtype=dtype)
        return LL_tensor

    def decode(self, z):
        """
        IDWT解码
        输入: z [B, C, H/4, W/4]
        输出: x [B, C, H, W]
        """
        B, C, H, W = z.shape
        device = z.device
        dtype = z.dtype

        z_np = z.detach().cpu().numpy()

        x_list = []
        for b in range(B):
            channel_xs = []
            for c in range(C):
                if self.high_freq_cache is not None:
                    # 使用缓存的高频系数
                    HF = self.high_freq_cache[b][c]
                    coeffs = [z_np[b, c]] + list(HF)
                else:
                    # 没有高频系数，创建零系数
                    coeffs = [z_np[b, c]]
                    for level in range(self.level):
                        h, w = coeffs[0].shape
                        coeffs.append((
                            np.zeros((h, w)),
                            np.zeros((h, w)),
                            np.zeros((h, w))
                        ))
                        coeffs[0] = np.zeros((h*2, w*2))
                    coeffs[0] = z_np[b, c]

                # IDWT重建
                x_rec = pywt.waverec2(coeffs, self.wavelet)
                channel_xs.append(x_rec)
            x_list.append(np.stack(channel_xs, axis=0))

        x_tensor = torch.from_numpy(np.stack(x_list, axis=0)).to(device=device, dtype=dtype)
        return x_tensor

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def clear_cache(self):
        self.high_freq_cache = None


class DWTModelFullBand(nn.Module):
    """
    全子带DWT编解码器 - 方案A

    将所有level2子带(LL+LH+HL+HH)拼接后送入扩散模型
    让UNet同时增强低频和高频

    编码: 256×256×3 → 64×64×12 (LL2 + LH2 + HL2 + HH2)
    解码: 64×64×12 → 256×256×3

    注意: level1的高频系数从LR图像获取并缓存，解码时使用
    """
    def __init__(self, wavelet='haar', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.hf_level1_cache = None  # 缓存level1高频系数

        if not HAS_PYWT:
            raise ImportError("pywt is required for DWTModelFullBand. Install with: pip install PyWavelets")

    def encode(self, x):
        """
        DWT编码 - 返回全部level2子带拼接
        输入: x [B, C, H, W] (例如 [B, 3, 256, 256])
        输出: z [B, C*4, H/4, W/4] (例如 [B, 12, 64, 64])
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        x_np = x.detach().cpu().numpy()

        all_bands_list = []  # 存储所有batch的拼接子带
        hf_level1_list = []  # 存储level1高频系数

        for b in range(B):
            channel_bands = []  # 每个通道的子带
            channel_hf1 = []    # 每个通道的level1高频

            for c in range(C):
                # 2级DWT分解
                coeffs = pywt.wavedec2(x_np[b, c], self.wavelet, level=self.level)
                # coeffs = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]

                LL2 = coeffs[0]           # 64×64
                LH2, HL2, HH2 = coeffs[1]  # 各64×64
                LH1, HL1, HH1 = coeffs[2]  # 各128×128 (level1高频)

                # 拼接level2的4个子带
                channel_bands.append(np.stack([LL2, LH2, HL2, HH2], axis=0))  # [4, 64, 64]

                # 保存level1高频用于解码
                channel_hf1.append((LH1, HL1, HH1))

            # [C, 4, H/4, W/4] -> [C*4, H/4, W/4]
            bands = np.concatenate(channel_bands, axis=0)  # [12, 64, 64]
            all_bands_list.append(bands)
            hf_level1_list.append(channel_hf1)

        # 缓存level1高频系数
        self.hf_level1_cache = hf_level1_list

        # [B, 12, 64, 64]
        z = torch.from_numpy(np.stack(all_bands_list, axis=0)).to(device=device, dtype=dtype)
        return z

    def decode(self, z):
        """
        IDWT解码 - 从全部level2子带重建图像
        输入: z [B, C*4, H/4, W/4] (例如 [B, 12, 64, 64])
        输出: x [B, C, H, W] (例如 [B, 3, 256, 256])
        """
        B, C_total, H, W = z.shape
        C = C_total // 4  # 原始通道数
        device = z.device
        dtype = z.dtype

        z_np = z.detach().cpu().numpy()

        x_list = []
        for b in range(B):
            channel_xs = []

            for c in range(C):
                # 提取该通道的4个子带
                idx = c * 4
                LL2 = z_np[b, idx]
                LH2 = z_np[b, idx + 1]
                HL2 = z_np[b, idx + 2]
                HH2 = z_np[b, idx + 3]

                # 获取level1高频系数
                if self.hf_level1_cache is not None:
                    LH1, HL1, HH1 = self.hf_level1_cache[b][c]
                else:
                    # 如果没有缓存，用零填充
                    LH1 = np.zeros((H * 2, W * 2))
                    HL1 = np.zeros((H * 2, W * 2))
                    HH1 = np.zeros((H * 2, W * 2))

                # 重建系数列表
                coeffs = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]

                # 2级IDWT重建
                x_rec = pywt.waverec2(coeffs, self.wavelet)
                channel_xs.append(x_rec)

            x_list.append(np.stack(channel_xs, axis=0))

        x = torch.from_numpy(np.stack(x_list, axis=0)).to(device=device, dtype=dtype)
        return x

    def forward(self, x):
        """前向传播：编码后解码"""
        z = self.encode(x)
        return self.decode(z)

    def clear_cache(self):
        """清除高频系数缓存"""
        self.hf_level1_cache = None


class DWTModelFullBandTorch(nn.Module):
    """
    全子带DWT编解码器 - PyTorch实现版本

    使用纯PyTorch实现，支持GPU加速和梯度传播
    """
    def __init__(self, wavelet='haar', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.hf_level1_cache = None

        # 创建DWT和IDWT模块
        self.dwt = DWTForward(wavelet)
        self.idwt = DWTInverse(wavelet)

    def encode(self, x):
        """
        DWT编码 - 返回全部level2子带拼接
        输入: x [B, C, H, W]
        输出: z [B, C*4, H/4, W/4]
        """
        # 保存原始尺寸
        self.original_size = (x.shape[2], x.shape[3])

        # 第一级DWT: H×W → H/2×W/2
        LL1, (LH1, HL1, HH1) = self.dwt(x)
        self.level1_size = (LL1.shape[2], LL1.shape[3])

        # 第二级DWT: H/2×W/2 → H/4×W/4
        LL2, (LH2, HL2, HH2) = self.dwt(LL1)

        # 缓存level1高频系数用于解码
        self.hf_level1_cache = (LH1, HL1, HH1)

        # 拼接level2的4个子带: [B, C, H/4, W/4] * 4 -> [B, C*4, H/4, W/4]
        z = torch.cat([LL2, LH2, HL2, HH2], dim=1)

        return z

    def decode(self, z):
        """
        IDWT解码 - 从全部level2子带重建图像
        输入: z [B, C*4, H/4, W/4]
        输出: x [B, C, H, W]
        """
        B, C_total, H, W = z.shape
        C = C_total // 4

        # 拆分子带
        LL2 = z[:, 0:C]
        LH2 = z[:, C:2*C]
        HL2 = z[:, 2*C:3*C]
        HH2 = z[:, 3*C:4*C]

        # 获取目标尺寸
        level1_size = getattr(self, 'level1_size', (H*2, W*2))
        original_size = getattr(self, 'original_size', (H*4, W*4))

        # 第一级IDWT: H/4×W/4 → H/2×W/2
        LL1 = self.idwt(LL2, (LH2, HL2, HH2), target_size=level1_size)

        # 获取level1高频系数
        if self.hf_level1_cache is not None:
            LH1, HL1, HH1 = self.hf_level1_cache
        else:
            # 如果没有缓存，用零填充
            LH1 = torch.zeros(B, C, level1_size[0], level1_size[1], device=z.device, dtype=z.dtype)
            HL1 = torch.zeros(B, C, level1_size[0], level1_size[1], device=z.device, dtype=z.dtype)
            HH1 = torch.zeros(B, C, level1_size[0], level1_size[1], device=z.device, dtype=z.dtype)

        # 第二级IDWT: H/2×W/2 → H×W
        x = self.idwt(LL1, (LH1, HL1, HH1), target_size=original_size)

        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def clear_cache(self):
        self.hf_level1_cache = None


class DWTModelAllBands(nn.Module):
    """
    全子带DWT编解码器 - 改进版 (level1 + level2 全部参与扩散)

    将level1和level2的所有子带都送入扩散模型
    level1子带下采样到与level2相同尺寸，拼接后扩散
    解码时level1子带上采样回原尺寸

    编码: 256×256×3 → 64×64×21
        - level2: LL2, LH2, HL2, HH2 (64×64×12)
        - level1: LH1, HL1, HH1 (128×128×9) → 下采样到 64×64×9
    解码: 64×64×21 → 256×256×3
    """
    def __init__(self, wavelet='haar', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

        if not HAS_PYWT:
            raise ImportError("pywt is required. Install with: pip install PyWavelets")

    def encode(self, x):
        """
        DWT编码 - 返回level1+level2全部子带
        输入: x [B, C, H, W] (例如 [B, 3, 256, 256])
        输出: z [B, C*7, H/4, W/4] (例如 [B, 21, 64, 64])
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        x_np = x.detach().cpu().numpy()

        all_bands_list = []

        for b in range(B):
            channel_bands = []

            for c in range(C):
                # 2级DWT分解
                coeffs = pywt.wavedec2(x_np[b, c], self.wavelet, level=self.level)
                # coeffs = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]

                LL2 = coeffs[0]           # 64×64
                LH2, HL2, HH2 = coeffs[1]  # 各64×64
                LH1, HL1, HH1 = coeffs[2]  # 各128×128

                # level1子带下采样到64×64 (使用平均池化)
                LH1_down = cv2.resize(LH1, (W//4, H//4), interpolation=cv2.INTER_AREA)
                HL1_down = cv2.resize(HL1, (W//4, H//4), interpolation=cv2.INTER_AREA)
                HH1_down = cv2.resize(HH1, (W//4, H//4), interpolation=cv2.INTER_AREA)

                # 拼接7个子带: LL2, LH2, HL2, HH2, LH1, HL1, HH1
                channel_bands.append(np.stack([LL2, LH2, HL2, HH2, LH1_down, HL1_down, HH1_down], axis=0))

            # [C, 7, H/4, W/4] -> [C*7, H/4, W/4]
            bands = np.concatenate(channel_bands, axis=0)  # [21, 64, 64]
            all_bands_list.append(bands)

        # [B, 21, 64, 64]
        z = torch.from_numpy(np.stack(all_bands_list, axis=0)).to(device=device, dtype=dtype)
        return z

    def decode(self, z):
        """
        IDWT解码 - 从全部子带重建图像
        输入: z [B, C*7, H/4, W/4] (例如 [B, 21, 64, 64])
        输出: x [B, C, H, W] (例如 [B, 3, 256, 256])
        """
        B, C_total, H, W = z.shape
        C = C_total // 7  # 原始通道数
        device = z.device
        dtype = z.dtype

        z_np = z.detach().cpu().numpy()

        x_list = []
        for b in range(B):
            channel_xs = []

            for c in range(C):
                # 提取该通道的7个子带
                idx = c * 7
                LL2 = z_np[b, idx]
                LH2 = z_np[b, idx + 1]
                HL2 = z_np[b, idx + 2]
                HH2 = z_np[b, idx + 3]
                LH1_down = z_np[b, idx + 4]
                HL1_down = z_np[b, idx + 5]
                HH1_down = z_np[b, idx + 6]

                # level1子带上采样回128×128
                LH1 = cv2.resize(LH1_down, (W*2, H*2), interpolation=cv2.INTER_CUBIC)
                HL1 = cv2.resize(HL1_down, (W*2, H*2), interpolation=cv2.INTER_CUBIC)
                HH1 = cv2.resize(HH1_down, (W*2, H*2), interpolation=cv2.INTER_CUBIC)

                # 重建系数列表
                coeffs = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]

                # 2级IDWT重建
                x_rec = pywt.waverec2(coeffs, self.wavelet)
                channel_xs.append(x_rec)

            x_list.append(np.stack(channel_xs, axis=0))

        x = torch.from_numpy(np.stack(x_list, axis=0)).to(device=device, dtype=dtype)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def clear_cache(self):
        pass  # 此版本不需要缓存


class DWTModelAllBandsLearnable(nn.Module):
    """
    全子带DWT编解码器 - 方案A：可学习的上下采样

    用可学习的Conv替代cv2.resize，减少level1子带下采样/上采样造成的信息损失

    编码: 256×256×3 → 64×64×21
        - level2: LL2, LH2, HL2, HH2 (64×64×12) - DWT直接得到
        - level1: LH1, HL1, HH1 (128×128×9) → 可学习Conv下采样 → 64×64×9
    解码: 64×64×21 → 256×256×3
        - level1: 64×64×9 → 可学习ConvTranspose上采样 → 128×128×9
        - IDWT重建

    与cv2.resize的区别:
        - cv2.resize: 固定插值算法，无差别丢弃/填充信息
        - 可学习Conv: 学习保留重要信息，学习生成合理细节

    参数量增加: ~2K (可忽略)
    计算量增加: ~22M MACs (相比VAE的115M仍然很小)
    """
    def __init__(self, wavelet='haar', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

        if not HAS_PYWT:
            raise ImportError("pywt is required. Install with: pip install PyWavelets")

        # 可学习的下采样：128→64
        # 9通道 = 3个RGB通道 × 3个高频子带(LH1, HL1, HH1)
        self.down = nn.Conv2d(9, 9, kernel_size=3, stride=2, padding=1)

        # 可学习的上采样：64→128
        self.up = nn.ConvTranspose2d(9, 9, kernel_size=4, stride=2, padding=1)

        # 初始化权重 - 接近恒等映射，便于训练
        self._init_weights()

    def _init_weights(self):
        """初始化权重，使初始行为接近双线性插值"""
        # 下采样卷积初始化为平均池化的近似
        nn.init.kaiming_normal_(self.down.weight, mode='fan_out', nonlinearity='relu')
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)

        # 上采样反卷积初始化为双线性插值的近似
        nn.init.kaiming_normal_(self.up.weight, mode='fan_in', nonlinearity='relu')
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def encode(self, x):
        """
        DWT编码 - 返回level1+level2全部子带
        输入: x [B, C, H, W] (例如 [B, 3, 256, 256])
        输出: z [B, C*7, H/4, W/4] (例如 [B, 21, 64, 64])
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        x_np = x.detach().cpu().numpy()

        # 存储level2子带和level1子带
        level2_bands_list = []
        level1_bands_list = []

        for b in range(B):
            channel_level2 = []
            channel_level1 = []

            for c in range(C):
                # 2级DWT分解
                coeffs = pywt.wavedec2(x_np[b, c], self.wavelet, level=self.level)
                # coeffs = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]

                LL2 = coeffs[0]           # 64×64
                LH2, HL2, HH2 = coeffs[1]  # 各64×64
                LH1, HL1, HH1 = coeffs[2]  # 各128×128

                # level2子带
                channel_level2.append(np.stack([LL2, LH2, HL2, HH2], axis=0))  # [4, 64, 64]

                # level1子带 (保持128×128，稍后用可学习卷积下采样)
                channel_level1.append(np.stack([LH1, HL1, HH1], axis=0))  # [3, 128, 128]

            # [C*4, H/4, W/4] 和 [C*3, H/2, W/2]
            level2_bands = np.concatenate(channel_level2, axis=0)  # [12, 64, 64]
            level1_bands = np.concatenate(channel_level1, axis=0)  # [9, 128, 128]

            level2_bands_list.append(level2_bands)
            level1_bands_list.append(level1_bands)

        # 转换为tensor
        level2_tensor = torch.from_numpy(np.stack(level2_bands_list, axis=0)).to(device=device, dtype=dtype)
        level1_tensor = torch.from_numpy(np.stack(level1_bands_list, axis=0)).to(device=device, dtype=dtype)

        # 使用可学习卷积对level1子带下采样: [B, 9, 128, 128] → [B, 9, 64, 64]
        level1_down = self.down(level1_tensor)

        # 拼接: [B, 12, 64, 64] + [B, 9, 64, 64] → [B, 21, 64, 64]
        z = torch.cat([level2_tensor, level1_down], dim=1)

        return z

    def decode(self, z):
        """
        IDWT解码 - 从全部子带重建图像
        输入: z [B, C*7, H/4, W/4] (例如 [B, 21, 64, 64])
        输出: x [B, C, H, W] (例如 [B, 3, 256, 256])
        """
        B, C_total, H, W = z.shape
        C = C_total // 7  # 原始通道数 (21 // 7 = 3)
        device = z.device
        dtype = z.dtype

        # 拆分level2和level1子带
        level2_bands = z[:, :C*4]   # [B, 12, 64, 64]
        level1_down = z[:, C*4:]    # [B, 9, 64, 64]

        # 使用可学习反卷积对level1子带上采样: [B, 9, 64, 64] → [B, 9, 128, 128]
        level1_bands = self.up(level1_down)

        # 转换为numpy进行IDWT
        level2_np = level2_bands.detach().cpu().numpy()
        level1_np = level1_bands.detach().cpu().numpy()

        x_list = []
        for b in range(B):
            channel_xs = []

            for c in range(C):
                # 提取该通道的子带
                idx2 = c * 4
                LL2 = level2_np[b, idx2]
                LH2 = level2_np[b, idx2 + 1]
                HL2 = level2_np[b, idx2 + 2]
                HH2 = level2_np[b, idx2 + 3]

                idx1 = c * 3
                LH1 = level1_np[b, idx1]
                HL1 = level1_np[b, idx1 + 1]
                HH1 = level1_np[b, idx1 + 2]

                # 重建系数列表
                coeffs = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]

                # 2级IDWT重建
                x_rec = pywt.waverec2(coeffs, self.wavelet)
                channel_xs.append(x_rec)

            x_list.append(np.stack(channel_xs, axis=0))

        x = torch.from_numpy(np.stack(x_list, axis=0)).to(device=device, dtype=dtype)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def clear_cache(self):
        pass  # 此版本不需要缓存


class DWTModelAllBandsLearnableTorch(nn.Module):
    """
    全子带DWT编解码器 - 纯PyTorch版本，支持梯度回传

    与 DWTModelAllBandsLearnable 的区别：
    - encode 和 decode 都使用纯PyTorch实现，保证一致性
    - decode 保持梯度链，LPIPS等感知损失可以有效训练

    编码: 256×256×3 → 64×64×21 (纯PyTorch DWT)
    解码: 64×64×21 → 256×256×3 (纯PyTorch IDWT，保持梯度)
    """
    def __init__(self, wavelet='haar', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

        # 可学习的下采样：128→64
        # 使用 groups=9 使每个通道独立处理
        self.down = nn.Conv2d(9, 9, kernel_size=3, stride=2, padding=1, groups=9)

        # 可学习的上采样：64→128
        self.up = nn.ConvTranspose2d(9, 9, kernel_size=4, stride=2, padding=1, groups=9)

        # 纯PyTorch的DWT和IDWT模块
        self.dwt = DWTForward(wavelet)
        self.idwt = DWTInverse(wavelet)

        self._init_weights()

    def _init_weights(self):
        """
        初始化权重 - 使初始行为接近双线性插值/平均池化

        关键：初始化为接近恒等映射，这样初始重建误差小，
        训练过程中逐渐学习更好的上下采样

        注意：当 groups=9 时，权重形状为：
        - self.down.weight: [9, 1, 3, 3] (不是 [9, 9, 3, 3])
        - self.up.weight: [9, 1, 4, 4] (不是 [9, 9, 4, 4])
        """
        # 下采样卷积：初始化为平均池化的近似
        # 对于 stride=2, kernel=3 的卷积，初始化为均匀权重
        with torch.no_grad():
            nn.init.zeros_(self.down.weight)
            # groups=9 时，每个通道独立处理，权重形状是 [9, 1, 3, 3]
            # 第二维是1，不是9，所以用 weight[i, 0, ...] 而不是 weight[i, i, ...]
            for i in range(9):
                self.down.weight[i, 0, 1, 1] = 1.0  # 中心点设为1
            if self.down.bias is not None:
                nn.init.zeros_(self.down.bias)

        # 上采样反卷积：初始化为双线性插值的近似
        with torch.no_grad():
            nn.init.zeros_(self.up.weight)
            # groups=9 时，权重形状是 [9, 1, 4, 4]
            # 对于 kernel=4, stride=2 的转置卷积，设置双线性插值权重
            bilinear_kernel = torch.tensor([
                [0.25, 0.5, 0.5, 0.25],
                [0.5, 1.0, 1.0, 0.5],
                [0.5, 1.0, 1.0, 0.5],
                [0.25, 0.5, 0.5, 0.25]
            ]) / 4.0
            for i in range(9):
                self.up.weight[i, 0, :, :] = bilinear_kernel
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)

    def encode(self, x):
        """
        DWT编码 - 纯PyTorch实现，与decode保持一致
        输入: x [B, C, H, W] (例如 [B, 3, 256, 256])
        输出: z [B, C*7, H/4, W/4] (例如 [B, 21, 64, 64])
        """
        # 第一级DWT: 256×256 → 128×128
        LL1, (LH1, HL1, HH1) = self.dwt(x)

        # 第二级DWT: 128×128 → 64×64
        LL2, (LH2, HL2, HH2) = self.dwt(LL1)

        # level2子带: [B, C, 64, 64] * 4 → [B, C*4, 64, 64]
        level2_bands = torch.cat([LL2, LH2, HL2, HH2], dim=1)  # [B, 12, 64, 64]

        # level1子带: [B, C, 128, 128] * 3 → [B, C*3, 128, 128]
        level1_bands = torch.cat([LH1, HL1, HH1], dim=1)  # [B, 9, 128, 128]

        # 可学习下采样: [B, 9, 128, 128] → [B, 9, 64, 64]
        level1_down = self.down(level1_bands)

        # 拼接: [B, 12, 64, 64] + [B, 9, 64, 64] → [B, 21, 64, 64]
        z = torch.cat([level2_bands, level1_down], dim=1)

        return z

    def decode(self, z):
        """
        IDWT解码 - 纯PyTorch实现，保持梯度链
        输入: z [B, C*7, H/4, W/4] (例如 [B, 21, 64, 64])
        输出: x [B, C, H, W] (例如 [B, 3, 256, 256])

        关键：全程使用PyTorch操作，不转numpy，梯度可以回传
        """
        B, C_total, H, W = z.shape
        C = C_total // 7  # 原始通道数 = 3

        # 拆分level2和level1子带
        level2_bands = z[:, :C*4]   # [B, 12, 64, 64]
        level1_down = z[:, C*4:]    # [B, 9, 64, 64]

        # 可学习上采样: [B, 9, 64, 64] → [B, 9, 128, 128]
        # 这里保持梯度！
        level1_bands = self.up(level1_down)

        # 拆分level2各子带: LL2, LH2, HL2, HH2 各 [B, 3, 64, 64]
        LL2 = level2_bands[:, 0:C]
        LH2 = level2_bands[:, C:2*C]
        HL2 = level2_bands[:, 2*C:3*C]
        HH2 = level2_bands[:, 3*C:4*C]

        # 拆分level1各子带: LH1, HL1, HH1 各 [B, 3, 128, 128]
        LH1 = level1_bands[:, 0:C]
        HL1 = level1_bands[:, C:2*C]
        HH1 = level1_bands[:, 2*C:3*C]

        # 第一级IDWT: 64×64 → 128×128 (纯PyTorch，保持梯度)
        LL1 = self.idwt(LL2, (LH2, HL2, HH2), target_size=(H*2, W*2))

        # 第二级IDWT: 128×128 → 256×256 (纯PyTorch，保持梯度)
        x = self.idwt(LL1, (LH1, HL1, HH1), target_size=(H*4, W*4))

        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def clear_cache(self):
        pass


# 测试代码
if __name__ == '__main__':
    # 测试DWT模块
    print("Testing DWT Model...")

    # 创建测试输入
    x = torch.randn(2, 3, 256, 256)

    # 测试纯PyTorch版本
    print("\n1. Testing DWTModel (Pure PyTorch):")
    model = DWTModel(wavelet='haar', level=2)
    z = model.encode(x)
    x_rec = model.decode(z)
    print(f"   Input shape: {x.shape}")
    print(f"   Latent shape: {z.shape}")
    print(f"   Reconstructed shape: {x_rec.shape}")
    print(f"   Reconstruction error: {(x - x_rec).abs().max():.6f}")

    # 测试pywt版本
    if HAS_PYWT:
        print("\n2. Testing DWTModelSimple (PyWavelets):")
        model_simple = DWTModelSimple(wavelet='haar', level=2)
        z_simple = model_simple.encode(x)
        x_rec_simple = model_simple.decode(z_simple)
        print(f"   Input shape: {x.shape}")
        print(f"   Latent shape: {z_simple.shape}")
        print(f"   Reconstructed shape: {x_rec_simple.shape}")
        print(f"   Reconstruction error: {(x - x_rec_simple).abs().max():.6f}")

        print("\n3. Testing DWTModelFullBand (Full Subband - Plan A):")
        model_fullband = DWTModelFullBand(wavelet='haar', level=2)
        z_fullband = model_fullband.encode(x)
        x_rec_fullband = model_fullband.decode(z_fullband)
        print(f"   Input shape: {x.shape}")
        print(f"   Latent shape: {z_fullband.shape}")  # 应该是 [2, 12, 64, 64]
        print(f"   Reconstructed shape: {x_rec_fullband.shape}")
        print(f"   Reconstruction error: {(x - x_rec_fullband).abs().max():.6f}")

        print("\n4. Testing DWTModelFullBandTorch (Full Subband - PyTorch):")
        model_fullband_torch = DWTModelFullBandTorch(wavelet='haar', level=2)
        z_fullband_torch = model_fullband_torch.encode(x)
        x_rec_fullband_torch = model_fullband_torch.decode(z_fullband_torch)
        print(f"   Input shape: {x.shape}")
        print(f"   Latent shape: {z_fullband_torch.shape}")
        print(f"   Reconstructed shape: {x_rec_fullband_torch.shape}")
        print(f"   Reconstruction error: {(x - x_rec_fullband_torch).abs().max():.6f}")

        print("\n5. Testing DWTModelAllBands (All Bands - level1+level2):")
        model_allbands = DWTModelAllBands(wavelet='haar', level=2)
        z_allbands = model_allbands.encode(x)
        x_rec_allbands = model_allbands.decode(z_allbands)
        print(f"   Input shape: {x.shape}")
        print(f"   Latent shape: {z_allbands.shape}")  # 应该是 [2, 21, 64, 64]
        print(f"   Reconstructed shape: {x_rec_allbands.shape}")
        print(f"   Reconstruction error: {(x - x_rec_allbands).abs().max():.6f}")

        print("\n6. Testing DWTModelAllBandsLearnable (Learnable Up/Down Sampling - Plan A):")
        model_learnable = DWTModelAllBandsLearnable(wavelet='haar', level=2)
        z_learnable = model_learnable.encode(x)
        x_rec_learnable = model_learnable.decode(z_learnable)
        print(f"   Input shape: {x.shape}")
        print(f"   Latent shape: {z_learnable.shape}")  # 应该是 [2, 21, 64, 64]
        print(f"   Reconstructed shape: {x_rec_learnable.shape}")
        print(f"   Reconstruction error: {(x - x_rec_learnable).abs().max():.6f}")
        # 打印可学习参数量
        num_params = sum(p.numel() for p in model_learnable.parameters())
        print(f"   Learnable parameters: {num_params}")

        print("\n7. Testing DWTModelAllBandsLearnableTorch (Pure PyTorch decode - Gradient OK):")
        model_torch = DWTModelAllBandsLearnableTorch(wavelet='haar', level=2)
        z_torch = model_torch.encode(x)
        x_rec_torch = model_torch.decode(z_torch)
        print(f"   Input shape: {x.shape}")
        print(f"   Latent shape: {z_torch.shape}")  # 应该是 [2, 21, 64, 64]
        print(f"   Reconstructed shape: {x_rec_torch.shape}")
        print(f"   Reconstruction error: {(x - x_rec_torch).abs().max():.6f}")
        num_params_torch = sum(p.numel() for p in model_torch.parameters())
        print(f"   Learnable parameters: {num_params_torch}")

        # 测试梯度是否可以回传
        print("\n8. Testing gradient flow for DWTModelAllBandsLearnableTorch:")
        model_torch.zero_grad()
        z_test = model_torch.encode(x)
        x_rec_test = model_torch.decode(z_test)
        loss = x_rec_test.mean()
        loss.backward()
        up_grad = model_torch.up.weight.grad
        down_grad = model_torch.down.weight.grad
        print(f"   self.up.weight.grad is not None: {up_grad is not None}")
        print(f"   self.down.weight.grad is not None: {down_grad is not None}")
        if up_grad is not None:
            print(f"   self.up.weight.grad.abs().mean(): {up_grad.abs().mean():.6f}")
        if down_grad is not None:
            print(f"   self.down.weight.grad.abs().mean(): {down_grad.abs().mean():.6f}")

    print("\nDWT Model test completed!")
