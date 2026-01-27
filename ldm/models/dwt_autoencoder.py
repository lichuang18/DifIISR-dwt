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
    2D离散小波变换 (前向)
    使用PyTorch实现，支持GPU加速和梯度传播
    """
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet

        # 获取小波滤波器
        if wavelet == 'haar':
            # Haar小波滤波器
            self.dec_lo = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.float32)
            self.dec_hi = torch.tensor([1/np.sqrt(2), -1/np.sqrt(2)], dtype=torch.float32)
        else:
            # 使用pywt获取其他小波的滤波器
            if HAS_PYWT:
                wavelet_obj = pywt.Wavelet(wavelet)
                self.dec_lo = torch.tensor(wavelet_obj.dec_lo, dtype=torch.float32)
                self.dec_hi = torch.tensor(wavelet_obj.dec_hi, dtype=torch.float32)
            else:
                raise ValueError(f"pywt not installed, only 'haar' wavelet is supported")

        # 注册为buffer
        self.register_buffer('dec_lo_buf', self.dec_lo)
        self.register_buffer('dec_hi_buf', self.dec_hi)

    def forward(self, x):
        """
        前向DWT
        输入: x [B, C, H, W]
        输出: LL, LH, HL, HH 各为 [B, C, H/2, W/2]
        """
        B, C, H, W = x.shape

        # 构建2D滤波器
        lo = self.dec_lo_buf.to(x.device)
        hi = self.dec_hi_buf.to(x.device)

        # 外积得到2D滤波器
        ll_filter = torch.outer(lo, lo).unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
        lh_filter = torch.outer(hi, lo).unsqueeze(0).unsqueeze(0)
        hl_filter = torch.outer(lo, hi).unsqueeze(0).unsqueeze(0)
        hh_filter = torch.outer(hi, hi).unsqueeze(0).unsqueeze(0)

        # 扩展到所有通道
        ll_filter = ll_filter.repeat(C, 1, 1, 1)  # [C, 1, k, k]
        lh_filter = lh_filter.repeat(C, 1, 1, 1)
        hl_filter = hl_filter.repeat(C, 1, 1, 1)
        hh_filter = hh_filter.repeat(C, 1, 1, 1)

        # 填充
        pad_size = len(lo) // 2
        x_pad = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

        # 卷积并下采样
        LL = F.conv2d(x_pad, ll_filter, stride=2, groups=C)
        LH = F.conv2d(x_pad, lh_filter, stride=2, groups=C)
        HL = F.conv2d(x_pad, hl_filter, stride=2, groups=C)
        HH = F.conv2d(x_pad, hh_filter, stride=2, groups=C)

        return LL, (LH, HL, HH)


class DWTInverse(nn.Module):
    """
    2D离散小波逆变换
    """
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet

        # 获取重建滤波器
        if wavelet == 'haar':
            self.rec_lo = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.float32)
            self.rec_hi = torch.tensor([1/np.sqrt(2), -1/np.sqrt(2)], dtype=torch.float32)
        else:
            if HAS_PYWT:
                wavelet_obj = pywt.Wavelet(wavelet)
                self.rec_lo = torch.tensor(wavelet_obj.rec_lo, dtype=torch.float32)
                self.rec_hi = torch.tensor(wavelet_obj.rec_hi, dtype=torch.float32)
            else:
                raise ValueError(f"pywt not installed, only 'haar' wavelet is supported")

        self.register_buffer('rec_lo_buf', self.rec_lo)
        self.register_buffer('rec_hi_buf', self.rec_hi)

    def forward(self, LL, highs):
        """
        逆DWT
        输入: LL [B, C, H, W], highs = (LH, HL, HH) 各为 [B, C, H, W]
        输出: x [B, C, H*2, W*2]
        """
        LH, HL, HH = highs
        B, C, H, W = LL.shape

        lo = self.rec_lo_buf.to(LL.device)
        hi = self.rec_hi_buf.to(LL.device)

        # 构建2D重建滤波器
        ll_filter = torch.outer(lo, lo).unsqueeze(0).unsqueeze(0)
        lh_filter = torch.outer(hi, lo).unsqueeze(0).unsqueeze(0)
        hl_filter = torch.outer(lo, hi).unsqueeze(0).unsqueeze(0)
        hh_filter = torch.outer(hi, hi).unsqueeze(0).unsqueeze(0)

        # 扩展到所有通道
        ll_filter = ll_filter.repeat(C, 1, 1, 1)
        lh_filter = lh_filter.repeat(C, 1, 1, 1)
        hl_filter = hl_filter.repeat(C, 1, 1, 1)
        hh_filter = hh_filter.repeat(C, 1, 1, 1)

        # 上采样并卷积
        k = len(lo)

        # 使用转置卷积进行上采样
        LL_up = F.conv_transpose2d(LL, ll_filter, stride=2, groups=C, padding=k//2, output_padding=0)
        LH_up = F.conv_transpose2d(LH, lh_filter, stride=2, groups=C, padding=k//2, output_padding=0)
        HL_up = F.conv_transpose2d(HL, hl_filter, stride=2, groups=C, padding=k//2, output_padding=0)
        HH_up = F.conv_transpose2d(HH, hh_filter, stride=2, groups=C, padding=k//2, output_padding=0)

        # 求和重建
        x = LL_up + LH_up + HL_up + HH_up

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
        # 第一级DWT: H×W → H/2×W/2
        LL1, highs1 = self.dwt(x)

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
        else:
            highs1, highs2 = self.high_freq_cache

        # 第一级IDWT: H/4×W/4 → H/2×W/2
        LL1 = self.idwt(z, highs2)

        # 第二级IDWT: H/2×W/2 → H×W
        x = self.idwt(LL1, highs1)

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
        # 第一级DWT: H×W → H/2×W/2
        LL1, (LH1, HL1, HH1) = self.dwt(x)

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

        # 第一级IDWT: H/4×W/4 → H/2×W/2
        LL1 = self.idwt(LL2, (LH2, HL2, HH2))

        # 获取level1高频系数
        if self.hf_level1_cache is not None:
            LH1, HL1, HH1 = self.hf_level1_cache
        else:
            # 如果没有缓存，用零填充
            LH1 = torch.zeros(B, C, H * 2, W * 2, device=z.device, dtype=z.dtype)
            HL1 = torch.zeros(B, C, H * 2, W * 2, device=z.device, dtype=z.dtype)
            HH1 = torch.zeros(B, C, H * 2, W * 2, device=z.device, dtype=z.dtype)

        # 第二级IDWT: H/2×W/2 → H×W
        x = self.idwt(LL1, (LH1, HL1, HH1))

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

    print("\nDWT Model test completed!")
