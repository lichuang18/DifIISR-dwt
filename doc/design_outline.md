# DifIISR-DWT: 基于离散小波变换的红外图像超分辨率设计大纲

## 1. 项目概述

### 1.1 项目目标

将DifIISR中的VQ-VAE潜空间编解码器替换为离散小波变换(DWT)，实现：
- 消除对预训练VAE模型的依赖
- 大幅降低编解码计算量（99%+）
- 减少模型参数和存储需求（-49M参数，-196MB）
- 保持或提升红外图像超分辨率质量

### 1.2 技术路线

```
原方案 (DifIISR):
  LR图像 → Bicubic↑ → VAE Encoder → 潜空间扩散 → VAE Decoder → SR图像
                      (50M MACs)                    (65M MACs)

新方案 (DifIISR-DWT):
  LR图像 → Bicubic↑ → DWT分解 → 频率空间扩散 → IDWT重建 → SR图像
                      (0.3M)                      (0.3M)
```

---

## 2. 架构设计

### 2.1 DWT编解码模块设计

```python
class DWTModel:
    """
    使用2级Haar小波变换替代VQ-VAE

    编码: 256×256×3 → 64×64×12 (LL + LH + HL + HH，两级)
    解码: 64×64×12 → 256×256×3
    """

    def encode(self, x):
        # 第一级DWT: 256→128
        LL1, (LH1, HL1, HH1) = dwt2(x, 'haar')
        # 第二级DWT: 128→64
        LL2, (LH2, HL2, HH2) = dwt2(LL1, 'haar')
        # 返回所有子带或仅LL
        return LL2  # 64×64×3，与VAE潜空间维度一致

    def decode(self, z, high_freq_coeffs):
        # 两级IDWT重建
        LL1 = idwt2((z, high_freq_coeffs[1]), 'haar')
        x = idwt2((LL1, high_freq_coeffs[0]), 'haar')
        return x
```

### 2.2 扩散空间对比

| 特性 | VAE潜空间 | DWT频率空间 |
|------|-----------|-------------|
| 维度 | 64×64×3 | 64×64×3 (仅LL) 或 64×64×12 (全部) |
| 物理意义 | 学习的语义特征 | 低频近似分量 |
| 可逆性 | 有损（量化） | 无损 |
| 计算量 | ~115M MACs | ~0.6M MACs |

### 2.3 两种DWT方案

#### 方案A：仅LL子带扩散（推荐）

```
优点：
- 维度与原VAE完全一致 (64×64×3)
- UNet结构无需修改
- 高频子带直接传递，保留细节

流程：
  x → DWT → LL (扩散) + HF (保留) → IDWT → x_sr
```

#### 方案B：全子带扩散

```
优点：
- 扩散模型可以增强高频细节
- 理论上质量上限更高

缺点：
- 需要修改UNet输入通道 (3→12)
- 计算量增加
```

---

## 3. 代码修改计划

### 3.1 新增文件

| 文件 | 功能 | 说明 |
|------|------|------|
| `ldm/models/dwt_autoencoder.py` | DWT编解码模块 | 替代VQ-VAE |
| `configs/DifIISR_dwt.yaml` | DWT版本配置 | 新配置文件 |
| `configs/DifIISR_dwt_train.yaml` | DWT训练配置 | 训练超参数 |
| `train_dwt.py` | 训练脚本 | 在DWT空间训练UNet |
| `inference_dwt.py` | 推理脚本 | DWT版本推理 |

### 3.2 修改文件

| 文件 | 修改内容 | 原因 |
|------|----------|------|
| `sampler.py` | 支持DWT autoencoder | 兼容新编解码器 |
| `models/gaussian_diffusion_test.py` | 调整scale_factor | DWT不需要缩放 |

### 3.3 核心代码实现

#### 3.3.1 DWT Autoencoder (`ldm/models/dwt_autoencoder.py`)

```python
import torch
import torch.nn as nn
import pywt
import numpy as np

class DWTModel(nn.Module):
    """
    离散小波变换编解码器
    替代VQ-VAE用于DifIISR
    """
    def __init__(self, wavelet='haar', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.high_freq_cache = {}  # 缓存高频系数

    def encode(self, x):
        """
        DWT编码
        输入: x [B, C, H, W] - 图像张量
        输出: LL [B, C, H/4, W/4] - 低频子带
        """
        B, C, H, W = x.shape
        device = x.device

        # 转换为numpy进行DWT
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

        # 缓存高频系数用于解码
        self.high_freq_cache = HF_list

        LL_tensor = torch.from_numpy(np.stack(LL_list, axis=0)).float().to(device)
        return LL_tensor

    def decode(self, z):
        """
        IDWT解码
        输入: z [B, C, H/4, W/4] - 低频子带
        输出: x [B, C, H, W] - 重建图像
        """
        B, C, H, W = z.shape
        device = z.device

        z_np = z.detach().cpu().numpy()

        x_list = []
        for b in range(B):
            channel_xs = []
            for c in range(C):
                # 获取缓存的高频系数
                HF = self.high_freq_cache[b][c]
                # 重建系数列表
                coeffs = [z_np[b, c]] + list(HF)
                # IDWT重建
                x_rec = pywt.waverec2(coeffs, self.wavelet)
                channel_xs.append(x_rec)
            x_list.append(np.stack(channel_xs, axis=0))

        x_tensor = torch.from_numpy(np.stack(x_list, axis=0)).float().to(device)
        return x_tensor

    def forward(self, x):
        """前向传播：编码后解码"""
        z = self.encode(x)
        return self.decode(z)
```

#### 3.3.2 配置文件 (`configs/DifIISR_dwt.yaml`)

```yaml
model:
  target: models.unet.UNetModelSwin
  ckpt_path: weights/DifIISR_dwt.pth  # 新训练的权重
  params:
    image_size: 64
    in_channels: 6      # 3(噪声) + 3(LQ条件)
    model_channels: 160
    out_channels: 3
    cond_lq: True
    attention_resolutions: [64,32,16,8]
    dropout: 0
    channel_mult: [1, 2, 2, 4]
    num_res_blocks: [2, 2, 2, 2]
    conv_resample: True
    dims: 2
    use_fp16: False
    num_head_channels: 32
    use_scale_shift_norm: True
    resblock_updown: False
    swin_depth: 2
    swin_embed_dim: 192
    window_size: 8
    mlp_ratio: 4

diffusion:
  target: models.script_util.create_gaussian_diffusion_test
  params:
    sf: 4
    schedule_name: exponential
    schedule_kwargs:
      power: 0.3
    etas_end: 0.99
    steps: 15
    min_noise_level: 0.04
    kappa: 2.0              # 可能需要调整
    weighted_mse: False
    predict_type: xstart
    timestep_respacing: ~
    scale_factor: 1.0       # DWT不需要缩放
    normalize_input: True
    latent_flag: True

# 使用DWT替代VAE
autoencoder:
  target: ldm.models.dwt_autoencoder.DWTModel
  ckpt_path: ~              # 无需权重文件！
  use_fp16: False
  params:
    wavelet: haar
    level: 2
```

#### 3.3.3 训练配置 (`configs/DifIISR_dwt_train.yaml`)

```yaml
# 继承测试配置
_base_: DifIISR_dwt.yaml

# 数据配置
data:
  train:
    type: paired_infrared
    params:
      hr_dir: /path/to/infrared/HR
      lr_dir: /path/to/infrared/LR
      gt_size: 256
      scale: 4
      use_hflip: True
      use_rot: True
  val:
    type: paired_infrared
    params:
      hr_dir: /path/to/infrared/val/HR
      lr_dir: /path/to/infrared/val/LR

# 训练配置
train:
  lr: 5e-5
  batch_size: 16
  iterations: 200000
  ema_rate: 0.999
  save_freq: 10000
  val_freq: 5000
  log_freq: 100

  # 优化器
  optimizer:
    type: AdamW
    weight_decay: 0.01

  # 学习率调度
  scheduler:
    type: CosineAnnealingLR
    T_max: 200000
    eta_min: 1e-6
```

---

## 4. 修改原因与收益分析

### 4.1 为什么用DWT替换VAE

| 原因 | 详细说明 |
|------|----------|
| **VAE是纯开销** | VAE在DifIISR中完全冻结，不参与训练，只做编解码 |
| **语义压缩无用** | 超分任务不需要语义理解，只需恢复高频细节 |
| **计算浪费** | 每次推理VAE编解码消耗115M MACs，但不产生价值 |
| **部署复杂** | 需要额外下载196MB的VAE权重文件 |
| **红外适配** | 红外图像低频主导，DWT压缩更高效 |

### 4.2 DWT的优势

| 优势 | 说明 |
|------|------|
| **零参数** | 无需预训练，无需存储权重 |
| **计算高效** | 编解码计算量降低99.5% |
| **无损变换** | 完美重建，无量化误差 |
| **物理可解释** | 频率分量有明确物理意义 |
| **边缘保持** | 小波变换天然保持边缘信息 |
| **多尺度** | 自然的多分辨率分析 |

### 4.3 预期收益量化

```
┌─────────────────────┬──────────────┬──────────────┬──────────────┐
│       指标          │   原方案     │   新方案     │    收益      │
├─────────────────────┼──────────────┼──────────────┼──────────────┤
│ 编解码参数          │   49M        │   0          │   -100%      │
│ 权重文件大小        │   196MB      │   0          │   -100%      │
│ 编解码计算量        │   115M MACs  │   0.6M MACs  │   -99.5%     │
│ 推理显存            │   +200MB     │   +10MB      │   -95%       │
│ 编解码延迟          │   ~50ms      │   ~0.5ms     │   -99%       │
│ 部署依赖            │   需VAE模型  │   无         │   简化       │
└─────────────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 5. 训练计划

### 5.1 为什么需要重新训练

原DifIISR的UNet是在VAE潜空间训练的，其特征分布与DWT频率空间不同：

| 空间 | 特征分布 | 数值范围 |
|------|----------|----------|
| VAE潜空间 | 学习的语义特征，近似高斯 | 约[-3, 3] |
| DWT频率空间 | 物理频率分量，能量集中 | 约[-1, 1] |

**直接替换会导致UNet无法正确去噪，必须在DWT空间重新训练。**

### 5.2 训练阶段

#### 阶段一：数据准备（1-2周）

1. **收集红外数据集**
2. **生成LR-HR配对**
3. **数据预处理和增强**

#### 阶段二：基础训练（2-3周）

```
目标：在DWT空间训练UNet基础能力

配置：
- 数据集：红外HR图像 + 合成LR
- Batch size: 16
- 学习率: 5e-5
- 迭代次数: 200K
- 扩散步数: 15

监控指标：
- 训练损失收敛
- 验证集PSNR/SSIM
```

#### 阶段三：微调优化与消融实验（1-2周）

```
目标：针对红外特性优化，验证最佳配置

调整项：
- 噪声调度参数 (kappa, etas)
- 学习率衰减策略
- 数据增强策略
```

##### 3.1 消融实验：小波基 vs 分解级数

**关键结论：小波基消融成本低，分解级数消融成本高**

| 消融类型 | 是否需要重新训练 | 原因 | 训练量 |
|----------|------------------|------|--------|
| **小波基** (haar/db2/bior) | 否/微调即可 | LL子带数值分布相近 | 0 或 20K-50K |
| **分解级数** (1/2/3级) | **必须重新训练** | 潜空间维度改变 | 完整 100K-200K |

##### 3.2 分解级数影响分析

不同分解级数会改变潜空间维度，直接影响UNet架构：

```
1级分解: 256×256 → 128×128×3  (潜空间变大，UNet需调整)
2级分解: 256×256 → 64×64×3    (与VAE一致，推荐，UNet无需改动)
3级分解: 256×256 → 32×32×3    (潜空间变小，UNet需调整)
```

**结论：推荐固定使用2级分解，与原VAE潜空间维度一致，避免修改UNet架构。**

##### 3.3 小波基影响分析

不同小波基的LL子带统计特性相近，UNet可能直接适应：

```python
# 不同小波基LL子带对比示例
import pywt
import numpy as np

x = np.random.randn(256, 256)

LL_haar, _ = pywt.dwt2(x, 'haar')  # mean≈0, std≈1
LL_db2, _ = pywt.dwt2(x, 'db2')    # mean≈0, std≈1 (略有差异)
LL_bior, _ = pywt.dwt2(x, 'bior1.3') # mean≈0, std≈1 (略有差异)

# 统计特性接近，UNet可能无需重新训练即可适应
```

##### 3.4 推荐消融实验流程

```
步骤1: 主训练 (haar + 2级分解)
        │
        │ 200K iterations，得到基准模型
        ▼
步骤2: 小波基消融 (低成本)
        │
        ├─→ 直接替换db2测试 (0 iterations)
        │   └─→ 如果PSNR下降 < 0.5dB → 可接受，无需微调
        │   └─→ 如果PSNR下降 > 0.5dB → 微调 20K-50K iterations
        │
        └─→ 直接替换bior测试 (0 iterations)
            └─→ 同上判断
        │
        ▼
步骤3: 分解级数消融 (高成本，可选)
        │
        ├─→ 1级分解：需修改UNet + 重新训练 100K-200K
        │   (潜空间128×128，计算量增加，可能质量更好)
        │
        └─→ 3级分解：需修改UNet + 重新训练 100K-200K
            (潜空间32×32，计算量减少，可能质量下降)
```

##### 3.5 消融实验优先级

| 优先级 | 实验 | 预期收益 | 成本 |
|--------|------|----------|------|
| **P0** | haar + 2级 (主训练) | 基准 | 200K iter |
| **P1** | 小波基替换测试 | 找最佳小波基 | ~0 |
| **P2** | 小波基微调 (如需要) | 提升0.1-0.3dB | 20K-50K iter |
| **P3** | kappa/etas调参 | 收敛稳定性 | 多次短训练 |
| **P4** | 1级分解 (可选) | 可能+0.5dB | 200K iter |
| **P5** | 3级分解 (可选) | 速度提升 | 200K iter |

**建议：P0-P3为必做，P4-P5根据时间和资源决定。**

#### 阶段四：评估对比（1周）

```
对比实验：
- DifIISR-DWT vs 原DifIISR
- DifIISR-DWT vs Bicubic/EDSR/RCAN

评估指标：
- PSNR, SSIM, LPIPS
- 推理速度
- 显存占用
```

### 5.3 训练资源需求

| 资源 | 需求 |
|------|------|
| GPU | 1× RTX 3090/4090 或更高 |
| 显存 | 16GB+ |
| 训练时间 | 约3-5天 (200K iterations) |
| 存储 | 50GB+ (数据集+检查点) |

---

## 6. 数据集准备

### 6.1 推荐数据集

#### 主要训练数据

| 数据集 | 图像数量 | 分辨率 | 说明 |
|--------|----------|--------|------|
| **FLIR ADAS** | 10K+ | 640×512 | 自动驾驶红外，场景丰富 |
| **CVC-14** | 7K+ | 640×480 | 行人检测红外 |
| **KAIST** | 95K+ | 640×480 | 多光谱配对数据 |
| **M3FD** | 4.2K | 1024×768 | 高质量红外 |

#### 验证/测试数据

| 数据集 | 用途 |
|--------|------|
| **FMB** | 红外-可见光融合，可用于测试 |
| **TNO** | 经典红外测试集 |
| **自建测试集** | 特定场景评估 |

### 6.2 数据准备流程

```
1. 下载原始HR红外图像
   └── 确保分辨率 ≥ 256×256

2. 生成LR图像 (两种方式)
   ├── 方式A: Bicubic下采样 (简单)
   │   └── HR → Bicubic↓4× → LR
   │
   └── 方式B: Real-ESRGAN退化 (更真实)
       └── HR → 模糊+噪声+压缩+下采样 → LR

3. 数据组织结构
   dataset/
   ├── train/
   │   ├── HR/
   │   │   ├── 0001.png
   │   │   └── ...
   │   └── LR/
   │       ├── 0001.png
   │       └── ...
   └── val/
       ├── HR/
       └── LR/

4. 数据增强
   ├── 随机裁剪 (256×256 patches)
   ├── 水平翻转
   ├── 旋转 (90°, 180°, 270°)
   └── 可选: 亮度/对比度调整
```

### 6.3 数据集下载链接

```
FLIR ADAS:
https://www.flir.com/oem/adas/adas-dataset-form/

KAIST Multispectral:
https://soonminhwang.github.io/rgbt-ped-detection/

M3FD:
https://github.com/JinyuanLiu-CV/TarDAL

CVC-14:
http://adas.cvc.uab.es/elektra/datasets/
```

---

## 7. 实验设计

### 7.1 消融实验

| 实验 | 目的 |
|------|------|
| 小波基选择 | 对比haar/db2/bior对质量的影响 |
| 分解级数 | 对比1/2/3级分解的效果 |
| 扩散步数 | 对比5/10/15/20步的质量-速度权衡 |
| kappa参数 | 调整噪声强度对收敛的影响 |

### 7.2 对比实验

| 方法 | 类型 |
|------|------|
| Bicubic | 传统插值基线 |
| EDSR | CNN超分 |
| RCAN | 注意力CNN超分 |
| SwinIR | Transformer超分 |
| DifIISR (原版) | VAE潜空间扩散 |
| **DifIISR-DWT** | DWT频率空间扩散 |

### 7.3 评估指标

| 指标 | 说明 |
|------|------|
| PSNR | 峰值信噪比，衡量像素级误差 |
| SSIM | 结构相似性，衡量结构保持 |
| LPIPS | 感知相似性，衡量视觉质量 |
| FLOPs | 计算量 |
| 推理时间 | 实际速度 |
| 显存占用 | 部署友好性 |

---

## 8. 风险与应对

### 8.1 潜在风险

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|----------|
| DWT空间扩散收敛困难 | 中 | 高 | 调整噪声调度，增加训练步数 |
| 质量不如VAE版本 | 中 | 中 | 尝试全子带扩散，调整架构 |
| 高频细节丢失 | 低 | 中 | 保留并增强高频子带 |
| 训练不稳定 | 低 | 中 | 使用EMA，降低学习率 |

### 8.2 备选方案

如果方案A（仅LL扩散）效果不佳：

1. **方案B：全子带扩散**
   - 修改UNet输入通道为12
   - 同时处理LL和高频子带

2. **方案C：混合方案**
   - LL子带用扩散增强
   - 高频子带用轻量CNN增强

3. **方案D：DCT替代**
   - 如果DWT效果不理想，尝试DCT

---

## 9. 时间规划

```
┌─────────────────────────────────────────────────────────────────┐
│  周次  │  任务                                                   │
├─────────────────────────────────────────────────────────────────┤
│  1-2   │  数据集收集与预处理                                     │
│  3     │  DWT模块实现与单元测试                                  │
│  4     │  训练代码编写与调试                                     │
│  5-7   │  基础训练 (200K iterations)                            │
│  8-9   │  微调与消融实验                                         │
│  10    │  对比实验与结果分析                                     │
│  11    │  论文/报告撰写                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 检查清单

### 10.1 开始训练前

- [ ] 数据集下载完成
- [ ] LR-HR配对生成
- [ ] 数据加载器测试通过
- [ ] DWT模块单元测试通过
- [ ] 配置文件检查
- [ ] GPU环境确认

### 10.2 训练过程中

- [ ] 损失曲线正常下降
- [ ] 定期保存检查点
- [x] 验证集指标监控
- [x] 可视化中间结果

### 10.3 训练完成后

- [x] 最佳模型选择
- [ ] 测试集评估
- [ ] 与基线方法对比
- [ ] 推理速度测试
- [ ] 显存占用测试

---

## 11. 参考资料

1. DifIISR原论文: https://arxiv.org/abs/2503.01187
2. ResShift (基础框架): https://github.com/zsyOAOA/ResShift
3. PyWavelets文档: https://pywavelets.readthedocs.io/
4. Latent Diffusion Models: https://arxiv.org/abs/2112.10752

---

## 附录A: 环境配置

```bash
# 创建环境
conda create -n DifIISR-DWT python=3.10
conda activate DifIISR-DWT

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install PyWavelets
pip install omegaconf
pip install einops
pip install timm
pip install scikit-image
pip install tensorboard

# 验证安装
python -c "import pywt; print(pywt.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

## 附录B: 快速验证脚本

```python
# test_dwt_module.py
import torch
from ldm.models.dwt_autoencoder import DWTModel

# 创建模型
model = DWTModel(wavelet='haar', level=2)

# 测试编解码
x = torch.randn(1, 3, 256, 256)
z = model.encode(x)
x_rec = model.decode(z)

print(f"输入形状: {x.shape}")
print(f"潜空间形状: {z.shape}")
print(f"重建形状: {x_rec.shape}")
print(f"重建误差: {(x - x_rec).abs().max():.6f}")
```

---

*文档版本: v1.0*
*创建日期: 2025-01-26*
*作者: Claude Code Assistant*

---

## 12. 初版实验结果与问题分析 (2026-01-27更新)

### 12.1 实验结果

| 方法 | PSNR | SSIM | 视觉效果 |
|------|------|------|---------|
| DWT-haar (仅LL扩散) | **37.70 dB** | **0.9959** | 偏模糊 |
| VAE原版 | 较低 | 较低 | **更清晰** |

### 12.2 问题分析

**现象**：DWT版本指标更高，但视觉效果不如VAE版本清晰。

**根本原因**：
```
当前DWT流程:
  LR → DWT → LL(扩散增强) + HF(直接保留) → IDWT → SR
                              ↑
                    问题：LR的高频 ≠ HR的高频
                    LR高频本身就是残缺的，保留它无法提升清晰度
```

**本质区别**：
- **DWT**：保守策略，保留LR的"正确但模糊"的高频 → 指标高，视觉模糊
- **VAE**：生成策略，生成"可能有偏差但锐利"的高频 → 指标低，视觉清晰

**结论**：超分任务需要**生成**新的高频细节，而不是**保留**旧的。

---

## 13. 改进方案概览

### 13.1 方案对比

| 方案 | 思路 | 改动量 | 预期效果 | 推荐度 |
|------|------|--------|---------|--------|
| **A** | 全子带扩散 | 大 | 最好 | ⭐⭐⭐ |
| **B** | 高频CNN增强 | 中 | 较好 | ⭐⭐ |
| **C** | 感知损失 | 小 | 一般 | ⭐ |

### 13.2 方案A：全子带扩散（推荐）

**思路**：将所有DWT子带(LL+LH+HL+HH)拼接后送入扩散模型，让UNet同时增强低频和高频。

**架构变化**：
```
当前:
  LR(256) → DWT → LL(64×64×3) → UNet(in=6,out=3) → LL' → IDWT+HF → SR
                                                          ↑
                                                    HF未增强(问题)

改进:
  LR(256) → DWT → [LL,LH,HL,HH](64×64×12) → UNet(in=24,out=12) → [LL',LH',HL',HH'] → IDWT → SR
                                                                        ↑
                                                                  全部子带被增强
```

**代码改动**：
1. DWT编码器：返回全部子带拼接 (64×64×12)
2. UNet：输入通道 6→24，输出通道 3→12
3. DWT解码器：拆分子带后重建

**优缺点**：
- ✅ 高频子带被扩散模型增强
- ✅ 端到端训练，效果最优
- ❌ 需要修改UNet架构
- ❌ 需要重新训练

### 13.3 方案B：高频CNN增强

**思路**：低频用扩散增强，高频用轻量CNN增强，最后合并。

**架构变化**：
```
当前:
  LL → UNet → LL'
  HF → 直接保留 → HF (问题)

改进:
  LL → UNet → LL'
  HF → 轻量CNN → HF' (增强)
  合并: IDWT(LL', HF') → SR
```

**代码改动**：
1. 新增HF增强网络：轻量CNN（如3-5层卷积）
2. 修改解码流程：先增强HF再合并
3. 训练：可以冻结UNet，只训练CNN

**优缺点**：
- ✅ 不改变扩散模型架构
- ✅ 可以利用已训练的UNet
- ❌ 需要额外训练CNN
- ❌ 两阶段可能不如端到端

### 13.4 方案C：添加感知损失

**思路**：训练时加入LPIPS等感知损失，鼓励生成更锐利的结果。

**代码改动**：
```python
# 当前损失
loss = mse_loss

# 改进损失
loss = mse_loss + lambda_lpips * lpips_loss + lambda_edge * edge_loss
```

**优缺点**：
- ✅ 改动最小
- ✅ 不改变架构
- ❌ 效果有限（高频仍未被增强）
- ❌ 可能降低PSNR指标

---

## 14. 推荐实施路径

```
第一步：方案A（全子带扩散）
  ├── 改动DWT编解码器
  ├── 改动UNet通道数
  └── 重新训练

第二步：如果方案A效果不理想
  └── 尝试方案B（混合架构）

第三步：微调优化
  └── 加入感知损失（方案C）
```

---

## 15. 方案A详细设计

### 15.1 DWT编码器改动

```python
class DWTModelFullBand(nn.Module):
    def encode(self, x):
        """
        返回全部子带拼接
        输入: x [B, 3, 256, 256]
        输出: z [B, 12, 64, 64]
        """
        B, C, H, W = x.shape

        # 2级DWT分解
        coeffs = pywt.wavedec2(x, 'haar', level=2)
        LL = coeffs[0]           # 64×64×3
        LH2, HL2, HH2 = coeffs[1]  # 各64×64×3

        # 拼接: LL + LH + HL + HH = 64×64×12
        z = torch.cat([LL, LH2, HL2, HH2], dim=1)

        # 缓存level1高频用于解码
        self.hf_level1 = coeffs[2]

        return z  # [B, 12, 64, 64]

    def decode(self, z):
        """
        拆分子带后重建
        输入: z [B, 12, 64, 64]
        输出: x [B, 3, 256, 256]
        """
        # 拆分
        LL = z[:, 0:3]
        LH2 = z[:, 3:6]
        HL2 = z[:, 6:9]
        HH2 = z[:, 9:12]

        # 2级IDWT重建
        coeffs = [LL, (LH2, HL2, HH2), self.hf_level1]
        x = pywt.waverec2(coeffs, 'haar')

        return x
```

### 15.2 UNet配置改动

```yaml
model:
  target: models.unet.UNetModelSwin
  params:
    image_size: 64
    in_channels: 24    # 12(噪声子带) + 12(LQ条件) ← 原来是6
    out_channels: 12   # 输出全部子带 ← 原来是3
    # 其他参数不变
```

### 15.3 训练配置

```yaml
autoencoder:
  target: ldm.models.dwt_autoencoder.DWTModelFullBand
  params:
    wavelet: haar
    level: 2
    return_all_bands: True
```

---

## 16. 工作量估计

| 方案 | 代码改动文件 | 训练时间 | 总工作量 |
|------|-------------|---------|---------|
| A (全子带) | 3-4个 | 需重新训练 | 1-2周 |
| B (混合) | 2-3个 | 只训练CNN | 3-5天 |
| C (损失) | 1个 | 需重新训练 | 2-3天 |

---

## 17. 下一步行动

- [ ] 确认选择方案A/B/C
- [ ] 详细设计代码改动
- [ ] 实施并训练
- [ ] 评估对比

---

*文档版本: v1.1*
*更新日期: 2026-01-27*
*更新内容: 添加初版实验结果分析与改进方案*
