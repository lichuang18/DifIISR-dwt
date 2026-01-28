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

- [x] 确认选择方案A/B/C → **已选择方案A（全子带扩散）**
- [x] 详细设计代码改动 → **已完成**
- [ ] 实施并训练
- [ ] 评估对比

---

## 18. 方案A实现记录 (2026-01-27)

### 18.1 已完成的代码修改

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `ldm/models/dwt_autoencoder.py` | 新增类 | `DWTModelFullBand` - 全子带编解码器 |
| `ldm/models/dwt_autoencoder.py` | 新增类 | `DWTModelFullBandTorch` - PyTorch纯实现版本 |
| `configs/train_m3fd_dwt_fullband.yaml` | 新增 | 全子带扩散训练配置 |
| `models/gaussian_diffusion_test.py` | 修改 | `training_losses`: 用z_y替换model_kwargs['lq'] |
| `models/gaussian_diffusion_test.py` | 修改 | `p_sample_loop_progressive`: 同上 |
| `models/gaussian_diffusion_test.py` | 修改 | `ddim_sample_loop_progressive`: 同上 |
| `sampler.py` | 修改 | 支持DWT无checkpoint路径 |

### 18.2 关键改动详解

#### (1) DWTModelFullBand 编解码器

```python
# 编码: 256×256×3 → 64×64×12
def encode(self, x):
    coeffs = pywt.wavedec2(x, 'haar', level=2)
    # coeffs = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]

    LL2 = coeffs[0]           # 64×64×3
    LH2, HL2, HH2 = coeffs[1]  # 各64×64×3

    # 拼接4个子带 → 64×64×12
    z = concat([LL2, LH2, HL2, HH2], dim=channel)

    # 缓存level1高频用于解码
    self.hf_level1_cache = coeffs[2]  # (LH1, HL1, HH1)
    return z

# 解码: 64×64×12 → 256×256×3
def decode(self, z):
    # 拆分4个子带
    LL2, LH2, HL2, HH2 = split(z)

    # 使用缓存的level1高频重建
    coeffs = [LL2, (LH2, HL2, HH2), self.hf_level1_cache]
    return pywt.waverec2(coeffs, 'haar')
```

#### (2) UNet配置变化

```yaml
# 原配置 (仅LL扩散)
in_channels: 6      # 3(噪声LL) + 3(LQ的LL)
out_channels: 3     # 输出LL

# 新配置 (全子带扩散)
in_channels: 24     # 12(噪声全子带) + 12(LQ全子带)
out_channels: 12    # 输出全子带
```

#### (3) gaussian_diffusion_test.py 关键修改

```python
# 在 training_losses, p_sample_loop_progressive, ddim_sample_loop_progressive 中
# 编码LQ图像
z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)

# 关键：用编码后的z_y替换原始lq
if model_kwargs is not None and 'lq' in model_kwargs:
    model_kwargs = dict(model_kwargs)  # 复制避免修改原始
    model_kwargs['lq'] = z_y  # 现在是12通道
```

### 18.3 训练命令

```bash
conda activate sr
cd /home/lch/sr_recons/DifIISR-dwt

# 全子带扩散训练
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train_m3fd_dwt_fullband.yaml --seed 42
```

### 18.4 架构对比总结

| 项目 | 原方案(仅LL) | 方案A(全子带) |
|------|-------------|--------------|
| 潜空间维度 | 64×64×3 | 64×64×12 |
| UNet输入通道 | 6 | 24 |
| UNet输出通道 | 3 | 12 |
| 高频处理 | 直接保留(问题根源) | **扩散增强** |
| level1高频 | 缓存并保留 | 缓存并保留 |
| 预期效果 | PSNR高但模糊 | **更清晰** |

### 18.5 注意事项

1. **level1高频仍然保留**：只有level2的4个子带参与扩散，level1高频从LR缓存
2. **需要从头训练**：UNet架构改变，无法使用原有权重
3. **计算量增加**：输入输出通道增加4倍，但仍远低于VAE方案

### 18.6 潜空间变大的影响分析

#### (1) 计算量对比

| 组件 | 原VAE方案 (64×64×3) | 仅LL方案 (64×64×3) | 全子带方案 (64×64×12) |
|------|---------------------|--------------------|-----------------------|
| **编解码器** | ~115M MACs (VAE) | ~0.6M MACs (DWT) | ~2.4M MACs (DWT×4) |
| **UNet输入层** | 6→160通道 | 6→160通道 | 24→160通道 (×4) |
| **UNet中间层** | 不变 | 不变 | **不变** |
| **UNet输出层** | 160→3通道 | 160→3通道 | 160→12通道 (×4) |

**关键点**：UNet中间层（占计算量主体）不受影响，因为 `model_channels=160` 没变。只有首尾两层受影响。

#### (2) 实际增加量估算

```
UNet总参数约 118M（原方案）

输入层增加: (24-6) × 160 × 3 × 3 ≈ 26K 参数
输出层增加: 160 × (12-3) × 3 × 3 ≈ 13K 参数
总增加: ~39K 参数 (约 0.03%)

计算量增加也类似，约 1-2%
```

#### (3) 与VAE方案的总体对比

| 指标 | VAE方案 | 全子带DWT方案 | 对比 |
|------|---------|---------------|------|
| 编解码参数 | 49M | 0 | **-100%** |
| 编解码计算量 | 115M MACs | 2.4M MACs | **-98%** |
| UNet参数 | 118M | 118.04M | +0.03% |
| 总计算量 | ~233M MACs | ~120M MACs | **-48%** |

#### (4) 潜在问题

| 问题 | 严重程度 | 说明 |
|------|----------|------|
| 显存占用 | 低 | 潜空间×4，但远小于VAE节省的显存 |
| 训练难度 | 中 | 需要学习12通道而非3通道，可能需要更多迭代 |
| 收敛速度 | 中 | 信息量增加，可能需要调整学习率 |

#### (5) 结论

**计算量增加很小**（约1-2%），远小于去掉VAE带来的收益（-98%编解码计算量）。

主要代价是：
- 训练可能需要更多迭代（信息量增加）
- 可能需要微调学习率

#### (6) 可能的折中方案

如果全子带效果不理想，可考虑：
- **LL+HH方案**：只扩散 LL+HH（对角高频最重要），变成 64×64×6
- **加权方案**：对不同子带使用不同权重的损失函数

---

## 19. 方案A实验结果与问题分析 (2026-01-27)

### 19.1 实验结果

| 配置 | 迭代次数 | PSNR | SSIM | 视觉效果 |
|------|----------|------|------|----------|
| 全子带(level2) | 10000 | 较高 | 较高 | **仍然模糊** |

### 19.2 问题分析

#### 根本原因：level1高频仍未增强

```
当前方案A的流程:
  LR(256) → DWT 2级分解
           ├── level2: LL2, LH2, HL2, HH2 (64×64) → 扩散增强 ✓
           └── level1: LH1, HL1, HH1 (128×128) → 直接保留 ✗ ← 问题在这里！

level1高频占据128×128分辨率，包含大量边缘和纹理信息
这些高频仍然来自LR，未被增强，导致视觉效果模糊
```

#### 快速验证：后处理锐化

尝试了USM锐化和高通滤波锐化，效果略有改善但不理想。
说明问题确实是高频信息不足，但后处理无法真正"生成"缺失的高频。

### 19.3 改进方案

| 方案 | 改动 | 预期效果 | 代价 |
|------|------|----------|------|
| **增加迭代** | 训练50K-100K次 | 可能改善 | 时间 |
| **添加感知损失** | 加入LPIPS/VGG | 提升视觉质量 | 小改动 |
| **level1也扩散** | 潜空间包含level1 | **最彻底** | 架构改动 |

### 19.4 决定采用的方案

**综合方案：level1参与扩散 + 感知损失**

#### (1) 新的DWT编解码架构

```
改进后的流程:
  LR(256) → DWT 2级分解
           ├── level2: LL2, LH2, HL2, HH2 (64×64×12)  ─┐
           └── level1: LH1, HL1, HH1 (128×128×9)      ─┼→ 全部参与扩散
                                                       │
  潜空间总维度: 64×64×12 + 128×128×9 (需要特殊处理)    ←┘
```

#### (2) 架构设计选项

**选项A：多尺度UNet**
- level2子带用64×64分支
- level1子带用128×128分支
- 两个分支共享部分权重

**选项B：统一下采样**
- 将level1子带下采样到64×64
- 拼接成 64×64×21 (12+9)
- 解码时上采样回128×128

**选项C：级联扩散**
- 先扩散level2 (64×64×12)
- 再扩散level1 (128×128×9)，以level2结果为条件

#### (3) 感知损失

```python
# 训练损失
loss = mse_loss + lambda_lpips * lpips_loss + lambda_vgg * vgg_loss

# 推荐权重
lambda_lpips = 0.1
lambda_vgg = 0.01
```

---

## 20. AllBands方案实验结果与深入分析 (2026-01-27)

### 20.1 实验结果

| 配置 | 迭代次数 | PSNR | SSIM | 视觉效果 |
|------|----------|------|------|----------|
| AllBands (level1+level2) + LPIPS | 8000 | - | - | **仍不如DifIISR清晰** |

### 20.2 问题分析

#### (1) 训练次数不足

| 版本 | 通道数 | 相对复杂度 | 建议迭代次数 |
|------|--------|-----------|-------------|
| 原VAE | 3 | 1× | 100K |
| FullBand | 12 | 4× | 100K+ |
| AllBands | 21 | 7× | 150K+ |

21通道需要学习的东西是原来的7倍，8000次远远不够。

#### (2) level1下采样/上采样的信息损失

```
当前流程:
  level1 (128×128) → cv2.resize下采样 → 64×64 → 扩散 → cv2.resize上采样 → 128×128
                            ↑                              ↑
                      信息损失1                        信息损失2
```

cv2.resize是固定的插值算法：
- 下采样：简单平均，高频信息直接丢失
- 上采样：插值填充，无法恢复丢失的信息

#### (3) VAE vs DWT下采样的本质区别

**VAE的下采样（可学习）：**
```
VAE Encoder:
  256×256 → Conv+下采样 → 128×128 → Conv+下采样 → 64×64
                ↓                      ↓
           学习的卷积核            学习的卷积核
           (保留重要特征)          (保留重要特征)

VAE Decoder:
  64×64 → ConvTranspose+上采样 → 128×128 → ConvTranspose+上采样 → 256×256
                ↓                              ↓
           学习的卷积核                    学习的卷积核
           (生成合理细节)                  (生成合理细节)
```

**DWT+resize的下采样（固定）：**
```
DWT level1处理:
  128×128 → cv2.resize(INTER_AREA) → 64×64 → 扩散 → cv2.resize(INTER_CUBIC) → 128×128
                ↓                                          ↓
           固定的平均池化                              固定的插值
           (无差别丢弃高频)                            (无法恢复丢失信息)
```

**核心区别：**

| 特性 | VAE | DWT+resize |
|------|-----|------------|
| 下采样方式 | 学习的卷积 | 固定插值 |
| 上采样方式 | 学习的反卷积 | 固定插值 |
| 信息保留 | 学会保留重要特征 | 无差别丢弃 |
| 细节恢复 | 可以"生成"合理细节 | 只能插值模糊 |
| 参数量 | 49M可学习参数 | 0参数 |

**结论**：VAE确实也有信息损失，但它通过**学习**来最小化重要信息的损失，并在解码时生成合理的细节补偿。而cv2.resize是"无脑"的固定操作。

#### (4) DWT的根本局限性

```
VAE: 有生成能力，decoder可以"幻想"出合理的高频细节
DWT: 确定性变换，只能增强已有信息，无法凭空生成
```

### 20.3 可能的改进方向

| 方案 | 改动 | 预期效果 | 代价 |
|------|------|----------|------|
| **增加训练次数** | 50K-100K | 基础要求 | 时间 |
| **调大LPIPS权重** | 0.1 → 0.5 | 更注重视觉质量 | 可能降PSNR |
| **可学习的上下采样** | Conv代替resize | 减少信息损失 | 增加参数 |
| **多尺度UNet** | 不下采样level1 | 避免信息损失 | 架构大改 |
| **加入GAN损失** | 对抗训练 | 生成更锐利细节 | 训练复杂 |
| **混合方案** | DWT编码 + CNN解码器 | 结合两者优点 | 架构重设计 |

### 20.4 改进方案详细设计

#### 方案A：可学习的上下采样

```python
class DWTModelAllBandsLearnable(nn.Module):
    def __init__(self):
        # 用卷积代替cv2.resize
        self.down = nn.Conv2d(9, 9, 3, stride=2, padding=1)  # 128→64
        self.up = nn.ConvTranspose2d(9, 9, 4, stride=2, padding=1)  # 64→128

    def encode(self, x):
        # ... DWT分解 ...
        level1_down = self.down(level1_bands)  # 可学习的下采样
        return concat(level2_bands, level1_down)

    def decode(self, z):
        # ... 拆分 ...
        level1_up = self.up(level1_down)  # 可学习的上采样
        # ... IDWT重建 ...
```

#### 方案B：多尺度UNet（不下采样）

```
level2 (64×64×12) → UNet分支1 (小尺度)
                         ↓ 交互
level1 (128×128×9) → UNet分支2 (大尺度)
                         ↓
                    融合输出
```

#### 方案C：DWT编码 + 轻量生成式解码器

```
编码: DWT (确定性，高效)
解码: 轻量CNN解码器 (可学习，有生成能力)

这样既保留DWT编码的高效性，又获得生成式解码的细节恢复能力
```

### 20.5 下一步建议

1. **短期**：增加训练次数到50K+，验证是否有改善
2. **中期**：实现可学习的上下采样（方案A），减少信息损失
3. **长期**：考虑混合架构（方案C），结合DWT和生成式解码的优点

---

## 21. 方案深度分析与最终设计 (2026-01-27)

### 21.1 为什么需要下采样？—— VAE/DWT的本质作用

```
原图 256×256×3 → 潜空间 64×64×3 → 扩散 → 潜空间 64×64×3 → 重建 256×256×3
                      ↑
                 缩小4倍
                 扩散计算量降低16倍！
```

**核心认识：VAE的核心价值不是"学习特征"，而是"降低扩散计算量"**

### 21.2 方案B（多尺度UNet）的致命问题

```
方案B设想:
  level2: 64×64×12   → UNet分支1 (正常)
  level1: 128×128×9  → UNet分支2 (计算量爆炸！)

UNet计算量与空间尺寸的关系:
  64×64   → 1×
  128×128 → 4×
  256×256 → 16×
```

| 分支 | 尺寸 | 相对计算量 |
|------|------|-----------|
| level2分支 | 64×64 | 1× (~500M MACs) |
| level1分支 | 128×128 | 4× (~2000M MACs) |
| **总计** | - | **~2500M MACs** |

**结论：方案B计算量是原VAE方案的4倍，完全不可行！**

### 21.3 各方案综合对比

#### 参数量对比

```
原VAE编解码器: ~49M参数
DWT编解码器: 0参数
方案A额外参数:
  down_conv: 9 × 9 × 3 × 3 = 729
  up_convT: 9 × 9 × 4 × 4 = 1296
  总计: ~2K (可忽略)
```

| 方案 | 编解码参数 | UNet参数 | 总参数 | 相对VAE |
|------|-----------|----------|--------|---------|
| 原VAE | 49M | 118M | 167M | 基准 |
| DWT-LL | 0 | 118M | 118M | -29% |
| DWT-FullBand | 0 | 118M | 118M | -29% |
| DWT-AllBands | 0 | 118.7M | 118.7M | -29% |
| **方案A** | **2K** | **118.7M** | **118.7M** | **-29%** |
| ~~方案B~~ | 0 | 150-170M | 150-170M | 不可行 |

#### 计算量对比 (MACs)

```
原VAE编解码: ~115M MACs
DWT编解码: ~1.2M MACs
方案A额外计算:
  down_conv (128→64): 9×9×3×3×64×64 = ~2.7M MACs
  up_convT (64→128): 9×9×4×4×128×128 = ~19M MACs
  总计: ~22M MACs
```

| 方案 | 编解码MACs | UNet MACs | 总MACs | 相对VAE |
|------|-----------|-----------|--------|---------|
| 原VAE | 115M | 500M | 615M | 基准 |
| DWT-LL | 1.2M | 500M | 501M | -19% |
| DWT-FullBand | 1.2M | 500M | 501M | -19% |
| DWT-AllBands | 1.2M | 520M | 521M | -15% |
| **方案A** | **23M** | **520M** | **543M** | **-12%** |
| ~~方案B~~ | 1.2M | 2500M | 2500M | +306% ✗ |

#### 综合对比表

| 指标 | 原VAE | DWT-AllBands | 方案A |
|------|-------|--------------|-------|
| 编解码参数 | 49M | 0 | 2K |
| 总参数 | 167M | 118.7M | 118.7M |
| 编解码MACs | 115M | 1.2M | 23M |
| 总MACs | 615M | 521M | 543M |
| 需要预训练权重 | 是(196MB) | 否 | 否 |
| level1信息损失 | 有(可学习补偿) | 有(固定resize) | 有(可学习减少) |
| 实现难度 | - | 已完成 | 低 |
| 预期视觉质量 | 高 | 低 | 中 |

### 21.4 方案A详细设计：可学习上下采样

#### 核心改动

```python
class DWTModelAllBandsLearnable(nn.Module):
    """
    改进版AllBands：用可学习Conv替代cv2.resize

    编码: 256×256×3 → 64×64×21
      - level2: LL2, LH2, HL2, HH2 (64×64×12) - DWT直接得到
      - level1: LH1, HL1, HH1 (128×128×9) → Conv下采样 → 64×64×9

    解码: 64×64×21 → 256×256×3
      - level1: 64×64×9 → ConvTranspose上采样 → 128×128×9
      - IDWT重建
    """
    def __init__(self, wavelet='haar', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

        # 可学习的下采样：128→64
        # 9通道 = 3个RGB通道 × 3个高频子带(LH1, HL1, HH1)
        self.down = nn.Conv2d(9, 9, kernel_size=3, stride=2, padding=1)

        # 可学习的上采样：64→128
        self.up = nn.ConvTranspose2d(9, 9, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        # DWT分解得到level1和level2子带
        # ...

        # level1子带：可学习下采样
        level1_bands = torch.cat([LH1, HL1, HH1], dim=1)  # [B, 9, 128, 128]
        level1_down = self.down(level1_bands)  # [B, 9, 64, 64]

        # 拼接
        z = torch.cat([level2_bands, level1_down], dim=1)  # [B, 21, 64, 64]
        return z

    def decode(self, z):
        # 拆分
        level2_bands = z[:, :12]  # [B, 12, 64, 64]
        level1_down = z[:, 12:]   # [B, 9, 64, 64]

        # level1子带：可学习上采样
        level1_bands = self.up(level1_down)  # [B, 9, 128, 128]

        # IDWT重建
        # ...
```

#### 与cv2.resize的对比

| 特性 | cv2.resize | 可学习Conv |
|------|------------|-----------|
| 下采样方式 | 固定INTER_AREA | 学习的3×3卷积 |
| 上采样方式 | 固定INTER_CUBIC | 学习的4×4反卷积 |
| 参数量 | 0 | ~2K |
| 计算量 | ~1M MACs | ~22M MACs |
| 信息保留 | 无差别丢弃 | 学习保留重要信息 |
| 细节恢复 | 固定插值 | 学习生成合理细节 |

#### 训练策略

```yaml
# 配置建议
train:
  iterations: 50000-100000  # 需要足够迭代让Conv学习
  lr: 5e-5

perceptual_loss:
  enabled: True
  lpips_weight: 0.1-0.3  # 可适当调大
```

### 21.5 最终结论

1. **方案B不可行**：128×128分支计算量爆炸，违背DWT替代VAE的初衷
2. **方案A是唯一合理改进**：用可学习Conv替代cv2.resize，代价极小
3. **预期效果**：
   - 可学习下采样能保留更多重要的高频信息
   - 可学习上采样能生成更合理的细节
   - 视觉质量应该介于当前AllBands和原VAE之间

### 21.6 下一步行动

- [x] 实现 `DWTModelAllBandsLearnable` 类
- [x] 创建新配置文件
- [ ] 训练50K+迭代
- [ ] 对比评估

---

## 22. 纯PyTorch版本实现与LPIPS梯度修复 (2026-01-28)

### 22.1 问题发现

使用 `DWTModelAllBandsLearnable` 训练时，LPIPS 损失始终不下降（维持在 0.7+），经检查发现**梯度链断开**。

### 22.2 梯度断开的三处位置

| 位置 | 文件:行号 | 问题代码 | 影响 |
|------|----------|---------|------|
| 1 | `dwt_autoencoder.py` decode() | `.detach().cpu().numpy()` | decode输出无梯度 |
| 2 | `gaussian_diffusion_test.py:969` | `pred_zstart = model_output.detach()` | UNet输出被detach |
| 3 | `train.py:238` | `decode_first_stage()` 默认 `no_grad=True` | decode在no_grad中 |

### 22.3 解决方案

#### (1) 新增 `DWTModelAllBandsLearnableTorch` 类

decode 使用纯 PyTorch 实现的 IDWT，不经过 numpy：

```python
class DWTModelAllBandsLearnableTorch(nn.Module):
    def decode(self, z):
        # 全程PyTorch操作，梯度可回传
        level1_bands = self.up(level1_down)  # ConvTranspose2d
        LL1 = self.idwt(LL2, (LH2, HL2, HH2))  # 纯PyTorch IDWT
        x = self.idwt(LL1, (LH1, HL1, HH1))
        return x  # 有梯度
```

#### (2) 修改 `gaussian_diffusion_test.py`

```python
# 原代码
pred_zstart = model_output.detach()

# 修改后
pred_zstart = model_output  # 移除 .detach()
```

#### (3) 修改 `train.py`

```python
# 原代码
pred_img = diffusion.decode_first_stage(pred_zstart, autoencoder)

# 修改后
pred_img = diffusion.decode_first_stage(pred_zstart, autoencoder, no_grad=False)
```

#### (4) 修改 `decode_first_stage` 函数

```python
# 原代码
out = first_stage_model.decode(z_sample, grad_forward=True)

# 修改后（移除不存在的参数）
out = first_stage_model.decode(z_sample)
```

### 22.4 修复后的梯度链路

```
UNet输出 model_output (有梯度)
    ↓
pred_zstart = model_output (不再detach)
    ↓
decode_first_stage(no_grad=False)
    ↓
DWTModelAllBandsLearnableTorch.decode()
    ├── self.up() - ConvTranspose2d (有梯度)
    └── self.idwt() - 纯PyTorch (有梯度)
    ↓
pred_img (有梯度)
    ↓
lpips_fn(pred_img, gt)
    ↓
loss.backward() → 梯度回传到 UNet ✓
```

### 22.5 新增文件清单

| 文件 | 说明 |
|------|------|
| `ldm/models/dwt_autoencoder.py` | 新增 `DWTModelAllBandsLearnableTorch` 类 |
| `configs/train_m3fd_dwt_learnable_torch.yaml` | 纯PyTorch版本配置 |
| `test_gradient.py` | 梯度链路验证脚本 |

### 22.6 训练结果 (2600次迭代)

| 指标 | 初始 (100次) | 2600次 | 变化 |
|------|-------------|--------|------|
| **MSE** | 0.0179 | 0.0036 | **↓ 80%** ✓ |
| **LPIPS** | 0.7176 | 0.6951 | ↓ 3% (波动大) |
| **Total Loss** | 0.0897 | 0.0731 | ↓ 18% |

### 22.7 LPIPS 波动大的原因分析

扩散模型训练时，时间步 `t` 随机采样（0到14）：

| t 值 | pred_zstart 质量 | LPIPS |
|------|-----------------|-------|
| t ≈ 0 | 接近真实图像 | 低 |
| t ≈ 14 | 从噪声预测，质量差 | 高 |

每次迭代 `t` 随机变化，导致 LPIPS 波动剧烈，但**长期趋势是下降的**。

### 22.8 结论

1. **MSE 明显下降** → 模型在正常学习
2. **LPIPS 有下降趋势** → 梯度回传正常工作
3. **波动是正常现象** → 由随机时间步 `t` 导致
4. **建议继续训练** → 50000+ 次迭代后评估效果

### 22.9 版本对比

| 版本 | decode实现 | LPIPS梯度 | 使用场景 |
|------|-----------|----------|---------|
| `DWTModelAllBandsLearnable` | pywt (numpy) | ❌ 断开 | 不使用LPIPS |
| `DWTModelAllBandsLearnableTorch` | 纯PyTorch | ✅ 正常 | **使用LPIPS** |

### 22.10 训练命令

```bash
cd /home/lch/sr_recons/DifIISR-dwt
conda activate sr

# 验证梯度
python test_gradient.py

# 开始训练
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/train_m3fd_dwt_learnable_torch.yaml \
    --seed 42
```

---

## 23. 动态LPIPS权重策略 (2026-01-28)

### 23.1 问题背景

LPIPS损失在训练过程中波动剧烈，难以收敛。原因是扩散模型的时间步 `t` 随机采样：

| t 范围 | pred_zstart 质量 | LPIPS 信号 |
|--------|-----------------|------------|
| t ≈ 0 (低噪声) | 接近真实图像 | **稳定、有意义** |
| t ≈ T/2 (中等噪声) | 质量中等 | 有一定噪声 |
| t ≈ T (高噪声) | 从纯噪声预测，质量差 | **噪声大、不稳定** |

### 23.2 解决方案：基于时间步的动态权重

根据时间步 `t` 动态调整LPIPS权重：

```
T = 15 (总时间步数)
t1 = T // 3 = 5
t2 = 2 * T // 3 = 10

权重策略:
┌─────────────────┬─────────────┬─────────────────────────────┐
│   时间步范围     │  LPIPS权重   │  原因                        │
├─────────────────┼─────────────┼─────────────────────────────┤
│ t < 5  (0-4)    │    1.0      │ pred质量好，信号稳定          │
│ 5 ≤ t < 10      │    0.1      │ pred质量中等，适度使用        │
│ t ≥ 10 (10-14)  │    0.001    │ pred质量差，几乎忽略          │
└─────────────────┴─────────────┴─────────────────────────────┘
```

### 23.3 代码实现

```python
# train.py train_one_step() 函数

# 时间步阈值
T = diffusion.num_timesteps  # 15
t1 = T // 3      # 5
t2 = 2 * T // 3  # 10

# 计算每个样本的动态权重
weights = torch.ones(t.shape[0], device=device)
weights[(t >= t1) & (t < t2)] = 0.1
weights[t >= t2] = 0.001

# 计算每个样本的LPIPS
lpips_per_sample = lpips_fn(pred_img, gt).view(-1)  # [B]

# 加权平均
weighted_lpips = (lpips_per_sample * weights).mean()
loss = loss + base_weight * weighted_lpips
```

### 23.4 预期效果

| 改进点 | 说明 |
|--------|------|
| **减少波动** | 高噪声时间步的LPIPS贡献被大幅降低 |
| **稳定收敛** | 主要学习低噪声时的感知质量 |
| **保持梯度** | 所有样本都参与计算，梯度更稳定 |

### 23.5 与之前方案的对比

| 方案 | 做法 | 问题 |
|------|------|------|
| **旧方案** | 只在 t < T/3 时计算LPIPS，其他时间步跳过 | 2/3的batch没有LPIPS梯度 |
| **新方案** | 所有时间步都计算，但用不同权重 | 更平滑的梯度信号 |

### 23.6 训练日志分析与lpips_weight调整 (2026-01-28)

#### 问题发现

训练5500次迭代后，日志显示严重问题：

```
典型日志:
Iter 2600: loss=0.2890, mse=0.0136, lpips=0.5507
Iter 2900: loss=0.0356, mse=0.0091, lpips=0.0530
Iter 4900: loss=0.0270, mse=0.0045, lpips=0.0451

验证结果:
Validation PSNR: 9.65 dB, SSIM: 0.2881  ← 极低！正常应25-35dB
```

| 指标 | 最小值 | 最大值 | 波动倍数 |
|------|--------|--------|----------|
| loss | 0.0270 | 0.2890 | **10.7x** |
| mse | 0.0045 | 0.0261 | 5.8x |
| lpips | 0.0451 | 0.5507 | **12.2x** |

#### 原因分析

**lpips_weight=0.5 太大，LPIPS完全主导了loss：**

```
loss = mse + 0.5 * lpips

典型贡献:
- mse贡献: 0.01
- lpips贡献: 0.5 * 0.3 = 0.15

→ LPIPS贡献是MSE的15倍！
→ MSE无法正常收敛
→ PSNR只有9.65dB
```

#### 动态权重后的LPIPS贡献计算

假设 batch_size=8，时间步均匀分布：

| 时间步区间 | 样本数 | 动态权重 | 典型LPIPS | 加权贡献 |
|-----------|--------|----------|-----------|----------|
| t ∈ [0,4] | ~2.67 | 1.0 | 0.07 | 0.07 |
| t ∈ [5,9] | ~2.67 | 0.1 | 0.22 | 0.022 |
| t ∈ [10,14] | ~2.67 | 0.001 | 0.47 | 0.0005 |

```
weighted_lpips ≈ mean(0.07, 0.022, 0.0005) ≈ 0.031
```

#### 不同 base_weight 的效果对比

| base_weight | LPIPS贡献 | MSE贡献 | LPIPS/MSE | 效果 |
|-------------|-----------|---------|-----------|------|
| **0.1** | 0.003 | 0.01 | **0.3** | **MSE主导，LPIPS辅助（推荐）** |
| 0.2 | 0.006 | 0.01 | 0.6 | 接近平衡 |
| 0.3 | 0.009 | 0.01 | 0.9 | 接近平衡 |
| 0.5 | 0.015 | 0.01 | 1.5 | LPIPS主导（问题配置） |

#### 最终配置

```yaml
# configs/train_m3fd_dwt_learnable_torch.yaml
perceptual_loss:
  enabled: True
  lpips_weight: 0.1    # 从0.5降到0.1
  lpips_net: 'vgg'
```

#### 选择 lpips_weight=0.1 的理由

1. **首要目标是让MSE收敛**
   - 之前PSNR只有9.65dB，模型基本没学好
   - 需要MSE主导，先学会基本重建

2. **动态权重已经做了主要工作**
   - 高噪声时间步(t≥10)的LPIPS被压制1000倍
   - 主要是低噪声时间步的稳定LPIPS在起作用

3. **LPIPS/MSE ≈ 0.3 是合理的辅助比例**
   - LPIPS不会干扰MSE学习
   - 但仍能提供感知质量引导

#### 预期改善

| 指标 | 旧配置 (weight=0.5) | 新配置 (weight=0.1) |
|------|---------------------|---------------------|
| Loss波动 | 10x | **2~3x** |
| MSE收敛 | 被LPIPS干扰 | **正常收敛** |
| 验证PSNR | 9.65dB | **预期25+dB** |
| 视觉质量 | 不稳定 | 稳定提升 |

#### 后续调整建议

如果训练稳定后（PSNR达到25dB+）想增强视觉质量：
- 可调高到 `lpips_weight=0.2`（LPIPS/MSE ≈ 0.6）
- 或在后期微调时调高

---

## 24. 渐进式LPIPS权重调度策略 (2026-01-28)

### 24.1 问题背景

即使使用了动态时间步权重和lpips_weight=0.1，LPIPS贡献仍是MSE的3-6倍：

```
Iter 400: loss=0.0432, mse=0.0065, lpips=0.3667
  → LPIPS贡献: 0.1 * 0.3667 = 0.0367
  → MSE贡献:   0.0065
  → LPIPS/MSE = 5.6倍  ← LPIPS仍然主导
```

### 24.2 解决方案：渐进式权重调度

**核心思想**：训练初期让MSE主导学习基本重建，后期逐渐增大LPIPS权重提升视觉质量。

```
权重调度公式:
  weight = start + (end - start) * (iteration / total_iterations)

配置:
  lpips_weight_start: 0.01  (训练初期)
  lpips_weight_end:   0.15  (训练后期)
  total_iterations:   100000
```

### 24.3 权重变化曲线

| 迭代次数 | lpips_weight | 典型LPIPS | LPIPS贡献 | 典型MSE | LPIPS/MSE |
|----------|--------------|-----------|-----------|---------|-----------|
| 0K | 0.01 | 0.03 | 0.0003 | 0.010 | **1:33** (MSE主导) |
| 10K | 0.024 | 0.03 | 0.0007 | 0.008 | 1:11 |
| 20K | 0.038 | 0.03 | 0.0011 | 0.007 | 1:6 |
| 30K | 0.052 | 0.03 | 0.0016 | 0.006 | 1:4 |
| 40K | 0.066 | 0.03 | 0.0020 | 0.006 | 1:3 |
| 50K | 0.080 | 0.03 | 0.0024 | 0.005 | 1:2 |
| 60K | 0.094 | 0.03 | 0.0028 | 0.005 | 1:1.8 |
| 70K | 0.108 | 0.03 | 0.0032 | 0.005 | 1:1.5 |
| 80K | 0.122 | 0.03 | 0.0037 | 0.005 | 1:1.3 |
| 90K | 0.136 | 0.03 | 0.0041 | 0.005 | 1:1.2 |
| 100K | 0.150 | 0.03 | 0.0045 | 0.005 | **1:1** (接近等权) |

*注：典型LPIPS=0.03是经过时间步动态权重加权后的值*

### 24.4 代码实现

#### (1) 新增函数 `get_lpips_weight_schedule()`

```python
def get_lpips_weight_schedule(iteration, total_iterations, configs):
    """渐进式LPIPS权重调度"""
    lpips_config = configs.get('perceptual_loss', {})
    start_weight = lpips_config.get('lpips_weight_start', 0.01)
    end_weight = lpips_config.get('lpips_weight_end', 0.15)

    # 线性插值
    progress = min(iteration / total_iterations, 1.0)
    weight = start_weight + (end_weight - start_weight) * progress

    return weight
```

#### (2) 修改 `train_one_step()` 函数

```python
def train_one_step(..., iteration=0, total_iterations=100000):
    # 获取渐进式权重
    base_weight = get_lpips_weight_schedule(iteration, total_iterations, configs)

    # ... 时间步动态权重计算 ...

    loss = loss + base_weight * weighted_lpips
```

#### (3) 配置文件

```yaml
perceptual_loss:
  enabled: True
  lpips_weight_start: 0.01   # 起始权重
  lpips_weight_end: 0.15     # 结束权重
  lpips_net: 'vgg'
```

### 24.5 日志输出

训练日志现在会显示当前的lpips_weight：

```
Iter 1000:  loss=0.0320, mse=0.0050, lpips=0.2800, lpips_w=0.0114, lr=5.00e-05
Iter 50000: loss=0.0280, mse=0.0050, lpips=0.2800, lpips_w=0.0800, lr=2.50e-05
```

### 24.6 设计优势

| 优势 | 说明 |
|------|------|
| **前期稳定** | MSE主导，模型先学会基本重建 |
| **后期提升** | LPIPS逐渐增大，提升视觉质量 |
| **平滑过渡** | 线性插值，避免突变 |
| **可配置** | start/end权重可调整 |

### 24.7 与其他策略的组合

最终的LPIPS权重 = 渐进式权重 × 时间步动态权重

```
例如 iteration=50000, t=3:
  渐进式权重 = 0.08
  时间步权重 = 1.0 (t<5)
  最终权重 = 0.08 × 1.0 = 0.08

例如 iteration=50000, t=12:
  渐进式权重 = 0.08
  时间步权重 = 0.001 (t>=10)
  最终权重 = 0.08 × 0.001 = 0.00008
```

---

*文档版本: v1.9*
*更新日期: 2026-01-28*
*更新内容: 渐进式LPIPS权重调度策略*
