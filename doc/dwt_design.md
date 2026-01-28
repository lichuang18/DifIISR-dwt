# DifIISR-DWT 设计文档

## 概述

本项目将 DifIISR 中的 VQ-VAE 编解码器替换为基于离散小波变换 (DWT) 的方案，目标是降低计算量和参数量，同时保持视觉质量。

---

## 一、项目背景

### 1.1 原版 DifIISR 架构

```
LR → Bicubic↑ → VAE.encode(49M参数) → 潜空间扩散 → VAE.decode → SR
```

### 1.2 DWT 替代方案的动机

| 指标 | VAE | DWT | 收益 |
|------|-----|-----|------|
| 编解码参数 | 49M | 0~2.5M | -95%~100% |
| 编解码计算量 | 115M MACs | 0.6~25M MACs | -78%~99% |
| 权重文件 | 196MB | 0 | -100% |
| 推理显存 | +200MB | +10MB | -95% |

---

## 二、方案演进与失败分析

### 2.1 方案演进总览

| 版本 | 方案 | 结果 | 问题 |
|------|------|------|------|
| v1 | 仅LL子带扩散 | PSNR高，视觉模糊 | 高频未增强 |
| v2 | 全level2子带扩散 | 仍然模糊 | level1高频未增强 |
| v3 | AllBands (level1+2) + cv2.resize | 仍然模糊 | 固定resize信息损失 |
| v4 | AllBands + 可学习Conv | 仍然模糊 | IDWT无生成能力 |
| **v5** | **AllBands + LPIPS/Edge损失** | **待验证** | 当前方案A |
| **v6** | **DWT编码 + CNN解码** | **待验证** | 当前方案B |

### 2.2 核心问题分析

**根本矛盾**：DWT的IDWT是数学上的精确逆变换，无法"生成"新的高频细节。

```
VAE Decoder: 生成式模型，可以"幻想"出合理的高频细节
DWT IDWT:    确定性逆变换，只能恢复已有信息，无法凭空生成
```

**表现**：
- PSNR/SSIM 指标高（像素准确）
- LPIPS 指标好（结构正确）
- 视觉效果模糊（缺少锐利的高频细节）

### 2.3 各版本失败原因详解

| 版本 | 失败原因 | 教训 |
|------|----------|------|
| v1 | 只扩散LL，高频直接从LR保留 | LR的高频本身就是残缺的 |
| v2 | level1高频(128×128)未参与扩散 | level1包含大量边缘纹理信息 |
| v3 | cv2.resize下采样丢失高频信息 | 固定算法无差别丢弃 |
| v4 | 可学习Conv只能减少损失，不能生成 | IDWT本身无生成能力 |

### 2.4 关键发现

1. **LPIPS梯度断开问题**：使用pywt(numpy)会断开梯度链，需用纯PyTorch实现
2. **损失权重平衡**：LPIPS值(~0.2)比MSE(~0.007)大一个数量级，需渐进式调度
3. **时间步加权**：高噪声时间步的pred_zstart质量差，LPIPS信号噪声大

---

## 三、当前推荐方案

### 方案A：DWT+IDWT + 增强损失函数

**思路**：保持DWT架构，通过损失函数引导模型学习生成更锐利的结果。

**架构**：
```
LR(256) → DWT 2级分解
         ├── level2: LL2, LH2, HL2, HH2 (64×64×12)
         └── level1: LH1, HL1, HH1 (128×128×9) → Conv下采样 → 64×64×9
                                  ↓
         拼接: 64×64×21
                                  ↓
         UNet(in=42, out=21) → 扩散增强
                                  ↓
         拆分 + ConvTranspose上采样
                                  ↓
         IDWT重建 → SR(256×256×3)
```

**损失函数**：
```
Total Loss = MSE + λ_lpips * LPIPS + λ_edge * Edge

- MSE: 1.0 (基础，保证像素准确)
- LPIPS: 0.02 → 0.1 (渐进式，保证感知质量)
- Edge: 0.1 (Sobel边缘损失，增强锐度)
```

**配置文件**：`configs/train_m3fd_dwt_enhanced.yaml`

**参数量**：~243 (仅上下采样Conv)

**优点**：
- 参数量极少
- 计算量低
- PSNR/SSIM指标高

**缺点**：
- IDWT无生成能力，视觉清晰度受限
- 需要长时间训练才能看到效果

---

### 方案B：DWT编码 + CNN解码（混合架构）

**思路**：保留DWT编码的高效性，用可学习的CNN decoder替代IDWT，获得生成能力。

**架构**：
```
编码 (DWT，无参数):
  LR(256) → DWT 2级分解 → 64×64×21

解码 (CNN，~2.5M参数):
  64×64×21 → 特征提取(128ch) → 残差块×3
           → PixelShuffle上采样 → 128×128×64 → 残差块×2
           → PixelShuffle上采样 → 256×256×32 → 残差块×1
           → 输出层 → 256×256×3
```

**CNN Decoder结构**：
```python
class CNNDecoder(nn.Module):
    # 64×64×21 → 64×64×128 (特征提取 + 3个残差块)
    # 64×64×128 → 128×128×64 (PixelShuffle + 2个残差块)
    # 128×128×64 → 256×256×32 (PixelShuffle + 1个残差块)
    # 256×256×32 → 256×256×3 (输出层 + Tanh)
```

**配置文件**：`configs/train_m3fd_dwt_cnn_decoder.yaml`

**参数量对比**：

| 方案 | 编码 | 解码 | 总参数 |
|------|------|------|--------|
| VAE | 24.5M | 24.5M | 49M |
| 方案A (DWT+IDWT) | 0 | 243 | 243 |
| **方案B (DWT+CNN)** | 0 | ~2.5M | ~2.5M |
| 方案B Large | 0 | ~5M | ~5M |

**优点**：
- CNN decoder有生成能力，可学习生成高频细节
- 参数量仅为VAE的5%
- 保留DWT编码的高效性

**缺点**：
- 需要训练decoder
- 参数量比方案A多

---

## 四、实现细节

### 4.1 关键类说明

| 类名 | 文件 | 说明 |
|------|------|------|
| `DWTModelAllBandsLearnableTorch` | `ldm/models/dwt_autoencoder.py` | 方案A：纯PyTorch DWT+IDWT |
| `DWTModelCNNDecoder` | `ldm/models/dwt_autoencoder.py` | 方案B：DWT编码+CNN解码 |
| `DWTModelCNNDecoderLarge` | `ldm/models/dwt_autoencoder.py` | 方案B加强版 |
| `SobelEdgeLoss` | `train.py` | Sobel边缘损失 |
| `FFTFrequencyLoss` | `train.py` | FFT频率损失（可选） |

### 4.2 损失函数配置

```yaml
perceptual_loss:
  enabled: True
  lpips_weight_start: 0.02   # 初期MSE主导
  lpips_weight_end: 0.1      # 后期LPIPS增大
  lpips_net: 'vgg'

  edge_loss_enabled: True
  edge_loss_weight: 0.1

  freq_loss_enabled: False   # 可选，默认关闭
```

### 4.3 时间步动态权重

```python
# 基于时间步调整感知损失权重
T = 15  # 总时间步
t < 5:      weight = 1.0   # pred质量好
5 <= t < 10: weight = 0.5   # pred质量中等
t >= 10:    weight = 0.1   # pred质量差
```

---

## 五、训练命令

```bash
cd /home/lch/sr_recons/DifIISR-dwt
conda activate sr

# 方案A: DWT+IDWT + 增强损失
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/train_m3fd_dwt_enhanced.yaml \
    --seed 42

# 方案B: DWT编码 + CNN解码
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/train_m3fd_dwt_cnn_decoder.yaml \
    --seed 42
```

---

## 六、评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| PSNR | 像素准确度 | ≥35 dB |
| SSIM | 结构相似度 | ≥0.97 |
| LPIPS | 感知质量 | ≤0.27 (与DifIISR对齐) |
| 视觉效果 | 主观清晰度 | 接近VAE版本 |

---

## 七、文件清单

### 核心文件

| 文件 | 说明 |
|------|------|
| `ldm/models/dwt_autoencoder.py` | DWT编解码器实现 |
| `train.py` | 训练脚本 |
| `configs/train_m3fd_dwt_enhanced.yaml` | 方案A配置 |
| `configs/train_m3fd_dwt_cnn_decoder.yaml` | 方案B配置 |

### 数据相关

| 文件 | 说明 |
|------|------|
| `datapipe/paired_dataset.py` | 配对数据集加载器 |
| `datapipe/degradation.py` | Real-ESRGAN退化实现 |
| `datapipe/degradation_dataset.py` | 在线退化数据集 |

### 评估相关

| 文件 | 说明 |
|------|------|
| `eval_fmb.py` | DWT版本评估 |
| `eval_fmb_original.py` | VAE版本评估 |
| `inference_dwt.py` | DWT推理脚本 |

---

## 八、结论与建议

### 8.1 方案选择建议

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 追求极致轻量 | 方案A | 参数量243，计算量最低 |
| 追求视觉质量 | 方案B | CNN decoder有生成能力 |
| 平衡性能与质量 | 方案B | 参数量仅为VAE的5% |

### 8.2 后续优化方向

1. **方案A**：继续训练到50K+迭代，观察LPIPS是否持续下降
2. **方案B**：调整decoder深度和通道数，寻找最佳平衡点
3. **通用**：尝试GAN损失替代/补充LPIPS

---

## 九、方案B实验失败分析 (2026-01-28)

### 9.1 实验现象

使用自定义轻量CNN Decoder训练4000次迭代后：

| 阶段 | MSE | LPIPS | PSNR | SSIM |
|------|-----|-------|------|------|
| 训练 Iter 100 | 0.0129 | 0.4369 | - | - |
| 训练 Iter 4000 | 0.0042 | 0.2248 | - | - |
| **验证 Iter 1000** | - | 0.7648 | **6.01 dB** | **-0.0160** |
| **验证 Iter 4000** | - | 0.6221 | **11.75 dB** | **0.1609** |

**问题**：训练指标正常下降，但验证指标极差（正常PSNR应30+dB，SSIM应0.9+）。

### 9.2 失败原因分析

**根本原因：CNN Decoder 没有预训练，不知道如何将 DWT 子带转换为图像。**

```
DWT+IDWT 方案（能工作）：
  z = DWT.encode(image)
  image_rec = IDWT(z)  ← 数学精确逆变换，不需要学习

DWT+CNN 方案（失败）：
  z = DWT.encode(image)
  image_rec = CNN.decode(z)  ← CNN从随机初始化，不知道z→image的映射！
```

**训练vs验证的差异**：
- 训练时：CNN只看到扩散模型的单步预测 `pred_zstart`
- 验证时：CNN看到完整DDIM采样的结果，分布不同
- CNN没有学会通用的 `DWT子带→图像` 映射

### 9.3 核心教训

| 组件 | 能否用DWT替代 | 原因 |
|------|---------------|------|
| **Encoder** | ✅ 可以 | 任务是"压缩/下采样"，DWT天然能做到 |
| **Decoder** | ❌ 不行 | 任务是"生成/补全细节"，需要学习能力 |

```
Encoder 的任务：图像 → 潜空间
  - 本质是"压缩"和"提取特征"
  - DWT 天然能做到（数学变换，无损）
  - 不需要学习

Decoder 的任务：潜空间 → 图像
  - 本质是"生成"和"补全细节"
  - 扩散模型输出的是"增强后的潜空间"
  - 需要把它"画"成清晰的图像
  - 这需要**生成能力**和**预训练**
  - IDWT 只能做精确逆变换，不能"创造"
  - 随机初始化的CNN也不行，必须预训练
```

---

## 十、最终结论：DWT只能替代Encoder

### 10.1 最终可行方案

**DWT Encoder + VAE Decoder（使用原版预训练Decoder）**

| 组件 | 原版VAE | 最终方案 | 说明 |
|------|---------|----------|------|
| Encoder | CNN (24.5M) | **DWT (0参数)** | DWT替代 |
| Decoder | CNN (24.5M) | **CNN (24.5M)** | 保留原版 |
| **总参数** | **49M** | **24.5M** | **-50%** |

### 10.2 收益分析

| 指标 | 原版VAE | DWT Encoder + VAE Decoder |
|------|---------|---------------------------|
| Encoder参数 | 24.5M | **0** |
| Decoder参数 | 24.5M | 24.5M |
| 总参数 | 49M | **24.5M (-50%)** |
| 编码计算量 | ~57M MACs | **~1M MACs (-98%)** |
| 解码计算量 | ~57M MACs | ~57M MACs |
| 生成能力 | ✓ | ✓ |
| 视觉质量 | 基准 | 相当 |

### 10.3 实现思路

```python
class DWTEncoderVAEDecoder(nn.Module):
    def __init__(self, vae_decoder_path):
        self.dwt = DWTForward()  # DWT编码（0参数）
        self.vae_decoder = load_pretrained_vae_decoder(vae_decoder_path)  # 原版VAE Decoder

    def encode(self, x):
        # 用DWT替代VAE Encoder
        return self.dwt(x)

    def decode(self, z):
        # 用原版预训练的VAE Decoder
        return self.vae_decoder(z)
```

### 10.4 为什么之前的方案都失败了

| 方案 | 失败原因 |
|------|----------|
| DWT + IDWT | IDWT是精确逆变换，无生成能力 |
| DWT + 轻量CNN | CNN没有预训练，不知道如何解码 |
| DWT + 增强损失 | 损失函数无法弥补IDWT的生成能力缺失 |

**最终答案**：Decoder必须有生成能力且经过预训练，原版VAE Decoder是最佳选择。

---

## 十一、后续工作

1. **实现 DWT Encoder + VAE Decoder 混合架构**
2. **验证潜空间兼容性**：DWT输出的子带格式是否与VAE Decoder兼容
3. **如果不兼容**：需要一个轻量适配层，或者预训练CNN Decoder学习 `DWT子带→图像`

---

## 十二、VAE Decoder 兼容性分析

### 12.1 VAE 配置分析

```yaml
autoencoder:
  target: ldm.models.autoencoder.VQModelTorch
  params:
    embed_dim: 3
    ddconfig:
      z_channels: 3        # 潜空间通道数
      resolution: 256      # 输出分辨率
      in_channels: 3       # 输入图像通道
      out_ch: 3            # 输出图像通道
      ch: 128              # 基础通道数
      ch_mult: [1, 2, 4]   # 3级，下采样4倍
```

### 12.2 输入输出形状对比

| 组件 | 输入 | 输出 |
|------|------|------|
| VAE Encoder | 256×256×3 | 64×64×**3** |
| VAE Decoder | 64×64×**3** | 256×256×3 |
| DWT AllBands | 256×256×3 | 64×64×**21** |

### 12.3 兼容性问题

**不兼容！** VAE Decoder 期望输入是 `64×64×3`，但 DWT AllBands 输出是 `64×64×21`。

---

## 十三、最终方案：DWT编码 + UNet融合 + VAE解码

### 13.1 方案设计

**核心思路**：让UNet在DWT空间进行扩散，最后用投影层将21通道转为3通道，再送入预训练的VAE Decoder。

```
原版流程:
  LR → VAE.encode → 64×64×3 → UNet(in=6,out=3) → 64×64×3 → VAE.decode → SR

新版流程:
  LR → DWT.encode → 64×64×21 → UNet(in=42,out=21) → 64×64×21 → Proj(21→3) → VAE.decode → SR
                      ↑                    ↑              ↑           ↑
                 DWT全子带           UNet在DWT空间扩散   投影层    预训练VAE Decoder
```

### 13.2 架构对比

| 组件 | 原版 | 新版 | 说明 |
|------|------|------|------|
| Encoder | VAE (24.5M) | DWT (0) | DWT替代 |
| UNet输入 | 3+3=6 | 21+21=42 | 条件通道增加 |
| UNet输出 | 3 | **21** | 在DWT空间扩散 |
| 投影层 | 无 | **21→3** | 新增，将DWT子带投影到VAE潜空间 |
| Decoder | VAE (24.5M) | VAE (24.5M) | 保留原版预训练权重 |

### 13.3 UNet配置

```yaml
model:
  params:
    in_channels: 42     # 21(噪声DWT子带) + 21(LQ DWT子带)
    out_channels: 21    # 输出21通道，在DWT空间扩散
```

### 13.4 数据流详解

```
训练时:
  HR(256×256×3) → DWT.encode → z_hr(64×64×21)
  LR(64×64×3) → Bicubic↑ → (256×256×3) → DWT.encode → z_lr(64×64×21)

  z_hr + noise → z_t(64×64×21)
  UNet(z_t, z_lr) → pred(64×64×21)  ← UNet在DWT空间扩散

  pred → Proj(21→3) → VAE.decode → SR(256×256×3)
  Loss = MSE(SR, HR) + LPIPS(SR, HR)

推理时:
  LR → Bicubic↑ → DWT.encode → z_lr(64×64×21)
  noise(64×64×21) → DDIM采样 → pred(64×64×21)
  pred → Proj(21→3) → VAE.decode → SR(256×256×3)
```

### 13.5 关键点

1. **UNet输入21通道**：包含DWT的全部子带信息（LL+LH+HL+HH + level1高频）
2. **UNet输出21通道**：在DWT空间进行扩散，保持与输入相同的空间
3. **投影层(21→3)**：学习将DWT子带投影到VAE Decoder能理解的3通道表示
4. **VAE Decoder不变**：使用原版预训练权重，有生成能力
5. **训练目标**：让UNet学会在DWT空间增强图像，投影层学会"翻译"到VAE潜空间

### 13.6 收益分析

| 指标 | 原版 | 新版 | 收益 |
|------|------|------|------|
| Encoder参数 | 24.5M | 0 | **-100%** |
| Decoder参数 | 24.5M | 24.5M | 0 |
| 投影层参数 | 0 | ~66 | 可忽略 |
| UNet参数 | 118M | ~118M | ~0 (仅首层变化) |
| **总参数** | **167M** | **142.5M** | **-15%** |
| 编码计算量 | 57M MACs | 1M MACs | **-98%** |
| 生成能力 | ✓ | ✓ | 保持 |

### 13.7 实现要点

```python
class DWTEncoderOnly(nn.Module):
    def __init__(self, wavelet='haar', level=2):
        self.dwt = DWTForward(wavelet)
        self.down = nn.Conv2d(9, 9, ...)  # level1子带下采样
        self.proj = nn.Conv2d(21, 3, kernel_size=1)  # 投影层

    def encode(self, x):
        # DWT编码，输出21通道
        ...
        return z  # [B, 21, 64, 64]

    def project(self, z):
        # 投影到3通道
        return self.proj(z)  # [B, 3, 64, 64]
```

### 13.8 配置文件

```yaml
# configs/train_m3fd_dwt_vae_decoder.yaml
model:
  params:
    in_channels: 42     # 21 + 21
    out_channels: 21    # 在DWT空间扩散

dwt_encoder:
  target: ldm.models.dwt_autoencoder.DWTEncoderOnly
  params:
    wavelet: haar
    level: 2

autoencoder:
  target: ldm.models.autoencoder.VQModelTorch
  ckpt_path: weights/autoencoder_vq_f4.pth  # 使用预训练权重
```

---

## 十四、实现细节与问题解决 (2026-01-28)

### 14.1 架构实现

最终实现的数据流：

```
训练时:
  HR(256×256×3) → DWT.encode → z_hr(64×64×21)
  LR(64×64×3) → Bicubic↑ → DWT.encode → z_lr(64×64×21)

  z_hr + noise → z_t(64×64×21)
  UNet(z_t, z_lr) → pred(64×64×21)  ← UNet在DWT空间扩散

  pred → Proj(21→3) → VAE.decode(force_not_quantize=True) → SR(256×256×3)
  Loss = MSE(SR, HR) + LPIPS(SR, HR) + Edge(SR, HR)

推理时:
  LR → Bicubic↑ → DWT.encode → z_lr(64×64×21)
  noise(64×64×21) → DDIM采样 → pred(64×64×21)
  pred → Proj(21→3) → VAE.decode → SR(256×256×3)
```

### 14.2 关键代码修改

| 文件 | 修改内容 |
|------|----------|
| `ldm/models/dwt_autoencoder.py` | `DWTEncoderOnly`类新增`proj`投影层(21→3)和`project()`方法 |
| `models/gaussian_diffusion_test.py` | `training_losses()`支持`dwt_encoder`参数；新增`encode_first_stage_dwt()`方法；`decode_first_stage()`支持投影层和`force_not_quantize` |
| `train.py` | `validate()`/`save_checkpoint()`支持`dwt_encoder`；`plot_training_curves()`改进为每种loss单独绘图 |
| `configs/train_m3fd_dwt_vae_decoder.yaml` | `out_channels: 21`（UNet在DWT空间扩散） |

### 14.3 OOM问题解决

**问题**：训练时CUDA内存不足（OOM），错误发生在VAE decoder的`quantize`步骤。

**原因**：`VQModelTorch.decode()`默认调用向量量化（VectorQuantizer），消耗大量显存。

**解决方案**：
1. 调用`decode(z, force_not_quantize=True)`跳过量化步骤
2. 将batch_size从8减小到4

```python
# models/gaussian_diffusion_test.py
def decode_first_stage(self, z_sample, first_stage_model, ...):
    # 检查是否支持force_not_quantize参数
    if 'force_not_quantize' in sig.parameters:
        out = first_stage_model.decode(z_sample, force_not_quantize=True)
    else:
        out = first_stage_model.decode(z_sample)
```

### 14.4 训练配置

```yaml
train:
  batch_size: 4        # 从8改为4，解决OOM
  val_batch_size: 4
  lr: 5e-5
  iterations: 50000

perceptual_loss:
  enabled: True
  lpips_weight_start: 0.05
  lpips_weight_end: 0.2
  edge_loss_enabled: True
  edge_loss_weight: 0.1
```

### 14.5 训练监控改进

训练脚本现在生成以下图表：

| 文件 | 内容 |
|------|------|
| `loss_total.png` | 总损失曲线 |
| `loss_mse.png` | MSE损失曲线 |
| `loss_lpips.png` | LPIPS感知损失曲线 |
| `loss_edge.png` | Sobel边缘损失曲线 |
| `loss_freq.png` | FFT频率损失曲线（如启用） |
| `learning_rate.png` | 学习率调度曲线 |
| `training_curves_all.png` | 综合图（2×3布局） |
| `loss_history.csv` | 所有loss数据（含edge、freq列） |

### 14.6 参数量统计

```
DWT Encoder:
  - down conv (level1下采样): 90 参数
  - proj conv (21→3投影): 66 参数
  - 总计: 156 参数

UNet: 118.67M 参数

VAE Decoder: 24.5M 参数（冻结，仅用于解码）

总可训练参数: ~118.67M
```

### 14.7 训练空间一致性问题修复 (2026-01-28)

**问题现象**：1000次迭代后PSNR=23.67dB，2000次迭代后PSNR下降到22.42dB。

**根本原因**：训练和推理的空间不一致。

```
错误的实现（图像空间MSE）：
  训练时：UNet输出 → 投影层 → VAE decoder → pred_img
          Loss = MSE(pred_img, HR_image)  ← 在图像空间计算

  推理时：DDIM在21通道DWT空间迭代
          sample = pred_xstart * k + x * m + y * j
          ↑ 这个公式假设pred_xstart是对z_start的预测，但实际不是！

结果：DDIM迭代公式失效，采样质量下降。
```

**解决方案**：在DWT空间计算MSE loss。

```
正确的实现（DWT空间MSE）：
  训练时：UNet输出(21通道) vs z_start(21通道)
          Loss = MSE(model_output, z_start)  ← 在DWT空间计算

  推理时：DDIM在21通道DWT空间迭代（与训练一致）
          最后一步：pred → 投影层 → VAE decoder → SR图像
```

**损失函数组成**：
```
Total Loss = MSE(DWT空间) + λ_lpips * LPIPS(图像空间) + λ_edge * Edge(图像空间)

- MSE: 在DWT空间计算，保证DDIM采样正确
- LPIPS: 在图像空间计算，提升感知质量
- Edge: 在图像空间计算，增强边缘锐度
```

**代码修改**：
```python
# models/gaussian_diffusion_test.py - training_losses()
if dwt_encoder is not None:
    if self.model_mean_type == ModelMeanType.START_X:
        # 在DWT空间计算MSE（主要loss，让DDIM采样正确）
        terms["mse"] = mean_flat((z_start - model_output) ** 2)
```

---

## 十五、投影层瓶颈问题分析 (2026-01-28)

### 15.1 实验数据

训练日志：`/home/lch/sr_recons/experiments/DifIISR_DWT_VAE_Decoder_20260128_212243/train.log`

| 迭代 | PSNR | SSIM | 验证LPIPS | 训练lpips |
|------|------|------|-----------|-----------|
| 500 | 8.50 dB | -0.18 | 0.4720 | 0.45 |
| 1500 | 11.83 dB | 0.34 | 0.4380 | 0.32 |
| 3000 | 28.80 dB | 0.93 | 0.3820 | 0.13 |
| 4000 | 35.39 dB | 0.97 | 0.3478 | 0.24 |
| 5000 | 35.61 dB | 0.97 | 0.3382 | 0.21 |
| 6000 | 35.87 dB | 0.97 | 0.3307 | 0.18 |
| 7000 | 35.85 dB | 0.97 | **0.3310** | 0.22 |

### 15.2 问题现象

1. **PSNR/SSIM 快速收敛**：3000次迭代后达到35+ dB, 0.97
2. **训练lpips波动大**：在0.04~0.30之间波动
3. **验证LPIPS卡住**：从0.47降到0.33后停滞，无法继续下降
4. **目标差距**：原版DifIISR的LPIPS约0.27，当前方案卡在0.33

### 15.3 根本原因：投影层分布不匹配

```
问题所在：
  DWT子带(21ch) ──→ Proj(1×1 Conv, 66参数) ──→ 3ch ──→ VAE Decoder
                           ↑
                      这里是瓶颈！
```

**VAE Decoder的期望输入**：
- VAE Decoder是在VAE Encoder的输出上预训练的
- VAE Encoder输出的3通道潜空间有特定的分布特征（均值、方差、通道间相关性）
- VAE Decoder只"认识"这种分布

**实际输入**：
- 投影层只有66个参数（21×3 + 3 bias）
- 它是一个简单的线性变换，无法学会复杂的非线性映射
- 输出的分布与VAE Decoder期望的分布不匹配

**结果**：
- VAE Decoder收到"陌生"的输入分布
- 生成的图像整体结构正确（PSNR/SSIM高）
- 但缺少高频细节和纹理（LPIPS高）

### 15.4 训练vs验证的LPIPS差异分析

```
训练时：
  单步预测 pred_zstart → 投影 → VAE解码 → LPIPS
  (pred_zstart是UNet对z_start的直接预测，质量较好)

验证时：
  完整DDIM采样(15步迭代) → 投影 → VAE解码 → LPIPS
  (累积误差，且投影层分布不匹配问题被放大)
```

训练lpips可以降到0.04~0.13，但验证LPIPS卡在0.33，说明：
- 投影层在训练时的单步预测分布上能工作
- 但在DDIM多步采样的输出分布上泛化能力差
- 66参数的线性投影无法适应分布漂移

### 15.5 解决方案

#### 方案1：增强投影层（推荐）

```python
# 当前：简单1×1卷积（66参数）
self.proj = nn.Conv2d(21, 3, kernel_size=1)

# 改进：多层投影网络（~25K参数）
self.proj = nn.Sequential(
    nn.Conv2d(21, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 3, kernel_size=1),
)
```

**优点**：
- 非线性变换能力更强
- 可以学习更复杂的分布映射
- 参数量仍然很小（~25K vs VAE Encoder 24.5M）

#### 方案2：UNet直接输出3通道

```
当前：LR → DWT.encode → 64×64×21 → UNet(in=42, out=21) → Proj(21→3) → VAE.decode
改进：LR → DWT.encode → 64×64×21 → UNet(in=42, out=3)  → VAE.decode
                                              ↑
                                    让UNet学习投影
```

**优点**：
- UNet有足够的参数（118M）来学习DWT→VAE潜空间的映射
- 不需要额外的投影层
- 端到端训练更简单

**缺点**：
- UNet输出空间与输入空间不一致（输入21ch，输出3ch）
- DDIM采样公式可能需要调整

#### 方案3：预训练投影层

1. 收集大量图像对：`(DWT子带, VAE潜空间)`
2. 单独训练投影层：`min ||Proj(DWT(img)) - VAE.encode(img)||`
3. 然后再训练整个系统

**优点**：
- 投影层先学会正确的分布映射
- 后续训练更稳定

**缺点**：
- 需要额外的预训练步骤
- 增加实现复杂度

### 15.6 结论

当前设计的瓶颈在于**66参数的1×1卷积投影层无法将DWT子带正确映射到VAE Decoder期望的潜空间分布**。

**推荐方案**：方案2（UNet输出3通道）或方案1（增强投影层）

**下一步**：
1. 修改UNet配置：`out_channels: 3`
2. 移除投影层，让UNet直接输出VAE潜空间
3. 重新训练验证

---

*文档版本: v8.0*
*更新日期: 2026-01-28*
*更新内容: 添加投影层瓶颈问题分析，LPIPS卡在0.33无法下降的原因*
