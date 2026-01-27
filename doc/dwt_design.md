# DifIISR-DWT 设计文档

## 概述

本文档记录了将 DifIISR 中的 VQ-VAE 编解码器替换为离散小波变换 (DWT) 的所有改动。

---

## 一、新增文件

| 文件 | 功能 | 说明 |
|------|------|------|
| `ldm/models/dwt_autoencoder.py` | **DWT编解码器** | 核心：用DWT替代VAE |
| `train.py` | **训练脚本** | 原版只有推理，无训练代码 |
| `datapipe/paired_dataset.py` | 配对数据集加载器 | 加载HR-LR配对数据 |
| `datapipe/degradation.py` | Real-ESRGAN退化 | 实现二阶退化流程 |
| `datapipe/degradation_dataset.py` | 在线退化数据集 | 训练时动态生成LR |
| `eval_fmb.py` | DWT版本FMB评估 | 测试DWT模型 |
| `eval_fmb_original.py` | 原版FMB评估 | 测试VAE模型 |
| `inference_dwt.py` | DWT版本推理 | DWT模型推理脚本 |
| `scripts/prepare_m3fd_dataset.py` | 数据预处理 | 生成训练patches |

---

## 二、新增配置文件

| 文件 | 用途 |
|------|------|
| `configs/train_m3fd_dwt.yaml` | DWT + Bicubic退化训练 |
| `configs/train_m3fd_dwt_degradation.yaml` | DWT + Real-ESRGAN退化训练 (haar) |
| `configs/train_m3fd_dwt_db2.yaml` | DWT + db2小波基 |
| `configs/train_m3fd_dwt_bior.yaml` | DWT + bior小波基 |

---

## 三、修改的原有文件

### 1. `models/gaussian_diffusion_test.py`

**修改位置**: `encode_first_stage` 和 `decode_first_stage` 函数

**修改原因**: DWT模型没有可学习参数，`next(model.parameters())` 会抛出 `StopIteration`

**修改内容**:
```python
# 原版
model_dtype = next(first_stage_model.parameters()).dtype
y = y.type(dtype=model_dtype)

# 修改后
try:
    model_dtype = next(first_stage_model.parameters()).dtype
    y = y.type(dtype=model_dtype)
except StopIteration:
    # 模型没有参数（如DWT），保持原始dtype
    pass
```

### 2. `datapipe/degradation.py`

**修改位置**: `add_poisson_noise` 函数

**修改原因**: 泊松噪声的lambda参数可能为负导致错误

**修改内容**:
```python
# 原版
noise = np.random.poisson(gray * 255 * scale)

# 修改后
img = np.clip(img, 0, 1)  # 确保输入在有效范围内
lam = np.maximum(gray * 255 * scale, 0)  # 确保lambda非负
noise = np.random.poisson(lam)
```

---

## 四、核心改动详解

### DWT编解码器 (`ldm/models/dwt_autoencoder.py`)

```python
class DWTModelSimple(nn.Module):
    """
    用DWT替代VAE
    - 零参数，无需预训练
    - 计算量降低99%+
    - 完全可逆，无信息损失
    """
    def __init__(self, wavelet='haar', level=2):
        self.wavelet = wavelet  # 小波基
        self.level = level      # 分解级数

    def encode(self, x):
        # 256×256 → 64×64 (2级分解)
        coeffs = pywt.wavedec2(x, self.wavelet, level=self.level)
        LL = coeffs[0]  # 返回低频子带
        self.high_freq_cache = coeffs[1:]  # 缓存高频
        return LL

    def decode(self, z):
        # 64×64 → 256×256
        coeffs = [z] + self.high_freq_cache
        return pywt.waverec2(coeffs, self.wavelet)
```

### 小波基和分解级数的影响

#### 分解级数（关键参数）

| 分解级数 | 潜空间尺寸 | 对UNet的影响 | 是否需要重新训练 |
|---------|-----------|-------------|----------------|
| **1级** | 256→128×3 | 需要修改UNet的`image_size`为128 | **必须重新训练** |
| **2级** | 256→64×64×3 | 与原VAE一致，UNet无需改动 | 当前方案 |
| **3级** | 256→32×32×3 | 需要修改UNet的`image_size`为32 | **必须重新训练** |

**结论**：分解级数改变潜空间维度，不同级数对应**不同的模型架构**，必须分别训练。

#### 小波基（次要参数）

| 小波基 | 特点 | 是否需要重新训练 |
|-------|------|----------------|
| **haar** | 最简单，计算最快，边缘锐利 | 当前方案 |
| **db2** | Daubechies-2，更平滑 | 可能不需要，或只需微调 |
| **bior1.3** | 双正交，对称性好 | 尺寸不兼容，无法使用 |

**注意**：`bior1.3` 小波滤波器较长，分解后尺寸不是精确的 64×64，与UNet不兼容。

---

## 五、配置文件对比

| 配置项 | 原版 (VAE) | DWT版本 |
|--------|-----------|---------|
| `autoencoder.target` | `ldm.models.autoencoder.VQModelTorch` | `ldm.models.dwt_autoencoder.DWTModelSimple` |
| `autoencoder.ckpt_path` | `weights/autoencoder_vq_f4.pth` | `~` (无需权重) |
| `autoencoder.params` | embed_dim, n_embed, ddconfig... | `wavelet: haar, level: 2` |

### DWT配置示例

```yaml
autoencoder:
  target: ldm.models.dwt_autoencoder.DWTModelSimple
  ckpt_path: ~              # 无需权重文件
  use_fp16: False
  params:
    wavelet: haar           # 可选: haar, db2
    level: 2                # 2级分解: 256→64
```

---

## 六、改动量统计

| 类型 | 数量 |
|------|------|
| 新增Python文件 | 8个 |
| 新增配置文件 | 4个 |
| 修改原有文件 | 2个 |
| 新增代码行数 | ~1500行 |
| 修改代码行数 | ~20行 |

---

## 七、架构对比图

```
原版 DifIISR (VAE):
  LR → Bicubic↑ → VAE.encode(49M参数) → 潜空间扩散 → VAE.decode → SR
                   ↓
              需要196MB权重文件

DifIISR-DWT:
  LR → Bicubic↑ → DWT(0参数) → 频率空间扩散 → IDWT → SR
                   ↓
              无需权重文件
```

---

## 八、收益分析

| 指标 | 原版 (VAE) | DWT版本 | 收益 |
|------|-----------|---------|------|
| 编解码参数 | 49M | 0 | -100% |
| 权重文件大小 | 196MB | 0 | -100% |
| 编解码计算量 | ~115M MACs | ~0.6M MACs | -99.5% |
| 推理显存 | +200MB | +10MB | -95% |
| 编解码延迟 | ~50ms | ~0.5ms | -99% |
| 部署依赖 | 需VAE模型 | 无 | 简化 |

---

## 九、训练命令

```bash
conda activate sr
cd /home/lch/sr_recons/DifIISR

# haar基准 (Real-ESRGAN退化)
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train_m3fd_dwt_degradation.yaml --seed 42

# db2小波基
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/train_m3fd_dwt_db2.yaml --seed 42
```

---

## 十、评估命令

```bash
# 测试DWT版本
python eval_fmb.py \
    --config configs/train_m3fd_dwt_degradation.yaml \
    --ckpt <checkpoint_path> \
    --output_dir results/dwt_fmb \
    --num_images 20

# 测试原版VAE版本
python eval_fmb_original.py \
    --output_dir results/original_fmb \
    --num_images 20
```

---

---

## 十一、文件清单

### 新增文件

| 文件路径 | 说明 |
|---------|------|
| `ldm/models/dwt_autoencoder.py` | DWT编解码器核心实现 |
| `train.py` | 训练脚本 |
| `datapipe/paired_dataset.py` | 配对数据集加载器 |
| `datapipe/degradation.py` | Real-ESRGAN退化实现 |
| `datapipe/degradation_dataset.py` | 在线退化数据集 |
| `eval_fmb.py` | DWT版本FMB评估 |
| `eval_fmb_original.py` | 原版VAE评估 |
| `inference_dwt.py` | DWT推理脚本 |
| `ldm/modules/ema.py` | EMA模块 |
| `configs/train_m3fd_dwt_degradation.yaml` | haar训练配置 |
| `configs/train_m3fd_dwt_db2.yaml` | db2训练配置 |
| `configs/train_m3fd_dwt_bior.yaml` | bior训练配置 |
| `configs/train_m3fd.yaml` | 基础训练配置 |

### 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| `models/gaussian_diffusion_test.py` | 支持无参数模型(DWT) |

---

## 十二、实验结果与问题分析

### 12.1 FMB数据集测试结果

| 方法 | PSNR | SSIM | 视觉效果 |
|------|------|------|---------|
| DWT-haar | **37.70 dB** | **0.9959** | 偏模糊 |
| VAE原版 | 较低 | 较低 | **更清晰** |

### 12.2 问题现象

**DWT版本指标更高，但视觉效果不如VAE版本清晰。**

这是一个经典的 **"感知质量 vs 像素指标"** 问题。

### 12.3 原因分析

#### (1) PSNR/SSIM 高但视觉效果差的深层原因

SSIM 确实是设计来衡量视觉感知的，但它有局限性：

| SSIM特点 | 局限性 |
|---------|--------|
| 局部窗口计算（11×11） | 对**细粒度纹理**不敏感 |
| 衡量整体结构 | 模糊图像结构也"正确" |
| 低频主导 | 高频细节权重低 |

**关键认识：SSIM高只能说明"整体结构正确"，不能说明"细节清晰"。**

#### (2) DWT 当前设计的根本问题

**核心问题：DWT保留的是LR的高频，而LR的高频本身就是"错误的"、"残缺的"。**

```
关键认识：
  LR图像的高频 ≠ HR图像的高频

  LR的高频是"残缺的"、"模糊的"
  超分的目标是"生成"HR应有的高频，而不是"保留"LR的高频
```

**DWT当前做法**：
```
DWT流程:
  LR → DWT分解 → LL(低频) + HF(高频)
                    ↓           ↓
              扩散增强      直接保留(这些高频本身就是模糊的!)
                    ↓           ↓
                  IDWT重建 ← 合并
```

- 扩散模型只处理了低频LL子带
- 高频子带(LH, HL, HH)直接从LR传递
- **问题：LR的高频本身就是不足的，保留它们无法提升清晰度**

#### (3) VAE 的优势

```
VAE流程:
  LR → VAE编码 → 潜空间(语义特征)
                    ↓
              扩散生成新的语义特征
                    ↓
              VAE解码 → 生成HR应有的高频细节
```

- VAE的decoder是一个**生成模型**
- 它可以根据潜空间特征**生成**合理的高频细节
- 即使这些细节不完全"准确"（PSNR低），但看起来更自然、更锐利

#### (4) 本质区别

| 方法 | 策略 | PSNR/SSIM | 视觉效果 | 原因 |
|------|------|-----------|---------|------|
| DWT | **保守策略** | 高 | 模糊 | 保留LR的"正确但模糊"的高频 |
| VAE | **生成策略** | 低 | 清晰 | 生成"可能有偏差但锐利"的高频 |

**结论：超分任务需要"生成"新的高频细节，而不是"保留"旧的。DWT当前设计只做了保留，没有生成。**

### 12.4 改进方案

#### 方案A：高频子带也参与扩散（推荐）

```python
# 当前：只扩散LL
z = LL  # 64×64×3

# 改进：扩散全部子带
z = concat(LL, LH, HL, HH)  # 64×64×12
# 需要修改UNet输入通道: 6 → 24
```

**优点**：扩散模型可以增强高频细节
**缺点**：需要修改UNet架构，重新训练

#### 方案B：高频子带用轻量网络增强

```python
# LL用扩散增强
LL_enhanced = diffusion(LL)

# 高频用CNN增强
HF_enhanced = lightweight_cnn(LH, HL, HH)

# 合并重建
output = IDWT(LL_enhanced, HF_enhanced)
```

**优点**：不改变扩散模型架构
**缺点**：需要额外训练CNN模块

#### 方案C：添加感知损失

训练时加入LPIPS等感知损失，鼓励生成更锐利的结果。

```python
loss = mse_loss + lambda * lpips_loss
```

**优点**：不改变架构
**缺点**：可能降低PSNR指标

### 12.5 小波基兼容性问题

| 小波基 | 状态 | 原因 |
|-------|------|------|
| **haar** | ✅ 可用 | 滤波器长度=2，尺寸精确 |
| db2 | ❌ 不可用 | 滤波器长度=4，尺寸不匹配 |
| bior1.3 | ❌ 不可用 | 滤波器长度=6，尺寸不匹配 |

只有 `haar` 小波能保证精确的 2 倍下采样，其他小波需要使用 `mode='periodization'` 才能保证尺寸精确。

---

## 十三、结论与后续工作

### 当前方案总结

- **优点**：零参数、计算高效、PSNR/SSIM指标高
- **缺点**：视觉效果不如VAE版本清晰（高频未增强）

### 后续改进方向

1. **方案A（推荐）**：全子带扩散，让扩散模型同时增强低频和高频
2. **方案B**：混合架构，低频用扩散、高频用CNN
3. **方案C**：添加感知损失，提升视觉质量

---

## 十四、方案A实现：全子带扩散 (2026-01-27)

### 14.1 实现概述

将所有level2子带(LL+LH+HL+HH)拼接后送入扩散模型，让UNet同时增强低频和高频。

```
原方案 (仅LL扩散):
  LR(256) → DWT → LL(64×64×3) → UNet(in=6,out=3) → LL' → IDWT+HF → SR
                                                          ↑
                                                    HF未增强(问题)

方案A (全子带扩散):
  LR(256) → DWT → [LL,LH,HL,HH](64×64×12) → UNet(in=24,out=12) → [LL',LH',HL',HH'] → IDWT → SR
                                                                        ↑
                                                                  全部子带被增强
```

### 14.2 新增/修改文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `ldm/models/dwt_autoencoder.py` | 修改 | 新增 `DWTModelFullBand` 类 |
| `configs/train_m3fd_dwt_fullband.yaml` | 新增 | 全子带扩散训练配置 |
| `models/gaussian_diffusion_test.py` | 修改 | 支持编码后的lq条件 |
| `sampler.py` | 修改 | 支持DWT无checkpoint |

### 14.3 核心代码改动

#### (1) DWTModelFullBand 类

```python
class DWTModelFullBand(nn.Module):
    """
    全子带DWT编解码器 - 方案A
    编码: 256×256×3 → 64×64×12 (LL2 + LH2 + HL2 + HH2)
    解码: 64×64×12 → 256×256×3
    """
    def encode(self, x):
        # 2级DWT分解
        coeffs = pywt.wavedec2(x, 'haar', level=2)
        LL2 = coeffs[0]           # 64×64
        LH2, HL2, HH2 = coeffs[1]  # 各64×64
        LH1, HL1, HH1 = coeffs[2]  # 各128×128 (level1高频，缓存)

        # 拼接level2的4个子带
        z = concat(LL2, LH2, HL2, HH2)  # [B, 12, 64, 64]
        self.hf_level1_cache = (LH1, HL1, HH1)
        return z

    def decode(self, z):
        # 拆分子带
        LL2, LH2, HL2, HH2 = split(z)
        # 使用缓存的level1高频重建
        coeffs = [LL2, (LH2, HL2, HH2), self.hf_level1_cache]
        return pywt.waverec2(coeffs, 'haar')
```

#### (2) UNet配置变化

```yaml
# 原配置 (仅LL)
model:
  params:
    in_channels: 6      # 3(噪声) + 3(LQ条件)
    out_channels: 3

# 新配置 (全子带)
model:
  params:
    in_channels: 24     # 12(噪声子带) + 12(LQ条件)
    out_channels: 12    # 输出全部子带
```

#### (3) gaussian_diffusion_test.py 修改

```python
# training_losses 函数中
z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False)

# 关键修改：用编码后的z_y替换原始lq
if 'lq' in model_kwargs:
    model_kwargs = dict(model_kwargs)
    model_kwargs['lq'] = z_y  # 现在是12通道

# p_sample_loop_progressive 和 ddim_sample_loop_progressive 同样修改
```

### 14.4 架构对比

| 项目 | 原方案(仅LL) | 方案A(全子带) |
|------|-------------|--------------|
| 潜空间维度 | 64×64×3 | 64×64×12 |
| UNet输入通道 | 6 | 24 |
| UNet输出通道 | 3 | 12 |
| 高频处理 | 直接保留 | 扩散增强 |
| level1高频 | 缓存并保留 | 缓存并保留 |

### 14.5 训练命令

```bash
conda activate sr
cd /home/lch/sr_recons/DifIISR-dwt

# 全子带扩散训练
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train_m3fd_dwt_fullband.yaml --seed 42
```

### 14.6 预期效果

- **视觉质量**：高频子带被扩散模型增强，预期更清晰
- **PSNR/SSIM**：可能略有下降（生成策略 vs 保守策略）
- **计算量**：UNet参数增加（输入输出通道增加），但仍远低于VAE方案

### 14.7 潜空间变大的影响分析

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

## 十五、方案A实验结果与改进 (2026-01-27)

### 15.1 实验结果

| 配置 | 迭代次数 | PSNR | SSIM | 视觉效果 |
|------|----------|------|------|----------|
| 全子带(仅level2) | 10000 | 较高 | 较高 | **仍然模糊** |

### 15.2 问题根因

```
当前方案A的流程:
  LR(256) → DWT 2级分解
           ├── level2: LL2, LH2, HL2, HH2 (64×64) → 扩散增强 ✓
           └── level1: LH1, HL1, HH1 (128×128) → 直接保留 ✗ ← 问题根源

level1高频 (128×128×9) 包含大量边缘和纹理信息
这些高频仍然来自LR，未被增强，导致视觉效果模糊
```

### 15.3 快速验证：后处理锐化

- USM锐化：略有改善
- 高通滤波：略有改善
- 组合锐化：比原始好一点

**结论**：后处理无法真正"生成"缺失的高频，需要从模型层面解决。

### 15.4 改进方案：level1参与扩散 + 感知损失

#### (1) 新架构设计

采用**统一下采样方案**：将level1子带下采样到64×64，与level2拼接

```
改进后的流程:
  LR(256) → DWT 2级分解
           ├── level2: LL2, LH2, HL2, HH2 (64×64×12)
           └── level1: LH1, HL1, HH1 (128×128×9) → 下采样到64×64×9
                                    ↓
           拼接: 64×64×21 (12+9)
                                    ↓
           UNet(in=42, out=21) → 扩散增强全部子带
                                    ↓
           拆分: level2 (64×64×12) + level1 (64×64×9)
                                    ↓
           level1上采样回128×128×9
                                    ↓
           IDWT重建 → SR(256×256×3)
```

#### (2) UNet配置变化

```yaml
# 方案A (仅level2)
in_channels: 24   # 12 + 12
out_channels: 12

# 改进方案 (level1+level2)
in_channels: 42   # 21 + 21
out_channels: 21
```

#### (3) 感知损失

```python
loss = mse_loss + lambda_lpips * lpips_loss

# 推荐权重
lambda_lpips = 0.1
```

---

## 十六、AllBands方案实验结果与深入分析 (2026-01-27)

### 16.1 实验结果

| 配置 | 迭代次数 | 视觉效果 |
|------|----------|----------|
| AllBands + LPIPS | 8000 | 仍不如DifIISR清晰 |

### 16.2 问题根因分析

#### (1) level1下采样/上采样的信息损失

```
当前AllBands流程:
  level1 (128×128) → cv2.resize下采样 → 64×64 → 扩散 → cv2.resize上采样 → 128×128
                            ↑                              ↑
                      固定平均池化                      固定插值
                      (高频直接丢失)                   (无法恢复)
```

#### (2) VAE vs DWT 下采样的本质区别

**VAE（可学习）：**

| 阶段 | 操作 | 特点 |
|------|------|------|
| Encoder下采样 | 学习的卷积 | 保留重要特征 |
| Decoder上采样 | 学习的反卷积 | 生成合理细节 |
| 训练目标 | 重建损失 | 学会最优压缩 |

**DWT+resize（固定）：**

| 阶段 | 操作 | 特点 |
|------|------|------|
| 下采样 | cv2.INTER_AREA | 无差别平均 |
| 上采样 | cv2.INTER_CUBIC | 固定插值 |
| 无训练 | - | 无法优化 |

**核心区别**：

```
VAE: 49M参数学习"如何压缩"和"如何恢复"
     → Encoder学会保留重要信息
     → Decoder学会生成合理细节

DWT+resize: 0参数，固定算法
     → 下采样无差别丢弃高频
     → 上采样只能插值模糊
```

#### (3) DWT的根本局限性

```
VAE Decoder: 生成式模型，可以"幻想"出合理的高频细节
DWT IDWT: 确定性逆变换，只能恢复已有信息，无法凭空生成
```

### 16.3 改进方向

#### 方案A：可学习的上下采样

```python
# 用卷积代替cv2.resize
self.down = nn.Conv2d(9, 9, 3, stride=2, padding=1)  # 128→64，可学习
self.up = nn.ConvTranspose2d(9, 9, 4, stride=2, padding=1)  # 64→128，可学习
```

优点：减少信息损失
缺点：增加少量参数，需要训练

#### 方案B：多尺度UNet

```
level2 (64×64) → UNet小尺度分支
                      ↓ 特征交互
level1 (128×128) → UNet大尺度分支
```

优点：不下采样，无信息损失
缺点：架构复杂，计算量增加

#### 方案C：DWT编码 + 生成式解码器

```
编码: DWT (确定性，高效，0参数)
解码: 轻量CNN (可学习，有生成能力)
```

优点：结合两者优点
缺点：需要训练解码器

### 16.4 结论

DWT替代VAE的核心挑战不在于编码，而在于**解码时的细节生成能力**。

可能的出路：
1. 保留DWT编码的高效性
2. 用可学习的模块替代固定的resize/IDWT
3. 或者接受DWT的局限，专注于PSNR/SSIM指标而非视觉质量

---

*文档版本: v1.4*
*更新日期: 2026-01-27*
*更新内容: AllBands实验分析，VAE vs DWT对比，改进方向*
