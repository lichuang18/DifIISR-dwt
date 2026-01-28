"""
测试DWT编解码器重建误差
用于诊断PSNR只有9dB的问题

修复后重新测试
"""
import torch
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("DWT编解码器重建误差测试 (修复后)")
print("=" * 60)

# 创建测试输入
x = torch.randn(1, 3, 256, 256)
print(f"\n输入形状: {x.shape}")
print(f"输入范围: [{x.min():.3f}, {x.max():.3f}]")

# 测试1: PyTorch DWT/IDWT模块 (修复后)
print("\n" + "=" * 60)
print("Test 1: PyTorch DWT/IDWT模块 (修复后)")
print("=" * 60)
from ldm.models.dwt_autoencoder import DWTForward, DWTInverse
dwt = DWTForward('haar')
idwt = DWTInverse('haar')

x_test = torch.randn(1, 3, 128, 128)
LL, (LH, HL, HH) = dwt(x_test)
x_rec_dwt = idwt(LL, (LH, HL, HH))
print(f"输入形状: {x_test.shape}")
print(f"LL形状: {LL.shape}")
print(f"重建形状: {x_rec_dwt.shape}")
print(f"单级DWT/IDWT误差 (max): {(x_test - x_rec_dwt).abs().max():.6f}")
print(f"单级DWT/IDWT误差 (mean): {(x_test - x_rec_dwt).abs().mean():.6f}")

# 测试2: DWTModelSimple (基准)
print("\n" + "=" * 60)
print("Test 2: DWTModelSimple (只扩散LL子带，PSNR=35dB的版本)")
print("=" * 60)
from ldm.models.dwt_autoencoder import DWTModelSimple
model1 = DWTModelSimple(wavelet='haar', level=2)
z1 = model1.encode(x)
x_rec1 = model1.decode(z1)
print(f"潜空间形状: {z1.shape}")
print(f"重建形状: {x_rec1.shape}")
print(f"重建误差 (max): {(x - x_rec1).abs().max():.6f}")
print(f"重建误差 (mean): {(x - x_rec1).abs().mean():.6f}")

# 测试3: DWTModelAllBandsLearnableTorch (修复后)
print("\n" + "=" * 60)
print("Test 3: DWTModelAllBandsLearnableTorch (修复后)")
print("=" * 60)
from ldm.models.dwt_autoencoder import DWTModelAllBandsLearnableTorch
model2 = DWTModelAllBandsLearnableTorch(wavelet='haar', level=2)
z2 = model2.encode(x)
x_rec2 = model2.decode(z2)
print(f"潜空间形状: {z2.shape}")
print(f"重建形状: {x_rec2.shape}")
print(f"重建误差 (max): {(x - x_rec2).abs().max():.6f}")
print(f"重建误差 (mean): {(x - x_rec2).abs().mean():.6f}")
print(f"重建范围: [{x_rec2.min():.3f}, {x_rec2.max():.3f}]")

# 测试4: 可学习up/down采样的round-trip误差
print("\n" + "=" * 60)
print("Test 4: 可学习Up/Down采样的round-trip误差")
print("=" * 60)
level1_bands = torch.randn(1, 9, 128, 128)
level1_down = model2.down(level1_bands)
level1_up = model2.up(level1_down)
print(f"原始 level1: {level1_bands.shape}")
print(f"下采样后: {level1_down.shape}")
print(f"上采样后: {level1_up.shape}")
print(f"Round-trip误差 (max): {(level1_bands - level1_up).abs().max():.6f}")
print(f"Round-trip误差 (mean): {(level1_bands - level1_up).abs().mean():.6f}")

# 测试5: DWTModelFullBand
print("\n" + "=" * 60)
print("Test 5: DWTModelFullBand (全子带但无可学习采样)")
print("=" * 60)
from ldm.models.dwt_autoencoder import DWTModelFullBand
model3 = DWTModelFullBand(wavelet='haar', level=2)
z3 = model3.encode(x)
x_rec3 = model3.decode(z3)
print(f"潜空间形状: {z3.shape}")
print(f"重建形状: {x_rec3.shape}")
print(f"重建误差 (max): {(x - x_rec3).abs().max():.6f}")
print(f"重建误差 (mean): {(x - x_rec3).abs().mean():.6f}")

# 测试6: 2级DWT/IDWT
print("\n" + "=" * 60)
print("Test 6: 2级DWT/IDWT (PyTorch)")
print("=" * 60)
x_256 = torch.randn(1, 3, 256, 256)
# 第一级
LL1, highs1 = dwt(x_256)
# 第二级
LL2, highs2 = dwt(LL1)
print(f"输入: {x_256.shape}")
print(f"Level1 LL: {LL1.shape}")
print(f"Level2 LL: {LL2.shape}")
# 重建
LL1_rec = idwt(LL2, highs2)
x_rec_2level = idwt(LL1_rec, highs1)
print(f"重建: {x_rec_2level.shape}")
print(f"2级DWT/IDWT误差 (max): {(x_256 - x_rec_2level).abs().max():.6f}")
print(f"2级DWT/IDWT误差 (mean): {(x_256 - x_rec_2level).abs().mean():.6f}")

# 总结
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"PyTorch单级DWT/IDWT误差:             {(x_test - x_rec_dwt).abs().max():.6f}")
print(f"PyTorch 2级DWT/IDWT误差:             {(x_256 - x_rec_2level).abs().max():.6f}")
print(f"DWTModelSimple 重建误差:              {(x - x_rec1).abs().max():.6f}")
print(f"DWTModelFullBand 重建误差:            {(x - x_rec3).abs().max():.6f}")
print(f"DWTModelAllBandsLearnableTorch 重建误差: {(x - x_rec2).abs().max():.6f}")
print(f"可学习采样 round-trip 误差:           {(level1_bands - level1_up).abs().max():.6f}")

# 判断是否修复成功
if (x_test - x_rec_dwt).abs().max() < 1e-5:
    print("\n✓ PyTorch DWT/IDWT模块已修复!")
else:
    print("\n✗ PyTorch DWT/IDWT模块仍有问题")

if (x - x_rec2).abs().max() < 1.0:
    print("✓ DWTModelAllBandsLearnableTorch 重建误差在可接受范围")
else:
    print("✗ DWTModelAllBandsLearnableTorch 重建误差仍然过大")
    print("  注意: 可学习采样的round-trip误差是预期的，会在训练中学习")

