"""
梯度链路验证脚本
验证 LPIPS 损失的梯度是否能正确回传到 UNet 和 autoencoder
"""

import torch
import sys
sys.path.insert(0, '.')

from ldm.models.dwt_autoencoder import DWTModelAllBandsLearnableTorch

print("=" * 60)
print("梯度链路验证")
print("=" * 60)

# 1. 测试 autoencoder.decode 的梯度
print("\n[1] 测试 DWTModelAllBandsLearnableTorch.decode 梯度")
print("-" * 40)

model = DWTModelAllBandsLearnableTorch(wavelet='haar', level=2)
z = torch.randn(1, 21, 64, 64, requires_grad=True)
x = model.decode(z)
loss = x.mean()
loss.backward()

print(f"输入 z.grad is not None: {z.grad is not None}")
print(f"model.up.weight.grad is not None: {model.up.weight.grad is not None}")
print(f"model.down.weight.grad is not None: {model.down.weight.grad is not None}")

if model.up.weight.grad is not None:
    print(f"model.up.weight.grad.abs().mean(): {model.up.weight.grad.abs().mean():.6f}")
else:
    print("ERROR: model.up 没有收到梯度!")

# 2. 测试完整的训练流程梯度
print("\n[2] 测试完整训练流程梯度 (模拟 LPIPS)")
print("-" * 40)

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips 未安装，跳过完整测试")

if HAS_LPIPS:
    # 重新创建模型
    model = DWTModelAllBandsLearnableTorch(wavelet='haar', level=2)
    lpips_fn = lpips.LPIPS(net='vgg')
    lpips_fn.eval()
    for param in lpips_fn.parameters():
        param.requires_grad = False

    # 模拟 pred_zstart (来自UNet输出)
    pred_zstart = torch.randn(1, 21, 64, 64, requires_grad=True)

    # decode
    pred_img = model.decode(pred_zstart)
    pred_img = pred_img.clamp(-1, 1)

    # GT
    gt = torch.randn(1, 3, 256, 256)

    # LPIPS loss
    lpips_loss = lpips_fn(pred_img, gt).mean()

    # backward
    lpips_loss.backward()

    print(f"pred_zstart.grad is not None: {pred_zstart.grad is not None}")
    print(f"model.up.weight.grad is not None: {model.up.weight.grad is not None}")

    if pred_zstart.grad is not None:
        print(f"pred_zstart.grad.abs().mean(): {pred_zstart.grad.abs().mean():.6f}")
    else:
        print("ERROR: pred_zstart 没有收到梯度! LPIPS 无法训练 UNet!")

    if model.up.weight.grad is not None:
        print(f"model.up.weight.grad.abs().mean(): {model.up.weight.grad.abs().mean():.6f}")
    else:
        print("ERROR: model.up 没有收到梯度!")

print("\n" + "=" * 60)
if model.up.weight.grad is not None and (not HAS_LPIPS or pred_zstart.grad is not None):
    print("✓ 梯度链路正常，LPIPS 可以有效训练模型")
else:
    print("✗ 梯度链路有问题，请检查代码")
print("=" * 60)
