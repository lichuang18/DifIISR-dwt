"""
测试 DWTModelAllBandsLearnableTorch 的修复

修复内容：
1. _init_weights: 将 weight[i, i, ...] 改为 weight[i, 0, ...] (groups=9时第二维是1)
2. encode: 改用纯PyTorch的DWTForward，与decode的DWTInverse保持一致
"""

import torch
from ldm.models.dwt_autoencoder import DWTModelAllBandsLearnableTorch

def test_weight_shapes():
    """测试权重形状是否正确"""
    print("=" * 60)
    print("1. 测试权重形状")
    print("=" * 60)

    model = DWTModelAllBandsLearnableTorch(wavelet='haar', level=2)

    print(f"self.down.weight.shape: {model.down.weight.shape}")
    print(f"  期望: [9, 1, 3, 3] (groups=9 时第二维是1)")

    print(f"self.up.weight.shape: {model.up.weight.shape}")
    print(f"  期望: [9, 1, 4, 4] (groups=9 时第二维是1)")

    assert model.down.weight.shape == torch.Size([9, 1, 3, 3]), "down weight shape error"
    assert model.up.weight.shape == torch.Size([9, 1, 4, 4]), "up weight shape error"

    print("✓ 权重形状正确")


def test_encode_decode():
    """测试编解码功能"""
    print("\n" + "=" * 60)
    print("2. 测试编解码功能")
    print("=" * 60)

    model = DWTModelAllBandsLearnableTorch(wavelet='haar', level=2)

    x = torch.randn(2, 3, 256, 256)
    z = model.encode(x)
    x_rec = model.decode(z)

    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstructed shape: {x_rec.shape}")

    recon_error = (x - x_rec).abs().max().item()
    print(f"Reconstruction error: {recon_error:.6f}")

    # 修复后重建误差应该很小（接近0）
    if recon_error < 0.01:
        print("✓ 重建误差极小（接近完美重建）")
    elif recon_error < 1.0:
        print(f"⚠ 重建误差较小但不完美: {recon_error:.6f}")
    else:
        print(f"✗ 重建误差过大: {recon_error:.6f}")
        print("  可能原因: encode和decode的DWT/IDWT实现不一致")


def test_gradient_flow():
    """测试梯度是否能正确回传"""
    print("\n" + "=" * 60)
    print("3. 测试梯度回传")
    print("=" * 60)

    model = DWTModelAllBandsLearnableTorch(wavelet='haar', level=2)
    model.zero_grad()

    x = torch.randn(2, 3, 256, 256)
    z = model.encode(x)
    x_rec = model.decode(z)

    loss = x_rec.mean()
    loss.backward()

    up_grad = model.up.weight.grad
    down_grad = model.down.weight.grad

    print(f"self.up.weight.grad is not None: {up_grad is not None}")
    print(f"self.down.weight.grad is not None: {down_grad is not None}")

    if up_grad is not None:
        print(f"self.up.weight.grad.abs().mean(): {up_grad.abs().mean():.6f}")
    if down_grad is not None:
        print(f"self.down.weight.grad.abs().mean(): {down_grad.abs().mean():.6f}")

    if up_grad is not None and down_grad is not None:
        print("✓ 梯度回传正常")
    else:
        print("✗ 梯度回传失败")


def test_init_values():
    """测试初始化值是否正确"""
    print("\n" + "=" * 60)
    print("4. 测试初始化值")
    print("=" * 60)

    model = DWTModelAllBandsLearnableTorch(wavelet='haar', level=2)

    # 检查 down 卷积的中心点是否为1
    print("down 卷积权重 (应该中心点为1):")
    for i in range(3):  # 只打印前3个通道
        center_val = model.down.weight[i, 0, 1, 1].item()
        print(f"  通道 {i}: 中心值 = {center_val:.4f}")

    # 检查 up 反卷积是否接近双线性插值
    print("\nup 反卷积权重 (应该接近双线性插值):")
    print(f"  通道 0 的权重:\n{model.up.weight[0, 0]}")

    # 验证中心点值
    down_center_ok = all(
        abs(model.down.weight[i, 0, 1, 1].item() - 1.0) < 0.01
        for i in range(9)
    )

    if down_center_ok:
        print("\n✓ 初始化值正确")
    else:
        print("\n✗ 初始化值有问题")


def test_dwt_idwt_consistency():
    """测试DWT和IDWT的一致性（不经过可学习层）"""
    print("\n" + "=" * 60)
    print("5. 测试DWT/IDWT一致性（不经过可学习层）")
    print("=" * 60)

    from ldm.models.dwt_autoencoder import DWTForward, DWTInverse

    dwt = DWTForward('haar')
    idwt = DWTInverse('haar')

    x = torch.randn(2, 3, 256, 256)

    # 一级DWT + IDWT
    LL1, (LH1, HL1, HH1) = dwt(x)
    x_rec1 = idwt(LL1, (LH1, HL1, HH1))
    error1 = (x - x_rec1).abs().max().item()
    print(f"一级DWT+IDWT重建误差: {error1:.10f}")

    # 二级DWT + IDWT
    LL2, (LH2, HL2, HH2) = dwt(LL1)
    LL1_rec = idwt(LL2, (LH2, HL2, HH2))
    x_rec2 = idwt(LL1_rec, (LH1, HL1, HH1))
    error2 = (x - x_rec2).abs().max().item()
    print(f"二级DWT+IDWT重建误差: {error2:.10f}")

    if error2 < 1e-5:
        print("✓ DWT/IDWT完美可逆")
    else:
        print(f"⚠ DWT/IDWT有误差: {error2:.10f}")


if __name__ == "__main__":
    print("测试 DWTModelAllBandsLearnableTorch 修复")
    print("=" * 60)

    try:
        test_weight_shapes()
        test_dwt_idwt_consistency()
        test_encode_decode()
        test_gradient_flow()
        test_init_values()

        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
