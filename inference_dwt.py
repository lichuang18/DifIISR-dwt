"""
DifIISR-DWT 推理和评估脚本
生成SR图像并计算PSNR/SSIM指标
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils import util_net
from utils import util_common


def load_yaml_config(config_path):
    """加载YAML配置文件"""
    import yaml
    from omegaconf import OmegaConf
    with open(config_path, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    return config


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100
    return -10 * np.log10(mse)


def calculate_ssim(img1, img2):
    """计算SSIM (单通道)"""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


def calculate_ssim_color(img1, img2):
    """计算彩色图像的SSIM"""
    ssim_vals = []
    for i in range(img1.shape[2]):
        ssim_vals.append(calculate_ssim(img1[:, :, i], img2[:, :, i]))
    return np.mean(ssim_vals)


def create_comparison_image(lr_img, sr_img, hr_img, psnr, ssim, filename):
    """
    创建LR/SR/HR对比图
    lr_img, sr_img, hr_img: numpy arrays, [H, W, 3], range [0, 1]
    """
    # 将LR图像放大到与HR相同尺寸（用于对比显示）
    h, w = hr_img.shape[:2]
    lr_resized = np.array(Image.fromarray((lr_img * 255).astype(np.uint8)).resize((w, h), Image.NEAREST))

    # 转换为uint8
    lr_display = lr_resized
    sr_display = (sr_img * 255).clip(0, 255).astype(np.uint8)
    hr_display = (hr_img * 255).clip(0, 255).astype(np.uint8)

    # 创建对比图：LR | SR | HR 横向排列
    gap = 10  # 图像间隔
    title_height = 40  # 标题高度

    total_width = w * 3 + gap * 2
    total_height = h + title_height

    # 创建白色背景
    comparison = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # 粘贴图像
    comparison.paste(Image.fromarray(lr_display), (0, title_height))
    comparison.paste(Image.fromarray(sr_display), (w + gap, title_height))
    comparison.paste(Image.fromarray(hr_display), (w * 2 + gap * 2, title_height))

    # 添加标题
    draw = ImageDraw.Draw(comparison)

    # 尝试使用系统字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    # 绘制标题
    draw.text((w//2 - 50, 10), f"LR (64x64)", fill=(0, 0, 0), font=font)
    draw.text((w + gap + w//2 - 80, 10), f"SR (PSNR:{psnr:.1f})", fill=(0, 0, 255), font=font)
    draw.text((w*2 + gap*2 + w//2 - 60, 10), f"HR (GT)", fill=(0, 128, 0), font=font)

    return comparison


def build_model(configs, ckpt_path, device):
    """构建模型并加载权重"""
    # 构建扩散模型
    diffusion = util_common.instantiate_from_config(configs.diffusion)

    # 构建UNet
    model = util_common.instantiate_from_config(configs.model).to(device)

    # 加载训练好的权重
    print(f'Loading model from {ckpt_path}...')
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        util_net.reload_model(model, ckpt['state_dict'])
    else:
        util_net.reload_model(model, ckpt)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 构建autoencoder (DWT)
    autoencoder = None
    if configs.autoencoder is not None:
        autoencoder = util_common.instantiate_from_config(configs.autoencoder).to(device)
        autoencoder.eval()

    return model, diffusion, autoencoder


@torch.no_grad()
def inference_single(model, diffusion, autoencoder, lr_img, device, configs):
    """
    对单张LR图像进行超分
    lr_img: numpy array, [H, W, 3], range [0, 1]
    返回: numpy array, [H*4, W*4, 3], range [0, 1]
    """
    # 转换为tensor并归一化到[-1, 1]
    lr_tensor = torch.from_numpy(lr_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    lr_tensor = (lr_tensor - 0.5) / 0.5

    # 准备model_kwargs
    model_kwargs = {'lq': lr_tensor} if configs.model.params.get('cond_lq', True) else {}

    # DDIM采样
    sr_tensor = diffusion.ddim_sample_loop(
        y=lr_tensor,
        model=model,
        first_stage_model=autoencoder,
        clip_denoised=True if autoencoder is None else False,
        model_kwargs=model_kwargs,
        progress=False,
    )

    # 转换回[0, 1]范围
    sr_tensor = sr_tensor * 0.5 + 0.5
    sr_tensor = sr_tensor.clamp(0, 1)

    # 转换为numpy
    sr_img = sr_tensor[0].cpu().numpy().transpose(1, 2, 0)

    return sr_img


def main():
    parser = argparse.ArgumentParser(description='DifIISR-DWT Inference and Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--lr_dir', type=str, required=True, help='LR images directory')
    parser.add_argument('--hr_dir', type=str, default=None, help='HR images directory (for evaluation)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for SR images')
    parser.add_argument('--num_images', type=int, default=None, help='Number of images to process')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载配置
    configs = load_yaml_config(args.config)

    # 构建模型
    model, diffusion, autoencoder = build_model(configs, args.ckpt, device)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建对比图目录
    comparison_dir = output_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # 获取LR图像列表
    lr_dir = Path(args.lr_dir)
    lr_files = sorted(list(lr_dir.glob('*.png')) + list(lr_dir.glob('*.jpg')))

    if args.num_images:
        lr_files = lr_files[:args.num_images]

    print(f'Processing {len(lr_files)} images...')

    # 评估指标
    psnr_list = []
    ssim_list = []

    hr_dir = Path(args.hr_dir) if args.hr_dir else None

    for lr_path in tqdm(lr_files, desc='Inference'):
        # 读取LR图像
        lr_img = np.array(Image.open(lr_path).convert('RGB')).astype(np.float32) / 255.0

        # 超分
        sr_img = inference_single(model, diffusion, autoencoder, lr_img, device, configs)

        # 保存SR图像
        sr_path = output_dir / lr_path.name
        sr_img_uint8 = (sr_img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(sr_img_uint8).save(sr_path)

        # 如果有HR图像，计算指标并生成对比图
        psnr = 0
        ssim = 0
        if hr_dir:
            hr_path = hr_dir / lr_path.name
            if hr_path.exists():
                hr_img = np.array(Image.open(hr_path).convert('RGB')).astype(np.float32) / 255.0

                # 确保尺寸匹配
                if sr_img.shape == hr_img.shape:
                    psnr = calculate_psnr(sr_img, hr_img)
                    ssim = calculate_ssim_color(sr_img, hr_img)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    # 生成对比图
                    comparison = create_comparison_image(lr_img, sr_img, hr_img, psnr, ssim, lr_path.name)
                    comparison.save(comparison_dir / f'cmp_{lr_path.stem}.png')

    # 输出结果
    print(f'\nResults saved to {output_dir}')
    print(f'Comparison images saved to {comparison_dir}')

    if psnr_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(f'\n===== Evaluation Results =====')
        print(f'Number of images: {len(psnr_list)}')
        print(f'Average PSNR: {avg_psnr:.2f} dB')
        print(f'Average SSIM: {avg_ssim:.4f}')

        # 保存结果到文件
        with open(output_dir / 'metrics.txt', 'w') as f:
            f.write(f'Number of images: {len(psnr_list)}\n')
            f.write(f'Average PSNR: {avg_psnr:.2f} dB\n')
            f.write(f'Average SSIM: {avg_ssim:.4f}\n')
            f.write(f'\nPer-image results:\n')
            for i, (p, s) in enumerate(zip(psnr_list, ssim_list)):
                f.write(f'{lr_files[i].name}: PSNR={p:.2f}, SSIM={s:.4f}\n')


if __name__ == '__main__':
    main()
