"""
测试全子带DWT模型 (方案A)
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils import util_net
from utils import util_common


def load_yaml_config(config_path):
    import yaml
    from omegaconf import OmegaConf
    with open(config_path, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    return config


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100
    return -10 * np.log10(mse)


def calculate_ssim(img1, img2):
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
    ssim_vals = []
    for i in range(img1.shape[2]):
        ssim_vals.append(calculate_ssim(img1[:, :, i], img2[:, :, i]))
    return np.mean(ssim_vals)


def usm_sharpen(img, sigma=1.0, strength=1.5):
    """USM锐化 (Unsharp Mask)"""
    # img: [H, W, C], float32, [0, 1]
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(img_uint8, (0, 0), sigma)
    sharp = cv2.addWeighted(img_uint8, 1 + strength, blur, -strength, 0)
    return sharp.astype(np.float32) / 255.0


def high_pass_sharpen(img, kernel_size=3, strength=0.5):
    """高通滤波锐化"""
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    # 拉普拉斯高通
    laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
    sharp = img_uint8.astype(np.float64) + strength * laplacian
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp.astype(np.float32) / 255.0


def build_model(configs, ckpt_path, device):
    """构建模型"""
    diffusion = util_common.instantiate_from_config(configs.diffusion)
    model = util_common.instantiate_from_config(configs.model).to(device)

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
    """对单张LR图像进行超分"""
    h, w = lr_img.shape[:2]

    # 确保尺寸是64的倍数（DWT 2级分解 + UNet下采样）
    multiple = 64
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        lr_img_padded = np.pad(lr_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        lr_img_padded = lr_img

    # 转换为tensor，归一化到[-1, 1]
    lr_tensor = torch.from_numpy(lr_img_padded.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    lr_tensor = (lr_tensor - 0.5) / 0.5

    # 准备条件
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

    # 转换回[0, 1]
    sr_tensor = sr_tensor * 0.5 + 0.5
    sr_tensor = sr_tensor.clamp(0, 1)
    sr_img = sr_tensor[0].cpu().numpy().transpose(1, 2, 0)

    # 裁剪回原始尺寸（4倍超分）
    sr_img = sr_img[:h*4, :w*4, :]

    return sr_img


def main():
    parser = argparse.ArgumentParser(description='Test DifIISR-DWT FullBand Model')
    parser.add_argument('--config', type=str, default='configs/train_m3fd_dwt_fullband.yaml',
                        help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, default='results/fullband_test',
                        help='Output directory')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Ground truth directory for computing metrics')
    parser.add_argument('--sharpen', type=str, default=None, choices=['usm', 'highpass', 'both'],
                        help='Apply sharpening: usm, highpass, or both')
    parser.add_argument('--sharpen_strength', type=float, default=1.0,
                        help='Sharpening strength (default: 1.0)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载配置和模型
    configs = load_yaml_config(args.config)
    model, diffusion, autoencoder = build_model(configs, args.ckpt, device)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取输入图像列表
    input_path = Path(args.input)
    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = sorted(list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')))

    print(f'Processing {len(input_files)} images...')

    psnr_list = []
    ssim_list = []

    for img_path in tqdm(input_files, desc='Inference'):
        # 读取LR图像
        lr_img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0

        # 超分
        sr_img = inference_single(model, diffusion, autoencoder, lr_img, device, configs)

        # 应用锐化（如果指定）
        sr_img_sharp = sr_img
        if args.sharpen:
            if args.sharpen == 'usm':
                sr_img_sharp = usm_sharpen(sr_img, sigma=1.0, strength=args.sharpen_strength)
            elif args.sharpen == 'highpass':
                sr_img_sharp = high_pass_sharpen(sr_img, strength=args.sharpen_strength)
            elif args.sharpen == 'both':
                sr_img_sharp = usm_sharpen(sr_img, sigma=1.0, strength=args.sharpen_strength * 0.7)
                sr_img_sharp = high_pass_sharpen(sr_img_sharp, strength=args.sharpen_strength * 0.3)

        # 保存SR图像（原始）
        sr_path = output_dir / f'sr_{img_path.stem}.png'
        sr_img_uint8 = (sr_img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(sr_img_uint8).save(sr_path)

        # 保存锐化后的图像（如果有）
        if args.sharpen:
            sr_sharp_path = output_dir / f'sr_{img_path.stem}_sharp.png'
            sr_sharp_uint8 = (sr_img_sharp * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(sr_sharp_uint8).save(sr_sharp_path)

        # 如果有GT，计算指标
        if args.gt_dir:
            gt_dir = Path(args.gt_dir)
            # 尝试多种命名方式
            gt_candidates = [
                gt_dir / img_path.name,
                gt_dir / (img_path.stem.replace('x4', '') + '.png'),
                gt_dir / (img_path.stem.replace('_LR', '') + '.png'),
            ]

            gt_path = None
            for candidate in gt_candidates:
                if candidate.exists():
                    gt_path = candidate
                    break

            if gt_path:
                hr_img = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32) / 255.0

                if sr_img.shape == hr_img.shape:
                    # 原始SR指标
                    psnr = calculate_psnr(sr_img, hr_img)
                    ssim = calculate_ssim_color(sr_img, hr_img)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    # 锐化后指标
                    if args.sharpen:
                        psnr_sharp = calculate_psnr(sr_img_sharp, hr_img)
                        ssim_sharp = calculate_ssim_color(sr_img_sharp, hr_img)
                        print(f'{img_path.name}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f} | Sharp: PSNR={psnr_sharp:.2f}dB, SSIM={ssim_sharp:.4f}')
                    else:
                        print(f'{img_path.name}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}')
                else:
                    print(f'Size mismatch: SR {sr_img.shape} vs HR {hr_img.shape}')

    print(f'\nResults saved to {output_dir}')

    if psnr_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(f'\n===== Evaluation Results =====')
        print(f'Number of images: {len(psnr_list)}')
        print(f'Average PSNR: {avg_psnr:.2f} dB')
        print(f'Average SSIM: {avg_ssim:.4f}')

        # 保存结果
        with open(output_dir / 'metrics.txt', 'w') as f:
            f.write(f'FullBand DWT Model Evaluation\n')
            f.write(f'Checkpoint: {args.ckpt}\n')
            f.write(f'Number of images: {len(psnr_list)}\n')
            f.write(f'Average PSNR: {avg_psnr:.2f} dB\n')
            f.write(f'Average SSIM: {avg_ssim:.4f}\n')


if __name__ == '__main__':
    main()
