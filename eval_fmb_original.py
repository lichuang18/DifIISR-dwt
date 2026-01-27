"""
使用FMB数据集测试DifIISR原版（VAE版本）性能
直接使用原版的DifIISRSampler
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from sampler import DifIISRSampler
from utils import util_image


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


def create_comparison_image(lr_img, sr_img, hr_img, psnr, ssim, filename):
    """创建LR/SR/HR对比图"""
    h, w = hr_img.shape[:2]
    lr_h, lr_w = lr_img.shape[:2]

    lr_resized = np.array(Image.fromarray((lr_img * 255).astype(np.uint8)).resize((w, h), Image.NEAREST))

    lr_display = lr_resized
    sr_display = (sr_img * 255).clip(0, 255).astype(np.uint8)
    hr_display = (hr_img * 255).clip(0, 255).astype(np.uint8)

    gap = 10
    title_height = 40
    total_width = w * 3 + gap * 2
    total_height = h + title_height

    comparison = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    comparison.paste(Image.fromarray(lr_display), (0, title_height))
    comparison.paste(Image.fromarray(sr_display), (w + gap, title_height))
    comparison.paste(Image.fromarray(hr_display), (w * 2 + gap * 2, title_height))

    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw.text((w//2 - 60, 10), f"LR ({lr_w}x{lr_h})", fill=(0, 0, 0), font=font)
    draw.text((w + gap + w//2 - 100, 10), f"SR (PSNR:{psnr:.1f}dB)", fill=(0, 0, 255), font=font)
    draw.text((w*2 + gap*2 + w//2 - 60, 10), f"HR ({w}x{h})", fill=(0, 128, 0), font=font)

    return comparison


def main():
    parser = argparse.ArgumentParser(description='DifIISR Original (VAE) FMB Evaluation')
    parser.add_argument('--output_dir', type=str, default='results/DifIISR_original_fmb')
    parser.add_argument('--num_images', type=int, default=20)
    parser.add_argument('--chop_size', type=int, default=256, choices=[256, 512])
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    # 固定路径
    config_path = 'configs/DifIISR_test.yaml'
    ckpt_path = 'weights/DifIISR.pth'
    vae_path = 'weights/autoencoder_vq_f4.pth'

    # 加载配置并设置权重路径
    configs = OmegaConf.load(config_path)
    configs.model.ckpt_path = ckpt_path
    configs.autoencoder.ckpt_path = vae_path

    print(f'Config: {config_path}')
    print(f'UNet weights: {ckpt_path}')
    print(f'VAE weights: {vae_path}')

    # 设置chop参数
    if args.chop_size == 512:
        chop_stride = 448
    else:
        chop_stride = 224

    # 创建sampler
    sampler = DifIISRSampler(
        configs,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=True,
        seed=args.seed,
        ddim=True
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = output_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # FMB数据集路径
    lr_dir = Path('/home/lch/sr_recons/FMB/test/Infrared_LR')
    hr_dir = Path('/home/lch/sr_recons/FMB/test/Infrared')

    lr_files = sorted(list(lr_dir.glob('*.png')))[:args.num_images]

    print(f'Processing {len(lr_files)} images...')

    psnr_list = []
    ssim_list = []

    for lr_path in tqdm(lr_files, desc='Inference'):
        hr_name = lr_path.stem.replace('x4', '') + '.png'
        hr_path = hr_dir / hr_name

        # 读取LR图像
        lr_img = util_image.imread(lr_path, chn='rgb', dtype='float32')  # [0, 1], RGB
        lr_tensor = util_image.img2tensor(lr_img).cuda()  # [1, 3, H, W], [0, 1]

        # 使用sampler进行超分
        sr_tensor = sampler.sample_func(
            (lr_tensor - 0.5) / 0.5,  # 归一化到[-1, 1]
            noise_repeat=False,
            one_step=True
        )
        sr_tensor = sr_tensor * 0.5 + 0.5  # 反归一化到[0, 1]
        sr_tensor = sr_tensor.clamp(0, 1)

        # 转换为numpy
        sr_img = sr_tensor[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]

        # 保存SR图像
        sr_path = output_dir / f'sr_{lr_path.stem}.png'
        sr_img_uint8 = (sr_img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(sr_img_uint8).save(sr_path)

        # 计算指标
        psnr = 0
        ssim = 0
        if hr_path.exists():
            hr_img = np.array(Image.open(hr_path).convert('RGB')).astype(np.float32) / 255.0

            if sr_img.shape == hr_img.shape:
                psnr = calculate_psnr(sr_img, hr_img)
                ssim = calculate_ssim_color(sr_img, hr_img)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                # 生成对比图
                comparison = create_comparison_image(lr_img, sr_img, hr_img, psnr, ssim, lr_path.name)
                comparison.save(comparison_dir / f'cmp_{lr_path.stem}.png')
            else:
                print(f'Size mismatch: SR {sr_img.shape} vs HR {hr_img.shape}')

    print(f'\nResults saved to {output_dir}')
    print(f'Comparison images saved to {comparison_dir}')

    if psnr_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(f'\n===== DifIISR Original (VAE) FMB Results =====')
        print(f'Number of images: {len(psnr_list)}')
        print(f'Average PSNR: {avg_psnr:.2f} dB')
        print(f'Average SSIM: {avg_ssim:.4f}')

        with open(output_dir / 'metrics.txt', 'w') as f:
            f.write(f'DifIISR Original (VAE) FMB Evaluation\n')
            f.write(f'Config: {config_path}\n')
            f.write(f'UNet: {ckpt_path}\n')
            f.write(f'VAE: {vae_path}\n')
            f.write(f'\nNumber of images: {len(psnr_list)}\n')
            f.write(f'Average PSNR: {avg_psnr:.2f} dB\n')
            f.write(f'Average SSIM: {avg_ssim:.4f}\n')
            f.write(f'\nPer-image results:\n')
            for i, (p, s) in enumerate(zip(psnr_list, ssim_list)):
                f.write(f'{lr_files[i].name}: PSNR={p:.2f}, SSIM={s:.4f}\n')


if __name__ == '__main__':
    main()
