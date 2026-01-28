"""
使用FMB数据集验证DifIISR-DWT模型
输出 PSNR, SSIM, LPIPS 指标
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import lpips
from PIL import Image, ImageDraw, ImageFont
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


def calculate_lpips(img1, img2, lpips_fn, device):
    """计算LPIPS (输入为numpy数组, 范围[0,1])"""
    # 转换为tensor, 范围[-1, 1]
    img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    img1_tensor = img1_tensor * 2 - 1  # [0,1] -> [-1,1]
    img2_tensor = img2_tensor * 2 - 1

    with torch.no_grad():
        lpips_val = lpips_fn(img1_tensor, img2_tensor).item()
    return lpips_val


def create_comparison_image(lr_img, sr_img, hr_img, psnr, ssim, lpips_val, filename):
    """创建LR/SR/HR对比图"""
    h, w = hr_img.shape[:2]
    lr_h, lr_w = lr_img.shape[:2]

    # 将LR图像放大到与HR相同尺寸（用于对比显示）
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


def build_model(configs, ckpt_path, device):
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

    autoencoder = None
    if configs.autoencoder is not None:
        autoencoder = util_common.instantiate_from_config(configs.autoencoder).to(device)
        autoencoder.eval()

    return model, diffusion, autoencoder


@torch.no_grad()
def inference_single(model, diffusion, autoencoder, lr_img, device, configs):
    """对单张LR图像进行超分，支持任意尺寸输入"""
    h, w = lr_img.shape[:2]

    # UNet有4层下采样，DWT有2级分解
    # 潜空间尺寸 = LR尺寸 / 4 (DWT)
    # UNet最小特征图 = 潜空间尺寸 / 8 (4层下采样)
    # Swin window_size = 8
    # 所以潜空间需要是 8*8=64 的倍数
    # LR图像需要是 64*4=256 的倍数
    # 但这太大了，改用64的倍数试试（潜空间16的倍数）
    multiple = 64
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    # Padding
    if pad_h > 0 or pad_w > 0:
        lr_img_padded = np.pad(lr_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        lr_img_padded = lr_img

    lr_tensor = torch.from_numpy(lr_img_padded.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    lr_tensor = (lr_tensor - 0.5) / 0.5

    model_kwargs = {'lq': lr_tensor} if configs.model.params.get('cond_lq', True) else {}

    sr_tensor = diffusion.ddim_sample_loop(
        y=lr_tensor,
        model=model,
        first_stage_model=autoencoder,
        clip_denoised=True if autoencoder is None else False,
        model_kwargs=model_kwargs,
        progress=False,
    )

    sr_tensor = sr_tensor * 0.5 + 0.5
    sr_tensor = sr_tensor.clamp(0, 1)
    sr_img = sr_tensor[0].cpu().numpy().transpose(1, 2, 0)

    # 裁剪回原始尺寸（4倍超分）
    sr_img = sr_img[:h*4, :w*4, :]

    return sr_img


def main():
    parser = argparse.ArgumentParser(description='DifIISR-DWT FMB Evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    configs = load_yaml_config(args.config)
    model, diffusion, autoencoder = build_model(configs, args.ckpt, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = output_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # FMB数据集路径
    lr_dir = Path('/home/lch/sr_recons/FMB/test/Infrared_LR')
    hr_dir = Path('/home/lch/sr_recons/FMB/test/Infrared')

    # 获取LR图像列表
    lr_files = sorted(list(lr_dir.glob('*.png')))[:args.num_images]

    print(f'Processing {len(lr_files)} images...')

    # 初始化LPIPS模型
    print('Loading LPIPS model...')
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn.eval()

    psnr_list = []
    ssim_list = []
    lpips_list = []

    for lr_path in tqdm(lr_files, desc='Inference'):
        # FMB的LR文件名是 00043x4.png，对应HR是 00043.png
        hr_name = lr_path.stem.replace('x4', '') + '.png'
        hr_path = hr_dir / hr_name

        # 读取LR图像
        lr_img = np.array(Image.open(lr_path).convert('RGB')).astype(np.float32) / 255.0

        # 超分
        sr_img = inference_single(model, diffusion, autoencoder, lr_img, device, configs)

        # 保存SR图像
        sr_path = output_dir / f'sr_{lr_path.stem}.png'
        sr_img_uint8 = (sr_img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(sr_img_uint8).save(sr_path)

        # 计算指标
        psnr = 0
        ssim = 0
        lpips_val = 0
        if hr_path.exists():
            hr_img = np.array(Image.open(hr_path).convert('RGB')).astype(np.float32) / 255.0

            # 确保尺寸匹配
            if sr_img.shape == hr_img.shape:
                psnr = calculate_psnr(sr_img, hr_img)
                ssim = calculate_ssim_color(sr_img, hr_img)
                lpips_val = calculate_lpips(sr_img, hr_img, lpips_fn, device)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                lpips_list.append(lpips_val)

                # 生成对比图
                comparison = create_comparison_image(lr_img, sr_img, hr_img, psnr, ssim, lpips_val, lr_path.name)
                comparison.save(comparison_dir / f'cmp_{lr_path.stem}.png')
            else:
                print(f'Size mismatch: SR {sr_img.shape} vs HR {hr_img.shape}')

    print(f'\nResults saved to {output_dir}')
    print(f'Comparison images saved to {comparison_dir}')

    if psnr_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list)
        print(f'\n===== FMB Evaluation Results =====')
        print(f'Number of images: {len(psnr_list)}')
        print(f'Average PSNR:  {avg_psnr:.2f} dB')
        print(f'Average SSIM:  {avg_ssim:.4f}')
        print(f'Average LPIPS: {avg_lpips:.4f}')

        with open(output_dir / 'metrics.txt', 'w') as f:
            f.write(f'FMB Dataset Evaluation\n')
            f.write(f'Number of images: {len(psnr_list)}\n')
            f.write(f'Average PSNR:  {avg_psnr:.2f} dB\n')
            f.write(f'Average SSIM:  {avg_ssim:.4f}\n')
            f.write(f'Average LPIPS: {avg_lpips:.4f}\n')
            f.write(f'\nPer-image results:\n')
            for i, (p, s, l) in enumerate(zip(psnr_list, ssim_list, lpips_list)):
                f.write(f'{lr_files[i].name}: PSNR={p:.2f}, SSIM={s:.4f}, LPIPS={l:.4f}\n')


if __name__ == '__main__':
    main()
