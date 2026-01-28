"""
DifIISR 训练脚本
使用M3FD红外数据集训练扩散模型
"""

import os
import sys
import math
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from utils import util_net
from utils import util_image
from utils import util_common
from datapipe.paired_dataset import PairedImageDataset, ValPairedImageDataset
from datapipe.degradation_dataset import DegradationDatasetFromPatches, ValDegradationDataset
from ldm.modules.ema import LitEma

# 尝试导入LPIPS
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not installed. Perceptual loss will be disabled.")


def setup_logger(log_dir):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml_config(config_path):
    """加载YAML配置文件"""
    import yaml
    from omegaconf import OmegaConf
    with open(config_path, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    return config


def build_model(configs, device):
    """构建模型"""
    # 构建扩散模型
    diffusion = util_common.instantiate_from_config(configs.diffusion)

    # 构建UNet
    model = util_common.instantiate_from_config(configs.model).to(device)

    # 加载预训练权重（如果有）
    if configs.model.get('ckpt_path') and configs.model.ckpt_path:
        ckpt_path = configs.model.ckpt_path
        logging.info(f'Loading model from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'state_dict' in ckpt:
            util_net.reload_model(model, ckpt['state_dict'])
        else:
            util_net.reload_model(model, ckpt)

    # 构建autoencoder
    autoencoder = None
    if configs.autoencoder is not None:
        ckpt_path = configs.autoencoder.get('ckpt_path', None)
        autoencoder = util_common.instantiate_from_config(configs.autoencoder).to(device)

        # 只有当ckpt_path存在时才加载权重 (DWT不需要权重)
        if ckpt_path:
            logging.info(f'Loading AutoEncoder from {ckpt_path}...')
            ckpt = torch.load(ckpt_path, map_location=device)
            if 'state_dict' in ckpt:
                util_net.reload_model(autoencoder, ckpt['state_dict'])
            else:
                util_net.reload_model(autoencoder, ckpt)
        else:
            logging.info(f'Using AutoEncoder without pretrained weights (e.g., DWT)')

        # 检查autoencoder是否有可学习参数
        num_ae_params = sum(p.numel() for p in autoencoder.parameters())
        if num_ae_params > 0:
            # 有可学习参数（如DWTModelAllBandsLearnable），保持训练模式
            logging.info(f'AutoEncoder has {num_ae_params} learnable parameters, keeping trainable')
            autoencoder.train()
        else:
            # 无可学习参数（如DWTModelAllBands），冻结
            logging.info(f'AutoEncoder has no learnable parameters, freezing')
            autoencoder.eval()
            for param in autoencoder.parameters():
                param.requires_grad = False

        if configs.autoencoder.get('use_fp16', False):
            autoencoder = autoencoder.half()

    return model, diffusion, autoencoder


def build_dataloader(configs):
    """构建数据加载器"""
    # 检查是否使用在线退化
    data_type = configs.data.get('type', 'paired')

    if data_type == 'degradation':
        # 使用在线退化数据集
        # 获取退化配置
        degradation_config = None
        if hasattr(configs, 'degradation'):
            degradation_config = dict(configs.degradation)

        train_dataset = DegradationDatasetFromPatches(
            hr_dir=configs.data.train.hr_dir,
            gt_size=configs.data.train.get('gt_size', 256),
            scale=configs.data.train.get('scale', 4),
            use_hflip=configs.data.train.get('use_hflip', True),
            use_rot=configs.data.train.get('use_rot', True),
            mean=0.5,
            std=0.5,
            length=configs.data.train.get('length', None),
            degradation_config=degradation_config,
        )

        val_dataset = ValDegradationDataset(
            hr_dir=configs.data.val.hr_dir,
            scale=configs.data.train.get('scale', 4),
            mean=0.5,
            std=0.5,
            length=configs.data.val.get('length', 100),
            degradation_config=degradation_config,
        )
    else:
        # 使用预生成的配对数据集
        train_dataset = PairedImageDataset(
            hr_dir=configs.data.train.hr_dir,
            lr_dir=configs.data.train.lr_dir,
            gt_size=configs.data.train.get('gt_size', 256),
            scale=configs.data.train.get('scale', 4),
            use_hflip=configs.data.train.get('use_hflip', True),
            use_rot=configs.data.train.get('use_rot', True),
            mean=0.5,
            std=0.5,
            length=configs.data.train.get('length', None),
        )

        val_dataset = ValPairedImageDataset(
            hr_dir=configs.data.val.hr_dir,
            lr_dir=configs.data.val.lr_dir,
            mean=0.5,
            std=0.5,
            length=configs.data.val.get('length', 100),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(configs.train.batch_size),
        shuffle=True,
        num_workers=int(configs.train.get('num_workers', 4)),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(configs.train.get('val_batch_size', 4)),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_lpips_weight_schedule(iteration, total_iterations, configs):
    """
    渐进式LPIPS权重调度

    策略：
    - 前期：LPIPS权重小，MSE主导，让模型先学会基本重建
    - 中期：逐渐增大LPIPS权重
    - 后期：LPIPS与MSE接近等权，提升视觉质量

    权重计算：
    - 起始权重: lpips_weight_start (默认0.01)
    - 结束权重: lpips_weight_end (默认0.15)
    - 线性插值: weight = start + (end - start) * (iter / total)

    例如 total=100K:
    - 0K:   0.01 (LPIPS贡献约0.0003, MSE≈0.01, 比例1:33)
    - 50K:  0.08 (LPIPS贡献约0.0024, MSE≈0.005, 比例1:2)
    - 100K: 0.15 (LPIPS贡献约0.0045, MSE≈0.005, 比例1:1)
    """
    lpips_config = configs.get('perceptual_loss', {})
    start_weight = lpips_config.get('lpips_weight_start', 0.01)
    end_weight = lpips_config.get('lpips_weight_end', 0.15)

    # 线性插值
    progress = min(iteration / total_iterations, 1.0)
    weight = start_weight + (end_weight - start_weight) * progress

    return weight


class SobelEdgeLoss(nn.Module):
    """Sobel边缘损失 - 增强边缘锐度"""
    def __init__(self):
        super().__init__()
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # 扩展为卷积核 [out_ch, in_ch, H, W]
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        """
        计算边缘损失
        pred, target: [B, C, H, W], 范围[-1, 1]
        """
        # 转换到[0, 1]范围
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        loss = 0
        for c in range(pred.shape[1]):
            pred_c = pred[:, c:c+1, :, :]
            target_c = target[:, c:c+1, :, :]

            # 计算边缘
            pred_edge_x = F.conv2d(pred_c, self.sobel_x, padding=1)
            pred_edge_y = F.conv2d(pred_c, self.sobel_y, padding=1)
            pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)

            target_edge_x = F.conv2d(target_c, self.sobel_x, padding=1)
            target_edge_y = F.conv2d(target_c, self.sobel_y, padding=1)
            target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-8)

            # L1损失
            loss = loss + F.l1_loss(pred_edge, target_edge)

        return loss / pred.shape[1]


class FFTFrequencyLoss(nn.Module):
    """FFT频率损失 - 增强高频细节"""
    def __init__(self, weight_high_freq=1.0):
        super().__init__()
        self.weight_high_freq = weight_high_freq

    def forward(self, pred, target):
        """
        计算频率域损失
        pred, target: [B, C, H, W], 范围[-1, 1]
        """
        # 转换到[0, 1]范围
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        # FFT变换
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')

        # 计算幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # 创建高频权重掩码（中心是低频，边缘是高频）
        B, C, H, W = pred.shape
        # 将零频移到中心
        pred_mag_shifted = torch.fft.fftshift(pred_mag, dim=(-2, -1))
        target_mag_shifted = torch.fft.fftshift(target_mag, dim=(-2, -1))

        # 创建高频加权掩码
        y_coords = torch.linspace(-1, 1, H, device=pred.device)
        x_coords = torch.linspace(-1, 1, W, device=pred.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        dist_from_center = torch.sqrt(xx**2 + yy**2)

        # 高频权重：距离中心越远权重越大
        high_freq_weight = 1.0 + self.weight_high_freq * dist_from_center
        high_freq_weight = high_freq_weight.view(1, 1, H, W)

        # 加权L1损失
        loss = F.l1_loss(pred_mag_shifted * high_freq_weight,
                         target_mag_shifted * high_freq_weight)

        return loss


def train_one_step(model, diffusion, autoencoder, batch, optimizer, device, configs,
                   lpips_fn=None, edge_loss_fn=None, freq_loss_fn=None,
                   iteration=0, total_iterations=100000):
    """训练一步"""
    model.train()

    gt = batch['gt'].to(device)  # HR图像 [B, 3, 256, 256]
    lq = batch['lq'].to(device)  # LR图像 [B, 3, 64, 64]

    # 随机采样时间步
    t = torch.randint(0, diffusion.num_timesteps, (gt.shape[0],), device=device).long()

    # 准备model_kwargs
    model_kwargs = {'lq': lq} if configs.model.params.get('cond_lq', True) else {}

    # 计算损失
    losses, z_t, pred_zstart = diffusion.training_losses(
        model=model,
        x_start=gt,
        y=lq,
        t=t,
        first_stage_model=autoencoder,
        model_kwargs=model_kwargs,
    )

    loss = losses['loss'].mean()
    lpips_loss_val = 0.0
    edge_loss_val = 0.0
    freq_loss_val = 0.0
    current_lpips_weight = 0.0

    # 获取感知损失配置
    perceptual_config = configs.get('perceptual_loss', {})

    # 检查是否需要解码pred_img（LPIPS、边缘损失、频率损失都需要）
    need_decode = (
        (lpips_fn is not None and perceptual_config.get('enabled', False)) or
        (edge_loss_fn is not None and perceptual_config.get('edge_loss_enabled', False)) or
        (freq_loss_fn is not None and perceptual_config.get('freq_loss_enabled', False))
    )

    if need_decode and pred_zstart is not None:
        # 解码预测图像（只解码一次）
        pred_img = diffusion.decode_first_stage(pred_zstart, autoencoder, no_grad=False)
        pred_img = pred_img.clamp(-1, 1)

        # 时间步阈值（用于LPIPS动态权重）
        T = diffusion.num_timesteps  # 15
        t1 = T // 3      # 5
        t2 = 2 * T // 3  # 10

        # 计算每个样本的时间步权重（放宽版本）
        # t < t1: weight = 1.0
        # t1 <= t < t2: weight = 0.5 (原来是0.1)
        # t >= t2: weight = 0.1 (原来是0.001)
        t_weights = torch.ones(t.shape[0], device=device)
        t_weights[(t >= t1) & (t < t2)] = 0.5
        t_weights[t >= t2] = 0.1

        # 1. LPIPS损失
        if lpips_fn is not None and perceptual_config.get('enabled', False):
            base_weight = get_lpips_weight_schedule(iteration, total_iterations, configs)
            current_lpips_weight = base_weight

            lpips_per_sample = lpips_fn(pred_img, gt).view(-1)
            weighted_lpips = (lpips_per_sample * t_weights).mean()
            lpips_loss_val = weighted_lpips.item()

            loss = loss + base_weight * weighted_lpips

        # 2. 边缘损失（Sobel）
        if edge_loss_fn is not None and perceptual_config.get('edge_loss_enabled', False):
            edge_weight = perceptual_config.get('edge_loss_weight', 0.1)
            edge_loss = edge_loss_fn(pred_img, gt)
            edge_loss_val = edge_loss.item()

            # 边缘损失也使用时间步权重（但不那么激进）
            edge_t_weight = t_weights.mean()  # 使用batch平均权重
            loss = loss + edge_weight * edge_t_weight * edge_loss

        # 3. 频率损失（FFT）
        if freq_loss_fn is not None and perceptual_config.get('freq_loss_enabled', False):
            freq_weight = perceptual_config.get('freq_loss_weight', 0.1)
            freq_loss = freq_loss_fn(pred_img, gt)
            freq_loss_val = freq_loss.item()

            # 频率损失也使用时间步权重
            freq_t_weight = t_weights.mean()
            loss = loss + freq_weight * freq_t_weight * freq_loss

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 梯度裁剪
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    result = {
        'loss': loss.item(),
        'mse': losses['mse'].mean().item(),
        'grad_norm': grad_norm.item(),
    }

    if lpips_fn is not None:
        result['lpips'] = lpips_loss_val
        result['lpips_weight'] = current_lpips_weight

    if edge_loss_fn is not None:
        result['edge'] = edge_loss_val

    if freq_loss_fn is not None:
        result['freq'] = freq_loss_val

    return result


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100
    return -10 * np.log10(mse)


def calculate_ssim(img1, img2):
    """计算SSIM (简化版本，单通道)"""
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
    """计算彩色图像的SSIM (对每个通道计算后取平均)"""
    ssim_vals = []
    for i in range(img1.shape[2]):
        ssim_vals.append(calculate_ssim(img1[:, :, i], img2[:, :, i]))
    return np.mean(ssim_vals)


@torch.no_grad()
def validate(model, diffusion, autoencoder, val_loader, device, configs, lpips_fn=None):
    """验证"""
    model.eval()

    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    count = 0

    for batch in val_loader:
        gt = batch['gt'].to(device)
        lq = batch['lq'].to(device)

        # 使用DDIM采样
        model_kwargs = {'lq': lq} if configs.model.params.get('cond_lq', True) else {}

        # 采样
        sr = diffusion.ddim_sample_loop(
            y=lq,
            model=model,
            first_stage_model=autoencoder,
            clip_denoised=True if autoencoder is None else False,
            model_kwargs=model_kwargs,
            progress=False,
        )

        # 计算LPIPS（在[-1, 1]范围内计算）
        if lpips_fn is not None:
            lpips_val = lpips_fn(sr.clamp(-1, 1), gt).mean().item()
            total_lpips += lpips_val * sr.shape[0]

        # 转换到[0, 1]范围
        sr = sr * 0.5 + 0.5
        gt = gt * 0.5 + 0.5

        # 计算PSNR和SSIM
        for i in range(sr.shape[0]):
            sr_np = sr[i].cpu().numpy().transpose(1, 2, 0)
            gt_np = gt[i].cpu().numpy().transpose(1, 2, 0)

            # 裁剪到[0, 1]
            sr_np = np.clip(sr_np, 0, 1)
            gt_np = np.clip(gt_np, 0, 1)

            # 计算PSNR
            psnr = calculate_psnr(sr_np, gt_np)
            total_psnr += psnr

            # 计算SSIM
            ssim = calculate_ssim_color(sr_np, gt_np)
            total_ssim += ssim

            count += 1

    avg_psnr = total_psnr / count if count > 0 else 0
    avg_ssim = total_ssim / count if count > 0 else 0
    avg_lpips = total_lpips / count if count > 0 else 0

    return {'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips}


def save_checkpoint(model, ema, optimizer, scheduler, iteration, save_path, autoencoder=None):
    """保存检查点"""
    state = {
        'iteration': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if ema is not None:
        state['ema'] = ema.state_dict()
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    # 保存autoencoder的可学习参数（如果有）
    if autoencoder is not None:
        ae_params = [p for p in autoencoder.parameters() if p.requires_grad]
        if ae_params:
            state['autoencoder'] = autoencoder.state_dict()

    torch.save(state, save_path)


def main():
    parser = argparse.ArgumentParser(description='DifIISR Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # 设置随机种子
    setup_seed(args.seed)

    # 加载配置
    configs = load_yaml_config(args.config)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = configs.get('exp_name', 'DifIISR_train')
    output_dir = Path(configs.train.output_dir) / f'{exp_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # 设置日志
    logger = setup_logger(output_dir)
    logger.info(f'Config: {args.config}')
    logger.info(f'Output dir: {output_dir}')

    # 保存配置
    import shutil
    shutil.copy(args.config, output_dir / 'config.yaml')

    # 构建模型
    logger.info('Building models...')
    model, diffusion, autoencoder = build_model(configs, device)

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {num_params / 1e6:.2f}M')

    # 检查autoencoder是否有可学习参数
    ae_trainable_params = []
    if autoencoder is not None:
        ae_trainable_params = [p for p in autoencoder.parameters() if p.requires_grad]
        if ae_trainable_params:
            num_ae_params = sum(p.numel() for p in ae_trainable_params)
            logger.info(f'AutoEncoder trainable parameters: {num_ae_params}')

    # 构建EMA (只对UNet)
    ema = None
    if configs.train.get('use_ema', True):
        ema = LitEma(model, decay=float(configs.train.get('ema_rate', 0.999)))
        logger.info('Using EMA')

    # 构建优化器 - 包含UNet和autoencoder的可学习参数
    params_to_optimize = list(model.parameters())
    if ae_trainable_params:
        params_to_optimize.extend(ae_trainable_params)
        logger.info('Including AutoEncoder parameters in optimizer')

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=float(configs.train.lr),
        weight_decay=float(configs.train.get('weight_decay', 0.01)),
    )

    # 构建学习率调度器
    scheduler = None
    if configs.train.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(configs.train.iterations),
            eta_min=float(configs.train.get('min_lr', 1e-6)),
        )

    # 构建数据加载器
    logger.info('Building dataloaders...')
    train_loader, val_loader = build_dataloader(configs)
    logger.info(f'Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}')

    # TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')

    # 恢复训练
    start_iter = 0
    if args.resume:
        logger.info(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'ema' in ckpt and ema is not None:
            ema.load_state_dict(ckpt['ema'])
        # 恢复autoencoder的可学习参数
        if 'autoencoder' in ckpt and autoencoder is not None:
            autoencoder.load_state_dict(ckpt['autoencoder'])
            logger.info('Restored AutoEncoder learnable parameters')
        start_iter = ckpt['iteration']

    # 构建LPIPS损失函数
    lpips_fn = None
    if configs.get('perceptual_loss', {}).get('enabled', False):
        if HAS_LPIPS:
            lpips_net = configs.perceptual_loss.get('lpips_net', 'vgg')
            lpips_fn = lpips.LPIPS(net=lpips_net).to(device)
            lpips_fn.eval()
            for param in lpips_fn.parameters():
                param.requires_grad = False
            logger.info(f'Using LPIPS perceptual loss with {lpips_net} backbone')
        else:
            logger.warning('LPIPS not installed, perceptual loss disabled')

    # 构建边缘损失函数
    edge_loss_fn = None
    if configs.get('perceptual_loss', {}).get('edge_loss_enabled', False):
        edge_loss_fn = SobelEdgeLoss().to(device)
        edge_weight = configs.perceptual_loss.get('edge_loss_weight', 0.1)
        logger.info(f'Using Sobel edge loss with weight {edge_weight}')

    # 构建频率损失函数
    freq_loss_fn = None
    if configs.get('perceptual_loss', {}).get('freq_loss_enabled', False):
        freq_loss_fn = FFTFrequencyLoss().to(device)
        freq_weight = configs.perceptual_loss.get('freq_loss_weight', 0.1)
        logger.info(f'Using FFT frequency loss with weight {freq_weight}')

    # 训练循环
    logger.info('Starting training...')
    iteration = start_iter
    train_iter = iter(train_loader)

    # 记录loss历史用于绘图
    loss_history = {
        'iteration': [],
        'loss': [],
        'mse': [],
        'lpips': [],
        'edge': [],
        'freq': [],
        'lr': []
    }

    pbar = tqdm(total=int(configs.train.iterations) - start_iter, desc='Training')

    while iteration < int(configs.train.iterations):
        # 获取数据
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # 训练一步
        train_metrics = train_one_step(
            model, diffusion, autoencoder, batch, optimizer, device, configs,
            lpips_fn=lpips_fn, edge_loss_fn=edge_loss_fn, freq_loss_fn=freq_loss_fn,
            iteration=iteration, total_iterations=int(configs.train.iterations)
        )

        # 更新EMA
        if ema is not None:
            ema(model)

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        iteration += 1
        pbar.update(1)

        # 记录日志
        if iteration % int(configs.train.get('log_freq', 100)) == 0:
            lr = optimizer.param_groups[0]['lr']
            log_msg = f'Iter {iteration}: loss={train_metrics["loss"]:.4f}, mse={train_metrics["mse"]:.4f}'
            if 'lpips' in train_metrics and train_metrics['lpips'] > 0:
                log_msg += f', lpips={train_metrics["lpips"]:.4f}'
            if 'edge' in train_metrics and train_metrics['edge'] > 0:
                log_msg += f', edge={train_metrics["edge"]:.4f}'
            if 'freq' in train_metrics and train_metrics['freq'] > 0:
                log_msg += f', freq={train_metrics["freq"]:.4f}'
            if 'lpips_weight' in train_metrics:
                log_msg += f', lpips_w={train_metrics["lpips_weight"]:.4f}'
            log_msg += f', lr={lr:.2e}'
            logger.info(log_msg)

            writer.add_scalar('train/loss', train_metrics['loss'], iteration)
            writer.add_scalar('train/mse', train_metrics['mse'], iteration)
            writer.add_scalar('train/lr', lr, iteration)
            writer.add_scalar('train/grad_norm', train_metrics['grad_norm'], iteration)
            if 'lpips' in train_metrics:
                writer.add_scalar('train/lpips', train_metrics['lpips'], iteration)
            if 'edge' in train_metrics:
                writer.add_scalar('train/edge', train_metrics['edge'], iteration)
            if 'freq' in train_metrics:
                writer.add_scalar('train/freq', train_metrics['freq'], iteration)
            if 'lpips_weight' in train_metrics:
                writer.add_scalar('train/lpips_weight', train_metrics['lpips_weight'], iteration)

            # 记录到loss_history用于绘图
            loss_history['iteration'].append(iteration)
            loss_history['loss'].append(train_metrics['loss'])
            loss_history['mse'].append(train_metrics['mse'])
            loss_history['lpips'].append(train_metrics.get('lpips', 0))
            loss_history['edge'].append(train_metrics.get('edge', 0))
            loss_history['freq'].append(train_metrics.get('freq', 0))
            loss_history['lr'].append(lr)

        # 验证
        if iteration % int(configs.train.get('val_freq', 5000)) == 0:
            logger.info('Validating...')
            # 使用EMA模型验证
            if ema is not None:
                ema.store(list(model.parameters()))
                ema.copy_to(model)

            val_metrics = validate(model, diffusion, autoencoder, val_loader, device, configs, lpips_fn=lpips_fn)

            if ema is not None:
                ema.restore(list(model.parameters()))

            log_msg = f'Validation PSNR: {val_metrics["psnr"]:.2f} dB, SSIM: {val_metrics["ssim"]:.4f}'
            if val_metrics.get('lpips', 0) > 0:
                log_msg += f', LPIPS: {val_metrics["lpips"]:.4f}'
            logger.info(log_msg)

            writer.add_scalar('val/psnr', val_metrics['psnr'], iteration)
            writer.add_scalar('val/ssim', val_metrics['ssim'], iteration)
            if val_metrics.get('lpips', 0) > 0:
                writer.add_scalar('val/lpips', val_metrics['lpips'], iteration)

        # 保存检查点
        if iteration % int(configs.train.get('save_freq', 10000)) == 0:
            save_path = ckpt_dir / f'model_{iteration:06d}.pth'
            save_checkpoint(model, ema, optimizer, scheduler, iteration, save_path, autoencoder)
            logger.info(f'Saved checkpoint to {save_path}')

            # 保存最新的检查点
            save_checkpoint(model, ema, optimizer, scheduler, iteration, ckpt_dir / 'latest.pth', autoencoder)

    pbar.close()

    # 保存最终模型
    save_path = ckpt_dir / 'final.pth'
    save_checkpoint(model, ema, optimizer, scheduler, iteration, save_path, autoencoder)
    logger.info(f'Training finished. Final model saved to {save_path}')

    writer.close()

    # 绘制训练曲线
    plot_training_curves(loss_history, output_dir, configs)


def plot_training_curves(loss_history, output_dir, configs):
    """绘制训练曲线并保存"""
    import matplotlib.pyplot as plt

    if not loss_history['iteration']:
        return

    iterations = loss_history['iteration']

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Curves - {configs.get("exp_name", "DifIISR")}', fontsize=14)

    # 1. Total Loss
    ax1 = axes[0, 0]
    ax1.plot(iterations, loss_history['loss'], 'b-', alpha=0.7, linewidth=0.8)
    # 添加平滑曲线
    if len(iterations) > 10:
        window = min(50, len(iterations) // 10)
        smoothed = np.convolve(loss_history['loss'], np.ones(window)/window, mode='valid')
        ax1.plot(iterations[window-1:], smoothed, 'r-', linewidth=2, label='Smoothed')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. MSE Loss
    ax2 = axes[0, 1]
    ax2.plot(iterations, loss_history['mse'], 'g-', alpha=0.7, linewidth=0.8)
    if len(iterations) > 10:
        window = min(50, len(iterations) // 10)
        smoothed = np.convolve(loss_history['mse'], np.ones(window)/window, mode='valid')
        ax2.plot(iterations[window-1:], smoothed, 'r-', linewidth=2, label='Smoothed')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('MSE Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. LPIPS Loss
    ax3 = axes[1, 0]
    if loss_history['lpips'] and any(v > 0 for v in loss_history['lpips']):
        ax3.plot(iterations, loss_history['lpips'], 'm-', alpha=0.7, linewidth=0.8)
        if len(iterations) > 10:
            window = min(50, len(iterations) // 10)
            smoothed = np.convolve(loss_history['lpips'], np.ones(window)/window, mode='valid')
            ax3.plot(iterations[window-1:], smoothed, 'r-', linewidth=2, label='Smoothed')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('LPIPS Loss')
        ax3.set_title('LPIPS Loss (Perceptual)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'LPIPS not enabled', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('LPIPS Loss (Not Enabled)')

    # 4. Learning Rate
    ax4 = axes[1, 1]
    ax4.plot(iterations, loss_history['lr'], 'c-', linewidth=1.5)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    plt.tight_layout()

    # 保存图表
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f'Training curves saved to {plot_path}')

    # 额外保存loss数据到csv
    csv_path = output_dir / 'loss_history.csv'
    with open(csv_path, 'w') as f:
        f.write('iteration,loss,mse,lpips,lr\n')
        for i in range(len(iterations)):
            lpips_val = loss_history['lpips'][i] if loss_history['lpips'] else 0
            f.write(f"{iterations[i]},{loss_history['loss'][i]:.6f},{loss_history['mse'][i]:.6f},{lpips_val:.6f},{loss_history['lr'][i]:.2e}\n")
    logging.info(f'Loss history saved to {csv_path}')


if __name__ == '__main__':
    main()
