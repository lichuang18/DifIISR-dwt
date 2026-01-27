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

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from utils import util_net
from utils import util_image
from utils import util_common
from datapipe.paired_dataset import PairedImageDataset, ValPairedImageDataset
from datapipe.degradation_dataset import DegradationDatasetFromPatches, ValDegradationDataset
from ldm.modules.ema import LitEma


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

        # 冻结autoencoder
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


def train_one_step(model, diffusion, autoencoder, batch, optimizer, device, configs):
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

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 梯度裁剪
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return {
        'loss': loss.item(),
        'mse': losses['mse'].mean().item(),
        'grad_norm': grad_norm.item(),
    }


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
def validate(model, diffusion, autoencoder, val_loader, device, configs):
    """验证"""
    model.eval()

    total_psnr = 0
    total_ssim = 0
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

    return {'psnr': avg_psnr, 'ssim': avg_ssim}


def save_checkpoint(model, ema, optimizer, scheduler, iteration, save_path):
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

    # 构建EMA
    ema = None
    if configs.train.get('use_ema', True):
        ema = LitEma(model, decay=float(configs.train.get('ema_rate', 0.999)))
        logger.info('Using EMA')

    # 构建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
        start_iter = ckpt['iteration']

    # 训练循环
    logger.info('Starting training...')
    iteration = start_iter
    train_iter = iter(train_loader)

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
            model, diffusion, autoencoder, batch, optimizer, device, configs
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
            logger.info(
                f'Iter {iteration}: loss={train_metrics["loss"]:.4f}, '
                f'mse={train_metrics["mse"]:.4f}, lr={lr:.2e}'
            )
            writer.add_scalar('train/loss', train_metrics['loss'], iteration)
            writer.add_scalar('train/mse', train_metrics['mse'], iteration)
            writer.add_scalar('train/lr', lr, iteration)
            writer.add_scalar('train/grad_norm', train_metrics['grad_norm'], iteration)

        # 验证
        if iteration % int(configs.train.get('val_freq', 5000)) == 0:
            logger.info('Validating...')
            # 使用EMA模型验证
            if ema is not None:
                ema.store(list(model.parameters()))
                ema.copy_to(model)

            val_metrics = validate(model, diffusion, autoencoder, val_loader, device, configs)

            if ema is not None:
                ema.restore(list(model.parameters()))

            logger.info(f'Validation PSNR: {val_metrics["psnr"]:.2f} dB, SSIM: {val_metrics["ssim"]:.4f}')
            writer.add_scalar('val/psnr', val_metrics['psnr'], iteration)
            writer.add_scalar('val/ssim', val_metrics['ssim'], iteration)

        # 保存检查点
        if iteration % int(configs.train.get('save_freq', 10000)) == 0:
            save_path = ckpt_dir / f'model_{iteration:06d}.pth'
            save_checkpoint(model, ema, optimizer, scheduler, iteration, save_path)
            logger.info(f'Saved checkpoint to {save_path}')

            # 保存最新的检查点
            save_checkpoint(model, ema, optimizer, scheduler, iteration, ckpt_dir / 'latest.pth')

    pbar.close()

    # 保存最终模型
    save_path = ckpt_dir / 'final.pth'
    save_checkpoint(model, ema, optimizer, scheduler, iteration, save_path)
    logger.info(f'Training finished. Final model saved to {save_path}')

    writer.close()


if __name__ == '__main__':
    main()
