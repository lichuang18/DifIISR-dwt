"""
支持在线退化的数据集类
训练时从HR图像动态生成LR图像，使用Real-ESRGAN风格的退化
"""

import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2

from .degradation import RealESRGANDegradation


class DegradationDataset(Dataset):
    """
    在线退化数据集

    只需要HR图像目录，LR图像在训练时动态生成

    目录结构:
        hr_dir/
        ├── 000000.png
        ├── 000001.png
        └── ...
    """
    def __init__(
            self,
            hr_dir,
            gt_size=256,
            scale=4,
            use_hflip=True,
            use_rot=True,
            mean=0.5,
            std=0.5,
            length=None,
            degradation_config=None,
            crop_pad_size=300,  # 裁剪前先padding到这个尺寸，避免边缘效应
            ):
        super().__init__()

        self.hr_dir = Path(hr_dir)
        self.gt_size = gt_size
        self.scale = scale
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.mean = mean
        self.std = std
        self.crop_pad_size = crop_pad_size

        # 获取所有HR图像路径
        self.hr_paths = sorted(
            list(self.hr_dir.glob('*.png')) +
            list(self.hr_dir.glob('*.jpg')) +
            list(self.hr_dir.glob('*.jpeg'))
        )

        if length is not None and length < len(self.hr_paths):
            self.hr_paths = random.sample(self.hr_paths, length)

        # 初始化退化器
        if degradation_config is None:
            degradation_config = RealESRGANDegradation.get_default_config()
        degradation_config['sf'] = scale
        self.degradation = RealESRGANDegradation(degradation_config)

        # 归一化变换
        self.normalize = transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])

        print(f"[DegradationDataset] Loaded {len(self.hr_paths)} images from {hr_dir}")
        print(f"[DegradationDataset] GT size: {gt_size}, Scale: {scale}")
        print(f"[DegradationDataset] Using Real-ESRGAN degradation")

    def __len__(self):
        return len(self.hr_paths)

    def _random_crop(self, img, crop_size):
        """随机裁剪"""
        h, w = img.shape[:2]
        if h < crop_size or w < crop_size:
            # 如果图像太小，先resize
            scale = max(crop_size / h, crop_size / w) * 1.1
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]

        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        return img[top:top+crop_size, left:left+crop_size]

    def __getitem__(self, index):
        hr_path = self.hr_paths[index]

        # 读取HR图像
        hr_img = np.array(Image.open(hr_path).convert('RGB')).astype(np.float32) / 255.0

        # 随机裁剪到crop_pad_size (比gt_size大一些，给退化留余量)
        hr_img = self._random_crop(hr_img, self.crop_pad_size)

        # 应用退化生成LR图像
        lr_img = self.degradation(hr_img)

        # 将HR裁剪到gt_size (中心裁剪，与LR对应)
        h, w = hr_img.shape[:2]
        top = (h - self.gt_size) // 2
        left = (w - self.gt_size) // 2
        hr_img = hr_img[top:top+self.gt_size, left:left+self.gt_size]

        # LR也需要对应裁剪
        lr_h, lr_w = lr_img.shape[:2]
        lr_gt_size = self.gt_size // self.scale
        lr_top = (lr_h - lr_gt_size) // 2
        lr_left = (lr_w - lr_gt_size) // 2
        lr_img = lr_img[lr_top:lr_top+lr_gt_size, lr_left:lr_left+lr_gt_size]

        # 数据增强 (HR和LR同步)
        if self.use_hflip and random.random() > 0.5:
            hr_img = np.flip(hr_img, axis=1).copy()
            lr_img = np.flip(lr_img, axis=1).copy()

        if self.use_rot:
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                hr_img = np.rot90(hr_img, rot_k).copy()
                lr_img = np.rot90(lr_img, rot_k).copy()

        # 转换为tensor [C, H, W]
        hr_tensor = torch.from_numpy(hr_img.transpose(2, 0, 1).copy())
        lr_tensor = torch.from_numpy(lr_img.transpose(2, 0, 1).copy())

        # 归一化到[-1, 1]
        hr_tensor = self.normalize(hr_tensor)
        lr_tensor = self.normalize(lr_tensor)

        return {
            'gt': hr_tensor,      # HR图像 (ground truth)
            'lq': lr_tensor,      # LR图像 (low quality, 在线退化生成)
            'path': str(hr_path)
        }


class DegradationDatasetFromPatches(Dataset):
    """
    从预裁剪的HR patches加载，在线应用退化

    适用于已经预处理好的patch数据集（如M3FD_processed）
    """
    def __init__(
            self,
            hr_dir,
            gt_size=256,
            scale=4,
            use_hflip=True,
            use_rot=True,
            mean=0.5,
            std=0.5,
            length=None,
            degradation_config=None,
            ):
        super().__init__()

        self.hr_dir = Path(hr_dir)
        self.gt_size = gt_size
        self.scale = scale
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.mean = mean
        self.std = std

        # 获取所有HR图像路径
        self.hr_paths = sorted(
            list(self.hr_dir.glob('*.png')) +
            list(self.hr_dir.glob('*.jpg')) +
            list(self.hr_dir.glob('*.jpeg'))
        )

        if length is not None and length < len(self.hr_paths):
            self.hr_paths = random.sample(self.hr_paths, length)

        # 初始化退化器
        if degradation_config is None:
            degradation_config = RealESRGANDegradation.get_default_config()
        degradation_config['sf'] = scale
        self.degradation = RealESRGANDegradation(degradation_config)

        # 归一化变换
        self.normalize = transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])

        print(f"[DegradationDatasetFromPatches] Loaded {len(self.hr_paths)} patches from {hr_dir}")
        print(f"[DegradationDatasetFromPatches] Scale: {scale}")
        print(f"[DegradationDatasetFromPatches] Using Real-ESRGAN degradation")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, index):
        hr_path = self.hr_paths[index]

        # 读取HR图像 (已经是256x256的patch)
        hr_img = np.array(Image.open(hr_path).convert('RGB')).astype(np.float32) / 255.0

        # 应用退化生成LR图像
        lr_img = self.degradation(hr_img)

        # 数据增强 (HR和LR同步)
        if self.use_hflip and random.random() > 0.5:
            hr_img = np.flip(hr_img, axis=1).copy()
            lr_img = np.flip(lr_img, axis=1).copy()

        if self.use_rot:
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                hr_img = np.rot90(hr_img, rot_k).copy()
                lr_img = np.rot90(lr_img, rot_k).copy()

        # 转换为tensor [C, H, W]
        hr_tensor = torch.from_numpy(hr_img.transpose(2, 0, 1).copy())
        lr_tensor = torch.from_numpy(lr_img.transpose(2, 0, 1).copy())

        # 归一化到[-1, 1]
        hr_tensor = self.normalize(hr_tensor)
        lr_tensor = self.normalize(lr_tensor)

        return {
            'gt': hr_tensor,      # HR图像 (ground truth)
            'lq': lr_tensor,      # LR图像 (low quality, 在线退化生成)
            'path': str(hr_path)
        }


class ValDegradationDataset(Dataset):
    """
    验证集数据集 - 使用固定种子的退化，保证可复现
    """
    def __init__(
            self,
            hr_dir,
            scale=4,
            mean=0.5,
            std=0.5,
            length=None,
            degradation_config=None,
            seed=42,
            ):
        super().__init__()

        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.mean = mean
        self.std = std
        self.seed = seed

        # 获取所有HR图像路径
        self.hr_paths = sorted(
            list(self.hr_dir.glob('*.png')) +
            list(self.hr_dir.glob('*.jpg')) +
            list(self.hr_dir.glob('*.jpeg'))
        )

        if length is not None and length < len(self.hr_paths):
            self.hr_paths = self.hr_paths[:length]

        # 初始化退化器
        if degradation_config is None:
            degradation_config = RealESRGANDegradation.get_default_config()
        degradation_config['sf'] = scale
        self.degradation = RealESRGANDegradation(degradation_config)

        # 归一化变换
        self.normalize = transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])

        print(f"[ValDegradationDataset] Loaded {len(self.hr_paths)} images from {hr_dir}")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, index):
        hr_path = self.hr_paths[index]

        # 设置固定种子，保证同一张图每次退化结果相同
        np.random.seed(self.seed + index)
        random.seed(self.seed + index)

        # 读取HR图像
        hr_img = np.array(Image.open(hr_path).convert('RGB')).astype(np.float32) / 255.0

        # 应用退化生成LR图像
        lr_img = self.degradation(hr_img)

        # 转换为tensor [C, H, W]
        hr_tensor = torch.from_numpy(hr_img.transpose(2, 0, 1).copy())
        lr_tensor = torch.from_numpy(lr_img.transpose(2, 0, 1).copy())

        # 归一化到[-1, 1]
        hr_tensor = self.normalize(hr_tensor)
        lr_tensor = self.normalize(lr_tensor)

        return {
            'gt': hr_tensor,
            'lq': lr_tensor,
            'path': str(hr_path)
        }
