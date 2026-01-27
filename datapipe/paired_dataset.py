"""
配对数据集类，用于加载HR-LR配对的训练数据
"""

import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class PairedImageDataset(Dataset):
    """
    加载HR-LR配对图像的数据集

    目录结构:
        root/
        ├── HR/
        │   ├── 000000.png
        │   └── ...
        └── LR/
            ├── 000000.png
            └── ...
    """
    def __init__(
            self,
            hr_dir,
            lr_dir,
            gt_size=256,
            scale=4,
            use_hflip=True,
            use_rot=True,
            mean=0.5,
            std=0.5,
            length=None,
            ):
        super().__init__()

        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.gt_size = gt_size
        self.scale = scale
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.mean = mean
        self.std = std

        # 获取所有HR图像路径
        self.hr_paths = sorted(list(self.hr_dir.glob('*.png')) + list(self.hr_dir.glob('*.jpg')))

        if length is not None and length < len(self.hr_paths):
            self.hr_paths = random.sample(self.hr_paths, length)

        # 归一化变换
        self.normalize = transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, index):
        hr_path = self.hr_paths[index]
        lr_path = self.lr_dir / hr_path.name

        # 读取图像
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')

        # 转换为numpy数组
        hr_img = np.array(hr_img).astype(np.float32) / 255.0
        lr_img = np.array(lr_img).astype(np.float32) / 255.0

        # 数据增强
        if self.use_hflip and random.random() > 0.5:
            hr_img = np.flip(hr_img, axis=1).copy()
            lr_img = np.flip(lr_img, axis=1).copy()

        if self.use_rot:
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                hr_img = np.rot90(hr_img, rot_k).copy()
                lr_img = np.rot90(lr_img, rot_k).copy()

        # 转换为tensor [C, H, W]
        hr_tensor = torch.from_numpy(hr_img.transpose(2, 0, 1))
        lr_tensor = torch.from_numpy(lr_img.transpose(2, 0, 1))

        # 归一化到[-1, 1]
        hr_tensor = self.normalize(hr_tensor)
        lr_tensor = self.normalize(lr_tensor)

        return {
            'gt': hr_tensor,      # HR图像 (ground truth)
            'lq': lr_tensor,      # LR图像 (low quality)
            'path': str(hr_path)
        }


class ValPairedImageDataset(Dataset):
    """
    验证集数据集，不做数据增强
    """
    def __init__(
            self,
            hr_dir,
            lr_dir,
            mean=0.5,
            std=0.5,
            length=None,
            ):
        super().__init__()

        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.mean = mean
        self.std = std

        # 获取所有HR图像路径
        self.hr_paths = sorted(list(self.hr_dir.glob('*.png')) + list(self.hr_dir.glob('*.jpg')))

        if length is not None and length < len(self.hr_paths):
            self.hr_paths = self.hr_paths[:length]

        # 归一化变换
        self.normalize = transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, index):
        hr_path = self.hr_paths[index]
        lr_path = self.lr_dir / hr_path.name

        # 读取图像
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')

        # 转换为numpy数组
        hr_img = np.array(hr_img).astype(np.float32) / 255.0
        lr_img = np.array(lr_img).astype(np.float32) / 255.0

        # 转换为tensor [C, H, W]
        hr_tensor = torch.from_numpy(hr_img.transpose(2, 0, 1))
        lr_tensor = torch.from_numpy(lr_img.transpose(2, 0, 1))

        # 归一化到[-1, 1]
        hr_tensor = self.normalize(hr_tensor)
        lr_tensor = self.normalize(lr_tensor)

        return {
            'gt': hr_tensor,
            'lq': lr_tensor,
            'path': str(hr_path)
        }
