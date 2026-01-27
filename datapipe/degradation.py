"""
Real-ESRGAN风格的图像退化处理
参考DifIISR的退化配置实现

退化流程:
1. 第一阶段退化
   - 模糊 (多种核: iso, aniso, generalized, plateau)
   - 随机缩放
   - 添加噪声 (高斯/泊松)
   - JPEG压缩
2. 第二阶段退化 (50%概率)
   - 重复类似操作
3. 最终下采样
"""

import cv2
import math
import random
import numpy as np
from PIL import Image
from io import BytesIO
from scipy import special
from scipy.ndimage import filters as ndimage_filters


# ==================== 模糊核生成 ====================

def sigma_matrix2(sig_x, sig_y, theta):
    """计算2D高斯的协方差矩阵"""
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))


def mesh_grid(kernel_size):
    """生成网格坐标"""
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)),
                    yy.reshape(kernel_size * kernel_size, 1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix, grid):
    """计算2D高斯概率密度"""
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    """生成2D高斯核"""
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_generalized_Gaussian(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    """生成广义高斯核"""
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    """生成平顶核"""
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = 1 / (np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    kernel = kernel / np.sum(kernel)
    return kernel


def random_bivariate_Gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=None, isotropic=True):
    """随机生成2D高斯核"""
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic:
        sigma_y = sigma_x
    else:
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])

    kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

    # 添加乘性噪声
    if noise_range is not None:
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
        kernel = kernel / np.sum(kernel)

    return kernel


def random_bivariate_generalized_Gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range,
                                          beta_range, noise_range=None, isotropic=True):
    """随机生成广义高斯核"""
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic:
        sigma_y = sigma_x
    else:
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    beta = np.random.uniform(beta_range[0], beta_range[1])

    kernel = bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

    if noise_range is not None:
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
        kernel = kernel / np.sum(kernel)

    return kernel


def random_bivariate_plateau(kernel_size, sigma_x_range, sigma_y_range, rotation_range,
                             beta_range, noise_range=None, isotropic=True):
    """随机生成平顶核"""
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic:
        sigma_y = sigma_x
    else:
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    beta = np.random.uniform(beta_range[0], beta_range[1])

    kernel = bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

    if noise_range is not None:
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
        kernel = kernel / np.sum(kernel)

    return kernel


def random_mixed_kernels(kernel_list, kernel_prob, kernel_size, sigma_x_range, sigma_y_range,
                         rotation_range, betag_range, betap_range, noise_range=None):
    """随机选择并生成模糊核"""
    kernel_type = random.choices(kernel_list, kernel_prob)[0]

    if kernel_type == 'iso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range,
            noise_range=noise_range, isotropic=True)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range,
            noise_range=noise_range, isotropic=False)
    elif kernel_type == 'generalized_iso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betag_range,
            noise_range=noise_range, isotropic=True)
    elif kernel_type == 'generalized_aniso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betag_range,
            noise_range=noise_range, isotropic=False)
    elif kernel_type == 'plateau_iso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range,
            noise_range=noise_range, isotropic=True)
    elif kernel_type == 'plateau_aniso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range,
            noise_range=noise_range, isotropic=False)
    else:
        raise ValueError(f'Unknown kernel type: {kernel_type}')

    return kernel


# ==================== 噪声添加 ====================

def add_gaussian_noise(img, sigma_range, gray_prob=0):
    """添加高斯噪声"""
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    if np.random.uniform() < gray_prob:
        # 灰度噪声
        noise = np.random.randn(img.shape[0], img.shape[1], 1) * sigma / 255.0
        noise = np.repeat(noise, 3, axis=2)
    else:
        # 彩色噪声
        noise = np.random.randn(*img.shape) * sigma / 255.0

    img_noisy = img + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    return img_noisy


def add_poisson_noise(img, scale_range, gray_prob=0):
    """添加泊松噪声"""
    scale = np.random.uniform(scale_range[0], scale_range[1])

    # 确保输入在有效范围内
    img = np.clip(img, 0, 1)

    if np.random.uniform() < gray_prob:
        # 灰度噪声
        gray = np.mean(img, axis=2, keepdims=True)
        # 确保lambda参数非负
        lam = np.maximum(gray * 255 * scale, 0)
        noise = np.random.poisson(lam) / scale / 255.0 - gray
        noise = np.repeat(noise, 3, axis=2)
    else:
        # 彩色噪声
        # 确保lambda参数非负
        lam = np.maximum(img * 255 * scale, 0)
        noise = np.random.poisson(lam) / scale / 255.0 - img

    img_noisy = img + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    return img_noisy


# ==================== JPEG压缩 ====================

def add_jpeg_compression(img, quality_range):
    """添加JPEG压缩伪影"""
    quality = random.randint(quality_range[0], quality_range[1])

    # 转换为PIL Image
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)

    # JPEG压缩
    buffer = BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img_compressed = Image.open(buffer)

    # 转回numpy
    img_out = np.array(img_compressed).astype(np.float32) / 255.0
    return img_out


# ==================== 缩放操作 ====================

def random_resize(img, resize_prob, resize_range, target_size=None):
    """随机缩放"""
    h, w = img.shape[:2]

    # 选择缩放类型: up, down, keep
    resize_type = random.choices(['up', 'down', 'keep'], resize_prob)[0]

    if resize_type == 'up':
        scale = np.random.uniform(1.0, resize_range[1])
    elif resize_type == 'down':
        scale = np.random.uniform(resize_range[0], 1.0)
    else:
        scale = 1.0

    if scale != 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        # 随机选择插值方法
        interp = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # 如果指定了目标尺寸，缩放回去
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    return img


# ==================== 主退化类 ====================

class RealESRGANDegradation:
    """
    Real-ESRGAN风格的二阶退化处理

    参考DifIISR配置实现
    """

    def __init__(self, config=None):
        """
        初始化退化参数

        Args:
            config: 退化配置字典，如果为None则使用默认配置
        """
        if config is None:
            config = self.get_default_config()

        self.config = config
        self.sf = config.get('sf', 4)

        # 第一阶段参数
        self.blur_kernel_size = config.get('blur_kernel_size', 21)
        self.kernel_list = config.get('kernel_list', ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'])
        self.kernel_prob = config.get('kernel_prob', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.blur_sigma = config.get('blur_sigma', [0.2, 3.0])
        self.betag_range = config.get('betag_range', [0.5, 4.0])
        self.betap_range = config.get('betap_range', [1, 2.0])

        self.resize_prob = config.get('resize_prob', [0.2, 0.7, 0.1])
        self.resize_range = config.get('resize_range', [0.15, 1.5])

        self.gaussian_noise_prob = config.get('gaussian_noise_prob', 0.5)
        self.noise_range = config.get('noise_range', [1, 30])
        self.poisson_scale_range = config.get('poisson_scale_range', [0.05, 3.0])
        self.gray_noise_prob = config.get('gray_noise_prob', 0.4)

        self.jpeg_range = config.get('jpeg_range', [30, 95])

        # 第二阶段参数
        self.second_order_prob = config.get('second_order_prob', 0.5)
        self.second_blur_prob = config.get('second_blur_prob', 0.8)

        self.blur_kernel_size2 = config.get('blur_kernel_size2', 15)
        self.kernel_list2 = config.get('kernel_list2', ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'])
        self.kernel_prob2 = config.get('kernel_prob2', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.blur_sigma2 = config.get('blur_sigma2', [0.2, 1.5])
        self.betag_range2 = config.get('betag_range2', [0.5, 4.0])
        self.betap_range2 = config.get('betap_range2', [1, 2.0])

        self.resize_prob2 = config.get('resize_prob2', [0.3, 0.4, 0.3])
        self.resize_range2 = config.get('resize_range2', [0.3, 1.2])

        self.gaussian_noise_prob2 = config.get('gaussian_noise_prob2', 0.5)
        self.noise_range2 = config.get('noise_range2', [1, 25])
        self.poisson_scale_range2 = config.get('poisson_scale_range2', [0.05, 2.5])
        self.gray_noise_prob2 = config.get('gray_noise_prob2', 0.4)

        self.jpeg_range2 = config.get('jpeg_range2', [30, 95])

    @staticmethod
    def get_default_config():
        """获取默认配置（与DifIISR一致）"""
        return {
            'sf': 4,

            # 第一阶段
            'blur_kernel_size': 21,
            'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
            'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            'blur_sigma': [0.2, 3.0],
            'betag_range': [0.5, 4.0],
            'betap_range': [1, 2.0],

            'resize_prob': [0.2, 0.7, 0.1],
            'resize_range': [0.15, 1.5],

            'gaussian_noise_prob': 0.5,
            'noise_range': [1, 30],
            'poisson_scale_range': [0.05, 3.0],
            'gray_noise_prob': 0.4,

            'jpeg_range': [30, 95],

            # 第二阶段
            'second_order_prob': 0.5,
            'second_blur_prob': 0.8,

            'blur_kernel_size2': 15,
            'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
            'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            'blur_sigma2': [0.2, 1.5],
            'betag_range2': [0.5, 4.0],
            'betap_range2': [1, 2.0],

            'resize_prob2': [0.3, 0.4, 0.3],
            'resize_range2': [0.3, 1.2],

            'gaussian_noise_prob2': 0.5,
            'noise_range2': [1, 25],
            'poisson_scale_range2': [0.05, 2.5],
            'gray_noise_prob2': 0.4,

            'jpeg_range2': [30, 95],
        }

    def apply_blur(self, img, kernel_size, kernel_list, kernel_prob, sigma_range, betag_range, betap_range):
        """应用模糊"""
        kernel = random_mixed_kernels(
            kernel_list, kernel_prob, kernel_size,
            sigma_x_range=sigma_range, sigma_y_range=sigma_range,
            rotation_range=[-math.pi, math.pi],
            betag_range=betag_range, betap_range=betap_range,
            noise_range=None
        )

        # 应用卷积
        img_blur = cv2.filter2D(img, -1, kernel)
        return img_blur

    def apply_noise(self, img, gaussian_prob, noise_range, poisson_range, gray_prob):
        """应用噪声"""
        if np.random.uniform() < gaussian_prob:
            img = add_gaussian_noise(img, noise_range, gray_prob)
        else:
            img = add_poisson_noise(img, poisson_range, gray_prob)
        return img

    def first_degradation(self, img):
        """第一阶段退化"""
        h, w = img.shape[:2]

        # 1. 模糊
        img = self.apply_blur(
            img, self.blur_kernel_size, self.kernel_list, self.kernel_prob,
            self.blur_sigma, self.betag_range, self.betap_range
        )

        # 2. 随机缩放
        img = random_resize(img, self.resize_prob, self.resize_range, target_size=(h, w))

        # 3. 添加噪声
        img = self.apply_noise(
            img, self.gaussian_noise_prob, self.noise_range,
            self.poisson_scale_range, self.gray_noise_prob
        )

        # 4. JPEG压缩
        img = add_jpeg_compression(img, self.jpeg_range)

        return img

    def second_degradation(self, img):
        """第二阶段退化"""
        h, w = img.shape[:2]

        # 1. 模糊 (有概率跳过)
        if np.random.uniform() < self.second_blur_prob:
            img = self.apply_blur(
                img, self.blur_kernel_size2, self.kernel_list2, self.kernel_prob2,
                self.blur_sigma2, self.betag_range2, self.betap_range2
            )

        # 2. 随机缩放
        img = random_resize(img, self.resize_prob2, self.resize_range2, target_size=(h, w))

        # 3. 添加噪声
        img = self.apply_noise(
            img, self.gaussian_noise_prob2, self.noise_range2,
            self.poisson_scale_range2, self.gray_noise_prob2
        )

        # 4. JPEG压缩
        img = add_jpeg_compression(img, self.jpeg_range2)

        return img

    def final_downsample(self, img, target_size=None):
        """最终下采样"""
        h, w = img.shape[:2]

        if target_size is None:
            new_h, new_w = h // self.sf, w // self.sf
        else:
            new_h, new_w = target_size

        # 使用bicubic下采样
        img_lr = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img_lr

    def __call__(self, img_hr):
        """
        对HR图像应用完整退化流程

        Args:
            img_hr: numpy array, [H, W, 3], range [0, 1], RGB

        Returns:
            img_lr: numpy array, [H/sf, W/sf, 3], range [0, 1], RGB
        """
        img = img_hr.copy()

        # 第一阶段退化
        img = self.first_degradation(img)

        # 第二阶段退化 (50%概率)
        if np.random.uniform() < self.second_order_prob:
            img = self.second_degradation(img)

        # 最终下采样
        img_lr = self.final_downsample(img)

        # 确保值在[0, 1]范围内
        img_lr = np.clip(img_lr, 0, 1)

        return img_lr


# ==================== 测试代码 ====================

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 创建测试图像
    print("Testing RealESRGAN Degradation...")

    # 读取测试图像或创建随机图像
    try:
        img_hr = np.array(Image.open('/home/lch/sr_recons/dataset/M3FD_processed/train/HR/000000.png').convert('RGB')).astype(np.float32) / 255.0
    except:
        img_hr = np.random.rand(256, 256, 3).astype(np.float32)

    print(f"HR image shape: {img_hr.shape}")

    # 创建退化器
    degradation = RealESRGANDegradation()

    # 应用退化
    img_lr = degradation(img_hr)

    print(f"LR image shape: {img_lr.shape}")
    print(f"LR value range: [{img_lr.min():.3f}, {img_lr.max():.3f}]")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_hr)
    axes[0].set_title(f'HR ({img_hr.shape[0]}x{img_hr.shape[1]})')
    axes[0].axis('off')

    axes[1].imshow(img_lr)
    axes[1].set_title(f'LR ({img_lr.shape[0]}x{img_lr.shape[1]})')
    axes[1].axis('off')

    # Bicubic上采样对比
    img_lr_up = cv2.resize(img_lr, (img_hr.shape[1], img_hr.shape[0]), interpolation=cv2.INTER_CUBIC)
    axes[2].imshow(img_lr_up)
    axes[2].set_title('LR Bicubic Upsampled')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('/home/lch/sr_recons/DifIISR/work_start/degradation_test.png', dpi=150)
    print("Test image saved to work_start/degradation_test.png")
