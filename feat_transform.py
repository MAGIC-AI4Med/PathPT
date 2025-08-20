import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torchvision import transforms


class FeatureTransform:
    def __init__(self, 
                 noise_level=0.05, 
                 mixup_prob=0.3,
                 rotation_prob=0.3, 
                 p=0.5,
                 smoothing = 0.1,
                 num_classes = 3,
                 verify_input=True
                 ):
        """
        L2归一化特征的数据增强Transform
        
        Args:
            noise_level: 高斯噪声的标准差
            dropout_rate: 随机特征丢弃的比例
            mixup_prob: 使用mixup增强的概率
            rotation_prob: 使用随机旋转的概率
            p: 应用任何增强的总体概率
            verify_input: 是否验证输入特征是L2归一化的
        """
        self.noise_level = noise_level
        self.mixup_prob = mixup_prob
        self.rotation_prob = rotation_prob
        self.p = p
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.verify_input = verify_input
    
    def _verify_l2_norm(self, x):
        """验证输入是否已L2归一化"""
        if isinstance(x, np.ndarray):
            norm = np.linalg.norm(x, axis=-1)
            return np.allclose(norm, 1.0, rtol=1e-5, atol=1e-5)
        elif isinstance(x, torch.Tensor):
            norm = torch.norm(x, p=2, dim=-1)
            return torch.allclose(norm, torch.ones_like(norm), rtol=1e-5, atol=1e-5)
        return False
        
    def _l2_normalize(self, x):
        """对输入进行L2归一化"""
        if isinstance(x, np.ndarray):
            return normalize(x, norm='l2', axis=-1)
        elif isinstance(x, torch.Tensor):
            return F.normalize(x, p=2, dim=-1)
        return x
    
    def add_noise(self, x):
        """添加高斯噪声"""
        if isinstance(x, np.ndarray):
            noise = np.random.normal(0, self.noise_level, x.shape)
            return self._l2_normalize(x + noise)
        elif isinstance(x, torch.Tensor):
            noise = torch.randn_like(x) * self.noise_level
            return F.normalize(x + noise, p=2, dim=-1)
        return x
    
    def random_rotation_blocks(self, x, max_angle=np.pi/4):
        """
        使用对角块旋转矩阵应用有限角度的随机旋转
        
        参数:
            x: 输入特征
            max_angle: 最大旋转角度（弧度），默认为π/4（45度）
        """
        if isinstance(x, np.ndarray):
            dim = x.shape[-1]
            rotation = np.eye(dim)
            
            # 创建一系列2×2旋转块
            for i in range(0, dim-1, 2):
                # 确保我们有足够的维度来旋转
                if i+1 >= dim:
                    break
                    
                # 生成有限范围内的随机角度
                # 可以是[-max_angle, max_angle]或[0, max_angle]
                theta = np.random.uniform(-max_angle, max_angle)
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                
                # 创建2×2旋转矩阵
                rot_block = np.array([
                    [cos_t, -sin_t],
                    [sin_t, cos_t]
                ])
                
                # 将旋转矩阵填入对角块
                rotation[i:i+2, i:i+2] = rot_block
            
            # 应用旋转
            return np.dot(x, rotation)
        
        elif isinstance(x, torch.Tensor):
            dim = x.shape[-1]
            rotation = torch.eye(dim, device=x.device)
            
            # 创建一系列2×2旋转块
            for i in range(0, dim-1, 2):
                # 确保我们有足够的维度来旋转
                if i+1 >= dim:
                    break
                    
                # 生成有限范围内的随机角度
                theta = torch.rand(1, device=x.device) * 2 * max_angle - max_angle
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                
                # 创建2×2旋转矩阵
                rot_block = torch.tensor([
                    [cos_t, -sin_t],
                    [sin_t, cos_t]
                ], device=x.device).squeeze()
                
                # 将旋转矩阵填入对角块
                rotation[i:i+2, i:i+2] = rot_block
            
            # 应用旋转
            return torch.matmul(x, rotation)
        
        return x
    
    def feature_mixup(self, feature1, feature2, label1, label2, alpha=0.2, l2_normalize=True):
        """
        对两个特征和标签执行mixup
        
        Args:
            feature1: 第一个特征
            feature2: 第二个特征
            label1: 第一个标签，可选
            label2: 第二个标签，可选
            alpha: mixup参数，用于beta分布
            l2_normalize: 是否对混合后的特征执行L2归一化
            
        Returns:
            混合后的特征和标签（如果提供了标签）
        """
        # 生成mixup系数
        lam = np.random.beta(alpha, alpha)
        
        # 混合特征
        if isinstance(feature1, np.ndarray):
            mixed_feature = lam * feature1 + (1 - lam) * feature2
            # L2归一化
            if l2_normalize:
                mixed_feature = normalize(mixed_feature, norm='l2', axis=-1)
        elif isinstance(feature1, torch.Tensor):
            lam_tensor = torch.tensor(lam, device=feature1.device).float()
            mixed_feature = lam_tensor * feature1 + (1 - lam_tensor) * feature2
            # L2归一化
            if l2_normalize:
                mixed_feature = F.normalize(mixed_feature, p=2, dim=-1)
        
        # mixed_label = lam * label1 + (1 - lam) * label2
        
        label1_one_hot = F.one_hot(label1, self.num_classes).float()
        label2_one_hot = F.one_hot(label2, self.num_classes).float()
        mixed_onehot_label = lam * label1_one_hot + (1 - lam) * label2_one_hot
        smoothed_mixed_label = mixed_onehot_label * (1 - self.smoothing) + self.smoothing / self.num_classes
        
        return mixed_feature, smoothed_mixed_label
    
    def __call__(self, x, x2, label, label2):
        """
        应用特征增强变换
        
        Args:
            x: 输入特征，预期已经L2归一化，可以是numpy数组或PyTorch张量
            
        Returns:
            增强后的特征，保持L2归一化
        """
        # 验证输入是否已L2归一化  
        label = torch.tensor(label)
        label2 = torch.tensor(label2)
        
        if self.verify_input and not self._verify_l2_norm(x):
            x = self._l2_normalize(x)
        
        one_hot = F.one_hot(label, self.num_classes).float()
        # 应用平滑
        smoothed_label = one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        
        # 以概率p决定是否应用任何增强
        if np.random.random() > self.p:
            return x, smoothed_label
            
        # 决定应用哪种增强方法
        r = np.random.random()
        
        if r < 0.5:
            # 添加噪声
            return self.add_noise(x), smoothed_label
        # elif r < 0.66 and self.rotation_prob > 0:
        #     # 随机旋转
        #     return self.random_rotation_blocks(x), smoothed_label
        elif self.mixup_prob > 0:
            # Mixup (注意：这需要一对特征才能工作)
            return self.feature_mixup(x, x2, label, label2)
        
        return x, smoothed_label