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
        Data augmentation Transform for L2-normalized features
        
        Args:
            noise_level: Standard deviation of Gaussian noise
            dropout_rate: Proportion of random feature dropout
            mixup_prob: Probability of using mixup augmentation
            rotation_prob: Probability of using random rotation
            p: Overall probability of applying any augmentation
            verify_input: Whether to verify that input features are L2-normalized
        """
        self.noise_level = noise_level
        self.mixup_prob = mixup_prob
        self.rotation_prob = rotation_prob
        self.p = p
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.verify_input = verify_input
    
    def _verify_l2_norm(self, x):
        """Verify if input is L2-normalized"""
        if isinstance(x, np.ndarray):
            norm = np.linalg.norm(x, axis=-1)
            return np.allclose(norm, 1.0, rtol=1e-5, atol=1e-5)
        elif isinstance(x, torch.Tensor):
            norm = torch.norm(x, p=2, dim=-1)
            return torch.allclose(norm, torch.ones_like(norm), rtol=1e-5, atol=1e-5)
        return False
        
    def _l2_normalize(self, x):
        """Apply L2 normalization to input"""
        if isinstance(x, np.ndarray):
            return normalize(x, norm='l2', axis=-1)
        elif isinstance(x, torch.Tensor):
            return F.normalize(x, p=2, dim=-1)
        return x
    
    def add_noise(self, x):
        """Add Gaussian noise"""
        if isinstance(x, np.ndarray):
            noise = np.random.normal(0, self.noise_level, x.shape)
            return self._l2_normalize(x + noise)
        elif isinstance(x, torch.Tensor):
            noise = torch.randn_like(x) * self.noise_level
            return F.normalize(x + noise, p=2, dim=-1)
        return x
    
    def random_rotation_blocks(self, x, max_angle=np.pi/4):
        """
        Apply limited-angle random rotation using diagonal block rotation matrices
        
        Args:
            x: Input features
            max_angle: Maximum rotation angle (radians), default π/4 (45 degrees)
        """
        if isinstance(x, np.ndarray):
            dim = x.shape[-1]
            rotation = np.eye(dim)
            
            # Create a series of 2×2 rotation blocks
            for i in range(0, dim-1, 2):
                # Ensure we have enough dimensions to rotate
                if i+1 >= dim:
                    break
                    
                # Generate random angle within limited range
                # Can be [-max_angle, max_angle] or [0, max_angle]
                theta = np.random.uniform(-max_angle, max_angle)
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                
                # Create 2×2 rotation matrix
                rot_block = np.array([
                    [cos_t, -sin_t],
                    [sin_t, cos_t]
                ])
                
                # Fill rotation matrix into diagonal block
                rotation[i:i+2, i:i+2] = rot_block
            
            # Apply rotation
            return np.dot(x, rotation)
        
        elif isinstance(x, torch.Tensor):
            dim = x.shape[-1]
            rotation = torch.eye(dim, device=x.device)
            
            # Create a series of 2×2 rotation blocks
            for i in range(0, dim-1, 2):
                # Ensure we have enough dimensions to rotate
                if i+1 >= dim:
                    break
                    
                # Generate random angle within limited range
                theta = torch.rand(1, device=x.device) * 2 * max_angle - max_angle
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                
                # Create 2×2 rotation matrix
                rot_block = torch.tensor([
                    [cos_t, -sin_t],
                    [sin_t, cos_t]
                ], device=x.device).squeeze()
                
                # Fill rotation matrix into diagonal block
                rotation[i:i+2, i:i+2] = rot_block
            
            # Apply rotation
            return torch.matmul(x, rotation)
        
        return x
    
    def feature_mixup(self, feature1, feature2, label1, label2, alpha=0.2, l2_normalize=True):
        """
        Perform mixup on two features and labels
        
        Args:
            feature1: First feature
            feature2: Second feature
            label1: First label, optional
            label2: Second label, optional
            alpha: Mixup parameter for beta distribution
            l2_normalize: Whether to apply L2 normalization to mixed features
            
        Returns:
            Mixed features and labels (if labels are provided)
        """
        # Generate mixup coefficient
        lam = np.random.beta(alpha, alpha)
        
        # Mix features
        if isinstance(feature1, np.ndarray):
            mixed_feature = lam * feature1 + (1 - lam) * feature2
            # L2 normalization
            if l2_normalize:
                mixed_feature = normalize(mixed_feature, norm='l2', axis=-1)
        elif isinstance(feature1, torch.Tensor):
            lam_tensor = torch.tensor(lam, device=feature1.device).float()
            mixed_feature = lam_tensor * feature1 + (1 - lam_tensor) * feature2
            # L2 normalization
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
        Apply feature augmentation transforms
        
        Args:
            x: Input features, expected to be L2-normalized, can be numpy array or PyTorch tensor
            
        Returns:
            Augmented features, maintaining L2 normalization
        """
        # Verify if input is L2-normalized
        label = torch.tensor(label)
        label2 = torch.tensor(label2)
        
        if self.verify_input and not self._verify_l2_norm(x):
            x = self._l2_normalize(x)
        
        one_hot = F.one_hot(label, self.num_classes).float()
        # Apply smoothing
        smoothed_label = one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        
        # Decide whether to apply any augmentation with probability p
        if np.random.random() > self.p:
            return x, smoothed_label
            
        # Decide which augmentation method to apply
        r = np.random.random()
        
        if r < 0.33:
            # Add noise
            return self.add_noise(x), smoothed_label
        # elif r < 0.66 and self.rotation_prob > 0:
        #     # Random rotation
        #     return self.random_rotation_blocks(x), smoothed_label
        elif self.mixup_prob > 0:
            # Mixup (Note: this requires a pair of features to work)
            return self.feature_mixup(x, x2, label, label2)
        
        return x, smoothed_label