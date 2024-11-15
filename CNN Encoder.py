import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3, feature_dim=512):
        super(CNNEncoder, self).__init__()
        
        # 定义卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 8x8 -> 4x4
        )
        
        # 定义一个全连接层，用于将卷积特征映射到最终的特征维度
        self.fc = nn.Linear(512 * 4 * 4, feature_dim)

    def forward(self, x):
        # 卷积层提取特征
        x = self.conv_layers(x)
        
        # 扁平化操作
        x = x.view(x.size(0), -1)  # Flatten
        
        # 全连接层映射到指定的特征维度
        x = self.fc(x)
        
        return x

# 测试CNN编码器
encoder = CNNEncoder(input_channels=3, feature_dim=512)
sample_image = torch.randn(8, 3, 64, 64)  # Batch size 8, 3 channels, 64x64 resolution
features = encoder(sample_image)
print("Feature shape:", features.shape)  # Expected output: (8, 512)
