import torch
import torch.nn as nn
import torch.nn.functional as F

class TransFusion(nn.Module):
    def __init__(self, input_dim, fusion_dim, transform_dim):
        super(TransFusion, self).__init__()
        
        # 下采样模块
        self.downsample = nn.Sequential(
            nn.Conv2d(input_dim, transform_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(transform_dim),
            nn.ReLU()
        )
        
        # 上采样模块
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(transform_dim, fusion_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU()
        )
        
        # 双融合模块
        self.bifusion = Bifusion(transform_dim, fusion_dim)

    def forward(self, structure_feat, function_feat):
        # 下采样结构和功能特征
        downsampled_structure = self.downsample(structure_feat)
        downsampled_function = self.downsample(function_feat)
        
        # 上采样后的特征
        upsampled_structure = self.upsample(downsampled_structure)
        upsampled_function = self.upsample(downsampled_function)
        
        # 双融合过程
        fused_output = self.bifusion(downsampled_function, upsampled_structure)
        
        return fused_output


class Bifusion(nn.Module):
    def __init__(self, down_dim, up_dim):
        super(Bifusion, self).__init__()
        
        # 双融合的两个分支
        self.function_branch = nn.Sequential(
            nn.Conv2d(down_dim, up_dim, kernel_size=1),
            nn.ReLU()
        )
        
        self.structure_branch = nn.Sequential(
            nn.Conv2d(up_dim, up_dim, kernel_size=1),
            nn.ReLU()
        )

        # 最终的融合层
        self.fusion_layer = nn.Conv2d(2 * up_dim, up_dim, kernel_size=1)
    
    def forward(self, function_feat, structure_feat):
        # 功能特征分支
        function_fusion = self.function_branch(function_feat)
        
        # 结构特征分支
        structure_fusion = self.structure_branch(structure_feat)
        
        # 融合两个分支的输出
        fusion_output = torch.cat([function_fusion, structure_fusion], dim=1)
        fusion_output = self.fusion_layer(fusion_output)
        
        return fusion_output


# 示例代码：测试TransFusion模型
input_dim = 64   # 输入维度
fusion_dim = 128 # 融合后的维度
transform_dim = 64 # 中间转换维度

transfusion_model = TransFusion(input_dim, fusion_dim, transform_dim)

# 假设输入特征为结构和功能的特征表示
structure_feat = torch.randn(8, input_dim, 64, 64) # Batch size 8, input_dim 64, spatial 64x64
function_feat = torch.randn(8, input_dim, 64, 64)  # Same dimensions for function features

# 前向传播
output = transfusion_model(structure_feat, function_feat)
print("Output shape:", output.shape)
