import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, bidirectional=True):
        super(RNNEncoder, self).__init__()
        
        # 定义GRU层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, bidirectional=bidirectional)
        
        # 定义全连接层，将GRU的输出映射到最终的特征维度
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)

    def forward(self, feature_sequence):
        """
        输入:
        - feature_sequence: (batch_size, sequence_length, input_dim) 的特征序列张量
        
        输出:
        - global_feature: (batch_size, hidden_dim) 的全局特征表示
        """
        # 通过GRU提取时间序列特征
        gru_output, _ = self.gru(feature_sequence)  # (batch_size, sequence_length, hidden_dim * 2)
        
        # 获取最后一个时间步的特征
        last_output = gru_output[:, -1, :]  # (batch_size, hidden_dim * 2) for bidirectional
        
        # 全连接层映射到最终的特征维度
        global_feature = self.fc(last_output)
        
        return global_feature

# 测试RNN编码器
encoder = RNNEncoder(input_dim=512, hidden_dim=256, num_layers=2, bidirectional=True)
sequence_length = 10  # 例如有10帧的特征序列
batch_size = 8
feature_dim = 512

# 假设输入是从CNN提取的影像特征序列
sample_feature_sequence = torch.randn(batch_size, sequence_length, feature_dim)
features = encoder(sample_feature_sequence)
print("Global feature shape:", features.shape)  # Expected output: (8, 256)
