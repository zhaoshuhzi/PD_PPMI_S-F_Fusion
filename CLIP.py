import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torchvision import models

class CLIP(nn.Module):
    def __init__(self, embed_dim):
        super(CLIP, self).__init__()
        # 文本编码器：使用BERT
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        
        # 图像编码器：使用ResNet
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, embed_dim)
        
        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_text(self, text):
        # 对输入文本进行tokenize并编码
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        outputs = self.text_encoder(**inputs)
        text_embeddings = outputs.last_hidden_state[:, 0, :]  # 使用[CLS]标记的输出
        return self.text_projection(text_embeddings)

    def encode_image(self, image):
        # 使用图像编码器提取图像特征
        return self.image_encoder(image)

    def forward(self, image, text):
        # 获取图像和文本的嵌入
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # 对嵌入进行归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算对比学习的相似性分数
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        
        return logits_per_image, logits_per_text

def clip_loss(logits_per_image, logits_per_text):
    # 对比损失：使用图像和文本的标签对
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2

# 示例代码：加载模型并进行训练
embed_dim = 512  # 嵌入维度
clip_model = CLIP(embed_dim=embed_dim)

# 假设输入图像和文本数据
sample_images = torch.rand(8, 3, 224, 224)  # (batch_size, channels, height, width)
sample_texts = ["A photo of a cat", "A picture of a dog", "An image of a car", "A portrait of a person"] * 2

# 前向传播和计算损失
logits_image, logits_text = clip_model(sample_images, sample_texts)
loss = clip_loss(logits_image, logits_text)

print("Loss:", loss.item())
