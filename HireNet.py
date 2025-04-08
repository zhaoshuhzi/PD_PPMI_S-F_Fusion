import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ModalityEncoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (B, T, D)
        output, _ = self.rnn(x)
        return output  # (B, T, 2*H)

class GMU(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(GMU, self).__init__()
        self.projections = nn.ModuleList([nn.Linear(d, hidden_dim) for d in input_dims])
        self.gates = nn.ModuleList([nn.Linear(sum(input_dims), hidden_dim) for _ in input_dims])

    def forward(self, inputs):
        # inputs: list of modality tensors [(B, T, D1), (B, T, D2), ...]
        H = []
        for x, proj in zip(inputs, self.projections):
            H.append(torch.tanh(proj(x)))  # Projected to common space

        concat = torch.cat(inputs, dim=-1)
        G = [torch.sigmoid(gate(concat)) for gate in self.gates]

        out = sum([g * h for g, h in zip(G, H)])
        return out  # (B, T, H)

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        attn_scores = self.linear(x).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T)
        weighted_sum = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)  # (B, D)
        return weighted_sum, attn_weights

class ParkinsonProgressionPredictor(nn.Module):
    def __init__(self, input_dims, hidden_dim, num_classes):
        super(ParkinsonProgressionPredictor, self).__init__()
        self.encoders = nn.ModuleList([ModalityEncoder(dim, hidden_dim) for dim in input_dims])
        self.gmu = GMU([2*hidden_dim] * len(input_dims), hidden_dim)
        self.roi_attention = Attention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, inputs):
        # inputs: list of modality tensors [(B, R, T, D)] per modality
        B, R = inputs[0].shape[:2]  # Batch size and number of ROIs
        fused_roi_features = []

        for roi_idx in range(R):
            modality_roi_inputs = [mod[:, roi_idx, :, :] for mod in inputs]  # List of (B, T, D)
            encoded = [encoder(x) for encoder, x in zip(self.encoders, modality_roi_inputs)]
            fused = self.gmu(encoded)  # (B, T, H)
            pooled, _ = self.roi_attention(fused)  # (B, H)
            fused_roi_features.append(pooled)

        roi_tensor = torch.stack(fused_roi_features, dim=1)  # (B, R, H)
        context_vector, attn_weights = self.roi_attention(roi_tensor)  # (B, H)
        logits = self.classifier(context_vector)  # (B, num_classes)
        return logits, attn_weights
