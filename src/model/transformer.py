import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttentionOne(nn.Module):
    """
    Multi-Head Attention module with shared projection
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttentionOne, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qkvs = nn.Linear(d_model, n_head * d_k, bias=False)
        nn.init.normal_(self.w_qkvs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        k = k.view(k.size()[0], k.size()[1], -1)  # [bz, c, h, w]
        v = v.view(v.size()[0], v.size()[1], -1)  # [bz, c, h, w]

        k = k.permute(0, 2, 1).contiguous()  # [bz, hw, c]
        v = v.permute(0, 2, 1).contiguous()  # [bz, hw, c]

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qkvs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_qkvs(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_qkvs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # [(n*b), lq, dk]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # [(n*b), lk, dk]
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # [(n*b), lv, dv]

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # [b, lq, (n*dv)]

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class AttentionExtractor(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.attention = MultiHeadAttentionOne(n_head, d_model, d_k, d_v)

    def forward(self, q, s):
        """
        q: (batch_size, hidden_dim, h, w)  --> Feature map
        s: (5 * batch_size, hidden_dim)   --> Reference features
        """
        batch_size, hidden_dim, h, w = q.shape
        batch_size, n_shots, hidden_dim = s.shape

        # Flatten q to (batch_size, h*w, hidden_dim)
        q = q.view(batch_size, hidden_dim, -1).permute(0, 2, 1)  # (batch_size, h*w, hidden_dim)

        s = s.view(batch_size, hidden_dim, n_shots)  

        # Compute attention
        attended_q = self.attention(q, s, s)  # (batch_size, h*w, hidden_dim)

        # Reshape output back to (batch_size, hidden_dim, h, w)
        attended_q = attended_q.permute(0, 2, 1).view(batch_size, hidden_dim, h, w)

        return attended_q


class ConvFusion(nn.Module):
    """
    Combines f_q, f_q+, and f_q-
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # self.conv = nn.Conv2d(3 * hidden_dim, hidden_dim, kernel_size=1)  # 1x1 conv fusion
        self.conv = nn.Sequential(
            nn.Conv2d(3 * hidden_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

    def forward(self, x1, x2, x3):
        fused = torch.cat([x1, x2, x3], dim=1)  # [batch_size, 3 * hidden_dim, h, w]
        fused = self.conv(fused)  # [batch_size, hidden_dim, h, w]
        return fused
