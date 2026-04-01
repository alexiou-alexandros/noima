"""
NOIMAv2 model architecture — exact match to noima_v2plus_best.pt weights.
Hybrid Conv1D + Transformer with body-part MLPs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        attn = x.mean(dim=1)
        attn = self.conv(attn.unsqueeze(1)).squeeze(1)
        attn = torch.sigmoid(attn).unsqueeze(1)
        return x * attn


class Conv1DBlock(nn.Module):
    def __init__(self, dim, kernel_size=17, expand_ratio=2, drop_rate=0.1):
        super().__init__()
        hidden = dim * expand_ratio
        self.expand = nn.Sequential(nn.Linear(dim, hidden), nn.Mish())
        self.dw_conv = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.eca = ECA()
        self.project = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.dw_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.eca(x)
        x = self.project(x)
        x = self.drop(x)
        return x + residual


class TransformerBlock(nn.Module):
    def __init__(self, dim, nhead=8, expand=2, attn_dropout=0.1, drop_rate=0.1):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(drop_rate)
        self.norm2 = nn.BatchNorm1d(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * expand), nn.Mish(), nn.Linear(dim * expand, dim),
        )
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x, mask=None):
        normed = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        key_padding_mask = (mask == 0) if mask is not None else None
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + self.drop1(attn_out)
        normed = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop2(self.ff(normed))
        return x


class BodyPartMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Mish(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.mlp(x)


class NOIMAv2(nn.Module):
    def __init__(self, num_classes=310, dim=256, nhead=8, num_conv_blocks=3,
                 num_transformer_blocks=2, drop_rate=0.15, max_frames=64):
        super().__init__()
        self.dim = dim
        self.max_frames = max_frames
        self.num_transformer_blocks = num_transformer_blocks

        self.pose_mlp  = BodyPartMLP(66, dim // 4)
        self.lhand_mlp = BodyPartMLP(126, dim // 4)
        self.rhand_mlp = BodyPartMLP(126, dim // 4)
        self.face_mlp  = BodyPartMLP(132, dim // 4)

        self.fusion = nn.Sequential(
            nn.Linear(dim * 3 // 4, dim),
            nn.BatchNorm1d(dim),
            nn.Mish(),
        )

        self.conv_blocks_1 = nn.ModuleList([
            Conv1DBlock(dim, kernel_size=17, drop_rate=drop_rate) for _ in range(num_conv_blocks)
        ])
        self.transformer_1 = TransformerBlock(dim, nhead=nhead, drop_rate=drop_rate)

        self.conv_blocks_2 = nn.ModuleList([
            Conv1DBlock(dim, kernel_size=17, drop_rate=drop_rate) for _ in range(num_conv_blocks)
        ])
        self.transformer_2 = TransformerBlock(dim, nhead=nhead, drop_rate=drop_rate)

        self.head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Mish(),
            nn.Dropout(0.4),
            nn.Linear(dim * 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _extract_body_parts(self, x):
        pose  = torch.cat([x[:, :, 0:22],    x[:, :, 150:172], x[:, :, 300:322]], dim=-1)
        lhand = torch.cat([x[:, :, 22:64],   x[:, :, 172:214], x[:, :, 322:364]], dim=-1)
        rhand = torch.cat([x[:, :, 64:106],  x[:, :, 214:256], x[:, :, 364:406]], dim=-1)
        face  = torch.cat([x[:, :, 106:150], x[:, :, 256:300], x[:, :, 406:450]], dim=-1)
        return pose, lhand, rhand, face

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        pose, lhand, rhand, face = self._extract_body_parts(x)
        pose_fts  = self.pose_mlp(pose.reshape(B * T, -1))
        lhand_fts = self.lhand_mlp(lhand.reshape(B * T, -1))
        rhand_fts = self.rhand_mlp(rhand.reshape(B * T, -1))
        face_fts  = self.face_mlp(face.reshape(B * T, -1))

        hand_fts = torch.stack([lhand_fts, rhand_fts], dim=-1).amax(dim=-1)
        fts = torch.cat([pose_fts, hand_fts, face_fts], dim=-1)
        fts = self.fusion(fts)
        fts = fts.view(B, T, -1)

        for conv in self.conv_blocks_1:
            fts = conv(fts)
        fts = self.transformer_1(fts, mask)

        for conv in self.conv_blocks_2:
            fts = conv(fts)
        fts = self.transformer_2(fts, mask)

        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            fts = (fts * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        else:
            fts = fts.mean(dim=1)

        return self.head(fts)
