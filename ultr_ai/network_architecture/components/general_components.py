import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MOBILE_BACKBONES = [
    "mobilenetv3_large_100",
    "efficientnet_lite0",
    "ghostnetv2_100",
    "regnety_400mf",
    "levit_256",
    "deit_tiny_distilled_patch16_224",
]


class PathologyModule(nn.Module):
    """Per-pathology frame-weighted classifier for a single ultrasound site."""

    def __init__(self, feature_dim=512, hidden_dim=256, dropout=0.3, name=None):
        super().__init__()
        self.name = name
        self.feature_refine = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.frame_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, mask=None):
        refined = self.feature_refine(features)
        attn_logits = self.frame_attention(refined)
        if mask is not None:
            attn_logits = attn_logits.float().masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = torch.bmm(attn_weights.transpose(1, 2), refined)
        score = self.classifier(pooled.squeeze(1))
        return score, attn_weights.squeeze(-1), pooled.squeeze(1)


class SiteIntegrationModule(nn.Module):
    """Fuse site features with anatomical embeddings and pathology scores."""

    def __init__(
        self,
        feature_dim=512,
        site_embed_dim=256,
        hidden_dim=512,
        num_sites=15,
        num_pathologies=5,
        dropout=0.3,
    ):
        super().__init__()
        self.site_embedding = nn.Embedding(num_sites + 1, site_embed_dim)
        self.integration = nn.Sequential(
            nn.Linear(feature_dim + site_embed_dim + num_pathologies, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, site_features, site_indices, pathology_scores):
        site_embeddings = self.site_embedding(site_indices)
        combined = torch.cat([site_features, site_embeddings, pathology_scores], dim=2)
        return self.integration(combined)


class DeepAttentionMIL(nn.Module):
    """Patient-level attention MIL over integrated site features."""

    def __init__(self, feature_dim=512, hidden_dim=512, dropout=0.3, num_heads=8):
        super().__init__()
        self.attention1 = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.attention2 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.gating = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features, mask=None):
        key_padding_mask = ~mask if mask is not None else None
        attended_features, _ = self.attention1(
            features, features, features, key_padding_mask=key_padding_mask
        )
        transformed = self.transform(attended_features)
        enhanced = transformed + features
        attn_logits = self.attention2(enhanced).squeeze(-1)

        if mask is not None:
            attn_logits = attn_logits.float()
            for i in range(attn_logits.size(0)):
                for j in range(attn_logits.size(1)):
                    if j < mask.size(1) and not mask[i, j]:
                        attn_logits[i, j] = -1e9

        attn_weights = F.softmax(attn_logits, dim=1)
        gate_weights = self.gating(enhanced)
        gated_attention = attn_weights.unsqueeze(-1) * gate_weights
        normalizer = gated_attention.sum(dim=1, keepdim=True) + 1e-6
        normalized_attention = gated_attention / normalizer
        aggregated = torch.bmm(
            normalized_attention.transpose(1, 2), enhanced
        ).squeeze(1)
        return aggregated, attn_weights
