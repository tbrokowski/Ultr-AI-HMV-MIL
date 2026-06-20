# =============================================================================
# FRAME SELECTORS
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformFrameSelector(nn.Module):
    """No-RL baseline: Uniform temporal subsampling of k frames per site."""
    
    def __init__(self, feature_dim=768, output_dim=512, k_frames=3, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.k_frames = k_frames
        
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
        # Compatibility attributes
        self.saved_actions = []
        self.temperature = 1.0
    
    def get_temperature(self):
        return self.temperature
    
    def clear_history(self):
        self.saved_actions = []
    
    def reset_rewards(self):
        pass
    
    def reset_temperature(self):
        pass
    
    def update_temperature(self, decay=None):
        return self.temperature
    
    def forward(self, features, mask=None, batch_idxs=None, site_idxs=None):
        batch_size, seq_len = features.shape[:2]
        device = features.device
        
        encoded_features = self.feature_projection(features)
        action_logits = torch.ones(batch_size, seq_len, device=device)
        
        if mask is not None:
            action_logits = action_logits.masked_fill(~mask, -1e9)
        
        state_values = torch.zeros(batch_size, 1, device=device, requires_grad=True)
        return action_logits, state_values, encoded_features
    
    def select_action(self, logits, state_values=None, encoded_features=None, batch_idx=None, site_idx=None):
        batch_size, seq_len = logits.shape
        device = logits.device
        
        actions = []
        for b in range(batch_size):
            valid_indices = torch.where(logits[b] > -1e8)[0]
            
            if len(valid_indices) == 0:
                action = torch.tensor(0, device=device)
            elif len(valid_indices) <= self.k_frames:
                selected = valid_indices.repeat((self.k_frames + len(valid_indices) - 1) // len(valid_indices))
                action = selected[0]
            else:
                step = len(valid_indices) // self.k_frames
                uniform_indices = torch.arange(0, len(valid_indices), step, device=device)[:self.k_frames]
                selected_indices = valid_indices[uniform_indices]
                action = selected_indices[0]
            
            actions.append(action)
        
        actions = torch.stack(actions)
        log_probs = torch.zeros_like(actions, dtype=torch.float)
        return actions, log_probs


class MeanPoolSelector(nn.Module):
    """Mean-pool baseline: Average all frame features per site."""
    
    def __init__(self, feature_dim=768, output_dim=512, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
        # Compatibility attributes
        self.saved_actions = []
        self.temperature = 1.0
    
    def get_temperature(self):
        return self.temperature
    
    def clear_history(self):
        self.saved_actions = []
    
    def reset_rewards(self):
        pass
    
    def reset_temperature(self):
        pass
    
    def update_temperature(self, decay=None):
        return self.temperature
    
    def forward(self, features, mask=None, batch_idxs=None, site_idxs=None):
        batch_size, seq_len = features.shape[:2]
        device = features.device
        
        encoded_features = self.feature_projection(features)
        action_logits = torch.ones(batch_size, seq_len, device=device)
        
        if mask is not None:
            action_logits = action_logits.masked_fill(~mask, -1e9)
        
        state_values = torch.zeros(batch_size, 1, device=device, requires_grad=True)
        return action_logits, state_values, encoded_features
    
    def select_action(self, logits, state_values=None, encoded_features=None, batch_idx=None, site_idx=None):
        batch_size = logits.shape[0]
        device = logits.device
        
        actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        log_probs = torch.zeros(batch_size, device=device)
        return actions, log_probs


class AttentionPoolSelector(nn.Module):
    """Attention-pool baseline: Learned attention over frames without RL."""
    
    def __init__(self, feature_dim=768, hidden_dim=512, output_dim=512, num_heads=8, temperature=1.0, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.attention_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
        # Compatibility attributes
        self.saved_actions = []
        # NOTE: Temperature is stored for compatibility but not used in this selector
        # The actual temperature scaling is applied in the main model's process_site() method
        self.temperature = temperature
    
    def get_temperature(self):
        return self.temperature
    
    def clear_history(self):
        self.saved_actions = []
    
    def reset_rewards(self):
        pass
    
    def reset_temperature(self):
        pass
    
    def update_temperature(self, decay=None):
        return self.temperature
    
    def forward(self, features, mask=None, batch_idxs=None, site_idxs=None):
        batch_size, seq_len = features.shape[:2]
        device = features.device
        
        encoded = self.feature_encoder(features)
        
        key_padding_mask = ~mask if mask is not None else None
        attended, _ = self.attention(
            encoded, encoded, encoded,
            key_padding_mask=key_padding_mask
        )
        
        attention_scores = self.attention_scorer(attended).squeeze(-1)
        
        if mask is not None:
            if attention_scores.dtype == torch.float16:
                mask_value = torch.finfo(torch.float16).min
            else:
                mask_value = -1e4  # Safe for both fp32 and fp16
            
            attention_scores = attention_scores.masked_fill(~mask, mask_value)
        
        # Compute softmax for differentiable selection
        # NOTE: Temperature is applied in the main model's process_site() method, not here
        # This frame selector outputs raw attention logits
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, T]
        
        # Apply soft attention to get weighted features
        # This allows gradient to flow through the attention mechanism
        weighted_attended = attention_weights.unsqueeze(-1) * attended  # [B, T, hidden_dim]
        
        output_features = self.output_projection(weighted_attended)
        state_values = torch.zeros(batch_size, 1, device=device, requires_grad=True)
        
        return attention_scores, state_values, output_features
    
    def select_action(self, logits, state_values=None, encoded_features=None, batch_idx=None, site_idx=None):
        """
        Select action using soft attention over all frames (single aggregated vector approach).
        
        NOTE: This method outputs raw logits and dummy actions for API compatibility.
        The actual temperature-scaled attention is computed in the main model's process_site() method.
        The frame selector just provides the raw attention scores.
        """
        batch_size = logits.shape[0]
        device = logits.device
        
        # For backward compatibility, return dummy "actions" (not actually used downstream)
        # The actual frame aggregation happens via soft attention in process_site()
        actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Compute log probabilities (for potential RL integration)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        return actions, action_log_probs