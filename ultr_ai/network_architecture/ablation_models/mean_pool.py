import logging
import torch
from ultr_ai.network_architecture.components import MultiTaskModel, MeanPoolSelector

logger = logging.getLogger(__name__)


class MeanPoolMultiTaskModel(MultiTaskModel):
    """Ablation: Mean-pool baseline - average all frame features per site."""
    
    def __init__(self, config):
        config.selection_strategy = 'mean_pool'
        super().__init__(config)
        
        self.frame_selector = MeanPoolSelector(
            feature_dim=self.vision_dim,
            output_dim=self.hidden_dim
        )
        logger.info("Using MeanPoolSelector for mean-pool ablation")
    
    def process_site(self, video, site_idx, mask=None, batch_idx=None, site_pos=None):
        """Process site with mean pooling - average all valid frames."""
        clip_features = self.extract_clip_features(video)
        
        action_logits, state_values, enhanced_features = self.frame_selector(
            clip_features, mask, batch_idx, site_pos
        )
        
        # Mean pool over all valid frames
        if mask is not None:
            valid_mask = mask[0]
            if valid_mask.any():
                valid_features = enhanced_features[0, valid_mask]
                pooled_features = valid_features.mean(dim=0, keepdim=True)
            else:
                pooled_features = enhanced_features[0, :1]
        else:
            pooled_features = enhanced_features[0].mean(dim=0, keepdim=True)
        
        selected_features = pooled_features.repeat(3, 1).unsqueeze(0)
        selected_mask = torch.ones(1, 3, dtype=torch.bool, device=video.device)
        
        # Process pathologies
        pathology_scores = None
        if self.use_pathology_loss and self.pathology_modules is not None:
            pathology_scores = []
            for module in self.pathology_modules:
                score, _, _ = module(selected_features, selected_mask)
                pathology_scores.append(score)
            pathology_scores = torch.cat(pathology_scores, dim=1)
        
        return {
            'selected_features': selected_features,
            'selected_indices': torch.arange(3, device=video.device).unsqueeze(0),
            'pathology_scores': pathology_scores,
            'action_logits': action_logits,
            'state_values': state_values,
            'batch_idx': batch_idx,
            'site_idx': site_pos
        }