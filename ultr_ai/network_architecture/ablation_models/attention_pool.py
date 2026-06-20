import logging
from ultr_ai.network_architecture.components import MultiTaskModel, AttentionPoolSelector

logger = logging.getLogger(__name__)

class AttentionPoolMultiTaskModel(MultiTaskModel):
    """Ablation: Attention-pool baseline with learned attention (no RL)."""
    
    def __init__(self, config):
        config.selection_strategy = 'attention_pool'
        super().__init__(config)
        
        # Get temperature from config (default 0.5 if not specified)
        temperature = getattr(config, 'attention_temperature', 0.5)
        
        self.frame_selector = AttentionPoolSelector(
            feature_dim=self.vision_dim,
            hidden_dim=1024,
            output_dim=self.hidden_dim,
            num_heads=8,
            temperature=temperature
        )
        logger.info(f"Using AttentionPoolSelector for attention-pool ablation (temperature={temperature})")
