import logging

from ultr_ai.network_architecture.ablation_models.attention_pool import AttentionPoolMultiTaskModel

logger = logging.getLogger(__name__)


class SingleTaskMultiTaskModel(AttentionPoolMultiTaskModel):
    """HMV-MIL attention pooling without pathology auxiliary heads."""

    def __init__(self, config):
        config.use_pathology_loss = False
        super().__init__(config)
        logger.info("Attention-pool model without pathology detection (NoPathology ablation)")
