import logging

from ultr_ai.network_architecture.components import MultiTaskModel, UniformFrameSelector

logger = logging.getLogger(__name__)


class NoRLMultiTaskModel(MultiTaskModel):
    """Uniform temporal frame subsampling (UniformSampling / k-equal ablations)."""

    def __init__(self, config):
        config.selection_strategy = "uniform"
        super().__init__(config)
        k_frames = getattr(config, "k_frames", 3)
        self.frame_selector = UniformFrameSelector(
            feature_dim=self.vision_dim,
            output_dim=self.hidden_dim,
            k_frames=k_frames,
        )
        logger.info("Using UniformFrameSelector with k_frames=%s", k_frames)
