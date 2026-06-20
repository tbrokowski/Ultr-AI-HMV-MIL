import logging

from ultr_ai.network_architecture.ablation_models import (
    NoRLMultiTaskModel,
    MeanPoolMultiTaskModel,
    AttentionPoolMultiTaskModel,
    SingleTaskMultiTaskModel,
)
from ultr_ai.network_architecture.other_models import (
    ResNet3DMultiTaskModel,
    CNNLSTMMultiTaskModel,
    R2Plus1DMultiTaskModel,
    InceptionMultiTaskModel,
    VideoTransformerMultiTaskModel,
)

logger = logging.getLogger(__name__)


def create_ablation_model(model_type, config):
    """Create a model for HMV-MIL training or ablation experiments."""
    model_map = {
        "attention_pool": AttentionPoolMultiTaskModel,
        "no_rl": NoRLMultiTaskModel,
        "mean_pool": MeanPoolMultiTaskModel,
        "single_task": SingleTaskMultiTaskModel,
        "3d_cnn": ResNet3DMultiTaskModel,
        "cnn_lstm": CNNLSTMMultiTaskModel,
        "R2+1d": R2Plus1DMultiTaskModel,
        "Inception": InceptionMultiTaskModel,
        "video_transformer": VideoTransformerMultiTaskModel,
    }
    if model_type not in model_map:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(model_map.keys())}"
        )
    logger.info("Creating %s model", model_type)
    return model_map[model_type](config)
