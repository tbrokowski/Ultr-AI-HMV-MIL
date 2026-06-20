"""YAML-backed training configuration for HMV-MIL and ablation experiments."""

import logging
import os
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class MultiTaskConfig:
    """Configuration object; all YAML keys are accepted as attributes."""

    def __init__(self) -> None:
        self.train = True
        self.evaluate_best_valid_model = True
        self.root_dir = "./Data"
        self.labels_csv = "./Data/labels/labels_multidiagnosis.csv"
        self.file_metadata_csv = "./Data/processed_files_2.csv"
        self.split_csv = "./Data/test_files/Fold_0.csv"
        self.video_folder = "./Data/LusBeninVideos"
        self.image_folder = "images"
        self.model_type = "attention_pool"
        self.model_name = "hmv_mil"
        self.backbone = "clip"
        self.backbone_model_name = "openai/clip-vit-base-patch32"
        self.freeze_backbone = False
        self.pretrained = True
        self.local_weights_dir = "./CLIP_weights"
        self.hidden_dim = 512
        self.dropout_rate = 0.2
        self.num_pathologies = 4
        self.num_sites = 21
        self.num_classes = 1
        self.in_channels = 3
        self.active_tasks = ["TB Label"]
        self.use_pathology_loss = True
        self.use_patient_mil = True
        self.task_weights = {"TB Label": 1.0}
        self.task_pos_weights = {"TB Label": 2.0}
        self.pathology_weight = 1.2
        self.pathology_pos_weights = [1.0, 4.0, 15.0, 4.0]
        self.pathology_classes = [
            "A-line",
            "Large Consolidations",
            "Pleural Effusion",
            "Other Pathology",
        ]
        self.batch_size = 2
        self.num_workers = 6
        self.frame_sampling = 32
        self.depth_filter = "15"
        self.target_height = 224
        self.target_width = 224
        self.files_per_site = "all"
        self.site_order = None
        self.pad_missing_sites = True
        self.max_sites = None
        self.mode = "video"
        self.pooling = "attention"
        self.num_epochs = 200
        self.accumulation_steps = 35
        self.use_amp = True
        self.learning_rate = 2e-5
        self.weight_decay = 1e-5
        self.backbone_lr = 5e-6
        self.backbone_weight_decay = 1e-5
        self.backbone_eta_min = 5e-7
        self.pathology_lr = 1e-5
        self.pathology_weight_decay = 1e-5
        self.pathology_eta_min = 5e-7
        self.patient_pipeline_lr = 1.2e-5
        self.patient_pipeline_weight_decay = 1e-5
        self.patient_pipeline_eta_min = 5e-7
        self.attention_temperature = 0.2
        self.entropy_weight = 0.001
        self.k_frames = 3
        self.pos_weight = 2.0
        self.eval_metric = "auc"
        self.eval_metric_goal = "max"
        self.early_stopping_patience = 50
        self.seed = 42
        self.device = "cuda"
        self.experiment_name = "hmv_mil"
        self.experiment_dir = "./outputs/experiments"
        self.log_dir = "./outputs/logs"
        self.save_dir = "./outputs/saves"
        self.checkpoint_dir = "./outputs/checkpoints"
        self.pred_save_dir = "./outputs/predictions"
        self.checkpoint_base_dir = "./outputs"
        self.model_weights = None
        self.best_model_path = None
        self.resume_from_checkpoint = None
        self.reset_optimizers = False
        self.task = "TB Label"
        self.classification_type = "binary"
        self.distributed = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        for key, value in (data or {}).items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, filepath: str) -> "MultiTaskConfig":
        with open(filepath, "r") as f:
            data = yaml.safe_load(f) or {}
        config = cls()
        config.update_from_dict(data)
        if not getattr(config, "experiment_dir", None):
            config.experiment_dir = os.path.join(
                config.checkpoint_dir, config.model_name
            )
        return config


def load_config(
    config_name: Optional[str] = None,
    config_file: Optional[str] = None,
    **kwargs: Any,
) -> MultiTaskConfig:
    if config_file and os.path.exists(config_file):
        logger.info("Loading configuration from %s", config_file)
        config = MultiTaskConfig.load(config_file)
    else:
        if config_file:
            logger.warning("Config file not found: %s; using defaults", config_file)
        config = MultiTaskConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def ensure_output_dirs(config: MultiTaskConfig) -> None:
    for key in ("experiment_dir", "checkpoint_dir", "log_dir", "save_dir", "pred_save_dir"):
        path = getattr(config, key, None)
        if path:
            os.makedirs(path, exist_ok=True)
