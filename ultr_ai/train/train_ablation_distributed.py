import os
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '0'
if 'NCCL_NET_PLUGIN' in os.environ:
    del os.environ['NCCL_NET_PLUGIN']
if 'NCCL_SOCKET_IFNAME' in os.environ:
    del os.environ['NCCL_SOCKET_IFNAME']



import sys
import yaml
import pathlib
import argparse
import numpy as np
import random
from tqdm import tqdm
import gc
import logging
from contextlib import nullcontext

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultr_ai.config import load_config, ensure_output_dirs
from ultr_ai.dataset import LungUltrasoundDataModule, collate_patient_batch
from ultr_ai.network_architecture.factory import create_ablation_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def setup_distributed():
    """Initialize the distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        # Set device BEFORE initializing process group
        torch.cuda.set_device(local_rank)
        
        # Initialize process group with device_id specified
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30),
        )
        
        if rank == 0:
            logger.info(f"Distributed training initialized: {world_size} GPUs")
            logger.info(f"NCCL version: {torch.cuda.nccl.version()}")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get the total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def reduce_dict(input_dict, average=True):
    """
    Reduce values in a dictionary across all processes.
    
    Args:
        input_dict: Dictionary with values to reduce
        average: Whether to average or sum the values
    """
    if not dist.is_initialized():
        return input_dict
    
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        reduced_dict = {k: v.item() for k, v in zip(names, values)}
    
    return reduced_dict


# ============================================================================
# DDP Helper: Gather and Concatenate Numpy Arrays
# ============================================================================
def _ddp_concat_numpy(local_arr):
    """
    Gather numpy arrays from all ranks and concatenate along axis 0.
    If not in distributed mode, returns local_arr.
    """
    if not dist.is_initialized():
        return local_arr
    world_size = dist.get_world_size()
    parts = [None] * world_size
    # all_gather_object works with NCCL/Gloo and arbitrary Python objects
    dist.all_gather_object(parts, local_arr)
    parts = [p for p in parts if p is not None and len(p) > 0]
    if len(parts) == 0:
        return local_arr
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=0)


# ============================================================================
# Configuration Class
# ============================================================================

class Config:
    def __init__(self, params=None):
    """
    Ablation model trainer with full distributed training support.
    Supports multi-GPU and multi-node training with proper gradient accumulation.
    """
    
    def __init__(self, config, rank=0, world_size=1, local_rank=0):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = world_size > 1
        
        # Set device based on local rank
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        
        config.device = self.device
        
        # Only log from main process
        if is_main_process():
            # Main process: ensure INFO level
            logging.getLogger().setLevel(logging.INFO)
            logger.setLevel(logging.INFO)
            print("✓ Main process: Logger set to INFO level")
        else:
            # Other processes: reduce to WARNING
            logging.getLogger().setLevel(logging.WARNING)
            logger.setLevel(logging.WARNING)
            print(f"  Worker process {rank}: Logger set to WARNING level")
            
        print(f"\n{'='*80}")
        print(f"PROCESS INITIALIZATION:")
        print(f"  Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        print(f"  is_distributed: {self.is_distributed}")
        print(f"  is_main_process(): {is_main_process()}")
        print(f"{'='*80}\n")
    
        
        # Training state
        self.best_metric = None
        self.best_epoch = 0
        self.epoch = 0
        self.epochs_without_improvement = 0
        
        # Model configuration
        self.model_type = getattr(config, 'model_type', 'no_rl')
        self.active_tasks = list(getattr(config, 'active_tasks', ['TB Label']))
        self.use_pathology_loss = bool(getattr(config, 'use_pathology_loss', True))

        # Task weights (dict) and positive-class weights
        self.task_weights = dict(getattr(config, 'task_weights', {'TB Label': 1.0}))

        if hasattr(config, 'task_pos_weights'):
            # YAML dict, e.g., {"TB Label": 1.4}
            self.task_pos_weights = dict(getattr(config, 'task_pos_weights'))
        else:
            # Fallback: scalar pos_weight applied to all active tasks
            self.task_pos_weights = {task: float(getattr(config, 'pos_weight', 1.0))
                                    for task in self.active_tasks}

        # Optional global pathology loss weight from YAML
        self.pathology_weight = float(getattr(config, 'pathology_weight', 1.0))
        
        if is_main_process():
            logger.info(f"Using ablation model type: {self.model_type}")
            logger.info(f"Pathology loss enabled: {self.use_pathology_loss}")
            logger.info(f"Distributed training: {self.is_distributed} (world_size={world_size})")
        
        self._set_seed(config.seed)
        
        self._setup_data()
        self._setup_model()
        self._setup_training()
    
    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        # Add rank to seed for data loading diversity
        seed = seed + self.rank
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_data(self):
        """Set up the data module with distributed samplers."""
        self.data_module = LungUltrasoundDataModule(
            root_dir=self.config.root_dir,
            labels_csv=self.config.labels_csv,
            file_metadata_csv=self.config.file_metadata_csv,
            image_folder=self.config.image_folder,
            video_folder=self.config.video_folder,
            split_csv=self.config.split_csv,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            frame_sampling=self.config.frame_sampling,
            depth_filter=self.config.depth_filter,
            cache_size=100,
            files_per_site=getattr(self.config, 'files_per_site', 1),
            site_order=getattr(self.config, 'site_order', None),
            pad_missing_sites=getattr(self.config, 'pad_missing_sites', True),
            max_sites=getattr(self.config, 'max_sites', 15),
        )
        
        self.data_module.setup(stage='patient_level')
        
        # Create distributed samplers if using multiple GPUs
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                self.data_module.patient_train,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.config.seed,
                drop_last=True  # Important for consistent batch sizes
            )
            self.val_sampler = DistributedSampler(
                self.data_module.patient_val,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            self.test_sampler = DistributedSampler(
                self.data_module.patient_test,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None
        
        # Create data loaders
        self.train_loader = DataLoader(
        self.data_module.patient_train,
        batch_size=self.config.batch_size,
        sampler=self.train_sampler,                 # DDP sampler
        shuffle=(self.train_sampler is None),
        num_workers=self.config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_patient_batch,          
        prefetch_factor=2 if self.config.num_workers > 0 else None,
        persistent_workers=True if self.config.num_workers > 0 else False,
    )

        self.val_loader = DataLoader(
            self.data_module.patient_val,
            batch_size=self.config.batch_size,
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_patient_batch,          
            prefetch_factor=2 if self.config.num_workers > 0 else None,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )

        self.test_loader = DataLoader(
            self.data_module.patient_test,
            batch_size=self.config.batch_size,
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_patient_batch,           
            prefetch_factor=2 if self.config.num_workers > 0 else None,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )
        
        if is_main_process():
            logger.info(f"Training dataset size: {len(self.data_module.patient_train)}")
            logger.info(f"Validation dataset size: {len(self.data_module.patient_val)}")
            logger.info(f"Test dataset size: {len(self.data_module.patient_test)}")
            logger.info(f"Training batches per epoch: {len(self.train_loader)}")
    
    def _setup_model(self):
        """Set up the ablation model with DDP support."""
        self.model = create_ablation_model(self.model_type, self.config)
        
        # Load pretrained weights if provided
        if hasattr(self.config, 'model_weights') and self.config.model_weights:
            try:
                checkpoint = torch.load(
                    self.config.model_weights,
                    map_location=self.device,
                    weights_only=False
                )
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (from previous DDP training)
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                self.model.load_state_dict(state_dict)
                
                if is_main_process():
                    logger.info(f"Loaded model weights from {self.config.model_weights}")
            except Exception as e:
                if is_main_process():
                    logger.error(f"Failed to load pretrained weights: {e}")
                    logger.error("Continuing with randomly initialized weights")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Wrap with DDP if using distributed training
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True  # Set to False if all params are always used
            )
            if is_main_process():
                logger.info("Model wrapped with DistributedDataParallel")
        
        # Get the underlying model (unwrapped)
        self.model_without_ddp = self.model.module if self.is_distributed else self.model
        
        trainable_params = sum(p.numel() for p in self.model_without_ddp.parameters() 
                              if p.requires_grad)
        
        if is_main_process():
            logger.info(f"Number of trainable parameters: {trainable_params:,}")
            try:
            except:
                pass
    
    def _setup_training(self):
        """Set up optimizers and schedulers."""
        # Organize parameters by component
        backbone_params = []
        pathology_params = []
        patient_pipeline_params = []
        task_classifier_params = []
        
        num_pathology_modules = 0
        if hasattr(self.model_without_ddp, 'pathology_modules') and \
           self.model_without_ddp.pathology_modules:
            num_pathology_modules = len(self.model_without_ddp.pathology_modules)
        
        pathology_module_params = [[] for _ in range(num_pathology_modules)]
        
        # Categorize parameters
        for name, param in self.model_without_ddp.named_parameters():
            if not param.requires_grad:
                continue
            
            if any(component in name for component in [
                'vision_encoder', 'cnn_backbone', 'video_transformer', 'backbone',
                'multi_feature_extraction', 'multi_scale_extraction'
            ]):
                backbone_params.append(param)
            elif 'pathology_modules' in name and self.use_pathology_loss:
                for i in range(num_pathology_modules):
                    if f'pathology_modules.{i}' in name or f'pathology_modules[{i}]' in name:
                        pathology_module_params[i].append(param)
                        break
            elif 'task_classifiers' in name or 'tb_classifier' in name:
                task_classifier_params.append(param)
            elif any(component in name for component in [
                'site_integration', 'patient_mil', 'cross_site_attention', 'frame_selector'
            ]):
                patient_pipeline_params.append(param)
            else:
                patient_pipeline_params.append(param)
        
        # Create optimizers
        if backbone_params:
            self.backbone_optimizer = optim.AdamW(
                backbone_params,
                lr=getattr(self.config, 'backbone_lr', 0.00001),
                weight_decay=getattr(self.config, 'backbone_weight_decay', 0.00001)
            )
            if is_main_process():
                logger.info(f"Created backbone optimizer with {len(backbone_params)} parameters")
        else:
            self.backbone_optimizer = None
            if is_main_process():
                logger.info("No backbone parameters found")
        
        # Pathology optimizers
        self.pathology_optimizers = []
        if self.use_pathology_loss and num_pathology_modules > 0:
            for i, module_params in enumerate(pathology_module_params):
                if module_params:
                    optimizer = optim.AdamW(
                        module_params,
                        lr=getattr(self.config, 'pathology_lr', 0.0001),
                        weight_decay=getattr(self.config, 'pathology_weight_decay', 0.00001)
                    )
                    self.pathology_optimizers.append(optimizer)
            if is_main_process():
                logger.info(f"Created {len(self.pathology_optimizers)} pathology optimizers")
        
        # Patient pipeline optimizer
        all_patient_params = patient_pipeline_params + task_classifier_params
        if all_patient_params:
            self.patient_pipeline_optimizer = optim.AdamW(
                all_patient_params,
                lr=getattr(self.config, 'patient_pipeline_lr', 0.001),
                weight_decay=getattr(self.config, 'patient_pipeline_weight_decay', 0.00001)
            )
            if is_main_process():
                logger.info(f"Created patient pipeline optimizer with {len(all_patient_params)} parameters")
        else:
            self.patient_pipeline_optimizer = None
            if is_main_process():
                logger.info("No patient pipeline parameters found")
        
        # Set up schedulers
        self.schedulers = []
        batches_per_epoch = len(self.train_loader)
        total_steps = self.config.num_epochs * batches_per_epoch
        
        if self.backbone_optimizer:
            self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                self.backbone_optimizer,
                T_max=total_steps,
                eta_min=getattr(self.config, 'backbone_eta_min', 1e-6)
            ))
        
        for optimizer in self.pathology_optimizers:
            self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=getattr(self.config, 'pathology_eta_min', 1e-6)
            ))
        
        if self.patient_pipeline_optimizer:
            self.schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                self.patient_pipeline_optimizer,
                T_max=total_steps,
                eta_min=getattr(self.config, 'patient_pipeline_eta_min', 1e-6)
            ))
        
        # Mixed precision training
        self.use_amp = self.config.use_amp and torch.cuda.is_available()
        if self.use_amp:
            if self.backbone_optimizer:
                self.backbone_scaler = torch.amp.GradScaler('cuda')
            self.pathology_scalers = [torch.amp.GradScaler('cuda') for _ in self.pathology_optimizers]
            if self.patient_pipeline_optimizer:
                self.patient_pipeline_scaler = torch.amp.GradScaler('cuda')
        
        if is_main_process():
            logger.info(f"Mixed precision training: {self.use_amp}")
    
    def train_epoch(self, epoch):
        """
        Training epoch with proper distributed gradient accumulation.
        """
        # Set epoch for distributed sampler
        if self.is_distributed and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        self.model.train()
        self.epoch = epoch
        
        # Initialize tracking metrics
        running_losses = {
            'total': 0.0,
            'tb_loss': 0.0,
            'attention_entropy': 0.0,
            'attention_entropy_penalty': 0.0,
        }
        
        if self.use_pathology_loss:
            running_losses['pathology'] = 0.0
        
        # Metrics tracking
        all_tb_targets = []
        all_tb_predictions = []
        all_tb_logits = []
        
        pathology_labels_list = []
        pathology_scores_list = []
        pathology_masks_list = []
        
        accumulation_steps = self.config.accumulation_steps
        
        # Calculate effective batch size
        effective_batch_size = self.config.batch_size * accumulation_steps
        if self.is_distributed:
            effective_batch_size *= self.world_size
        
        if is_main_process():
            logger.info(f"Effective batch size: {effective_batch_size} "
                       f"(batch_size={self.config.batch_size} × "
                       f"accumulation_steps={accumulation_steps} × "
                       f"num_gpus={self.world_size})")
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            disable=not is_main_process()
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Determine if this is a sync step
                is_accumulation_step = (batch_idx + 1) % accumulation_steps != 0
                is_last_batch = (batch_idx + 1) == len(self.train_loader)
                should_sync = not is_accumulation_step or is_last_batch
                
                # Move data to device
                site_videos = batch['site_videos'].to(self.device, non_blocking=True)
                site_indices = batch['site_indices'].to(self.device, non_blocking=True)
                # site_masks may be missing in some DataLoader outputs; guard against KeyError
                if 'site_masks' in batch and batch['site_masks'] is not None:
                    site_masks = batch['site_masks'].to(self.device, non_blocking=True)
                else:
                    print("site_masks not found in batch; creating default mask")
                    # Create a default mask (all valid) matching site_findings or site_indices
                    if 'site_findings' in batch and batch['site_findings'] is not None:
                        sf = batch['site_findings']
                        # site_findings shape might be (B, N, P) or (B, N)
                        N = sf.shape[1] if sf.ndim >= 2 else (batch['site_indices'].shape[1] if 'site_indices' in batch else getattr(self.config, 'max_sites', 15))
                    elif 'site_indices' in batch and batch['site_indices'] is not None:
                        N = batch['site_indices'].shape[1]
                    else:
                        N = getattr(self.config, 'max_sites', 15)

                    site_masks = torch.ones((batch['site_videos'].shape[0], N), dtype=torch.bool, device=self.device)
                site_findings = batch['site_findings'].to(self.device, non_blocking=True)
                tb_labels = batch['tb_labels'].to(self.device, non_blocking=True).float()
                pneumonia_labels = torch.full_like(tb_labels, -1)
                covid_labels = torch.full_like(tb_labels, -1)
                
                inputs = {
                    'site_videos': site_videos,
                    'site_indices': site_indices,
                    'site_masks': site_masks,
                    'site_findings': site_findings,
                    'is_patient_level': True
                }
                
                targets = {
                    'tb_labels': tb_labels,
                    'pneumonia_labels': pneumonia_labels,
                    'covid_labels': covid_labels,
                    'pathology_labels': site_findings,
                    'site_masks': site_masks,
                }
                
                # ============================================
                # 1. Pathology Modules Update
                # ============================================
                if self.use_pathology_loss and self.pathology_optimizers:
                    for path_idx in range(self.config.num_pathologies):
                        if path_idx >= len(self.pathology_optimizers):
                            continue
                        
                        # Freeze all parameters
                        for param in self.model_without_ddp.parameters():
                            param.requires_grad = False
                        
                        # Unfreeze only this pathology module
                        for name, param in self.model_without_ddp.named_parameters():
                            if f'pathology_modules.{path_idx}' in name or \
                               f'pathology_modules[{path_idx}]' in name:
                                param.requires_grad = True
                        
                        # Zero gradients at start of accumulation
                        if batch_idx % accumulation_steps == 0:
                            self.pathology_optimizers[path_idx].zero_grad()
                        
                        try:
                            # Use context manager for gradient synchronization
                            sync_context = self.model.no_sync if (self.is_distributed and not should_sync) else nullcontext
                            
                            with sync_context():
                                if self.use_amp:
                                    with torch.amp.autocast('cuda'):
                                        path_outputs = self.model(inputs)
                                        path_scores = path_outputs['pathology_scores']
                                        
                                        if path_scores.dim() == 3:
                                            path_score_i = path_scores[:, :, path_idx]
                                            path_label_i = site_findings[:, :, path_idx] if site_findings.dim() == 3 else site_findings[:, path_idx]
                                        else:
                                            path_score_i = path_scores[:, path_idx]
                                            path_label_i = site_findings[:, path_idx]
                                        
                                        valid_mask = path_label_i >= 0
                                        if valid_mask.any():
                                            pos_weight = getattr(self.config, 'pathology_pos_weights', [2.0, 4.0, 3.0])[path_idx] if hasattr(self.config, 'pathology_pos_weights') else 2.0
                                            pos_weight_tensor = torch.tensor(pos_weight, device=self.device)
                                            
                                            path_loss = F.binary_cross_entropy_with_logits(
                                                path_score_i[valid_mask],
                                                path_label_i[valid_mask].float(),
                                                pos_weight=pos_weight_tensor
                                            )
                                            
                                            # Scale loss by accumulation steps
                                            path_loss = path_loss / accumulation_steps
                                            running_losses['pathology'] += path_loss.item() * accumulation_steps
                                            
                                            self.pathology_scalers[path_idx].scale(path_loss).backward()
                                    
                                    del path_outputs, path_scores, path_score_i, path_label_i
                                else:
                                    # Non-AMP version
                                    path_outputs = self.model(inputs)
                                    path_scores = path_outputs['pathology_scores']
                                    
                                    if path_scores.dim() == 3:
                                        path_score_i = path_scores[:, :, path_idx]
                                        path_label_i = site_findings[:, :, path_idx] if site_findings.dim() == 3 else site_findings[:, path_idx]
                                    else:
                                        path_score_i = path_scores[:, path_idx]
                                        path_label_i = site_findings[:, path_idx]
                                    
                                    valid_mask = path_label_i >= 0
                                    if valid_mask.any():
                                        pos_weight = getattr(self.config, 'pathology_pos_weights', [2.0, 4.0, 3.0])[path_idx] if hasattr(self.config, 'pathology_pos_weights') else 2.0
                                        pos_weight_tensor = torch.tensor(pos_weight, device=self.device)
                                        
                                        path_loss = F.binary_cross_entropy_with_logits(
                                            path_score_i[valid_mask],
                                            path_label_i[valid_mask].float(),
                                            pos_weight=pos_weight_tensor
                                        )
                                        
                                        path_loss = path_loss / accumulation_steps
                                        running_losses['pathology'] += path_loss.item() * accumulation_steps
                                        path_loss.backward()
                                    
                                    del path_outputs, path_scores, path_score_i, path_label_i
                            
                            # Update optimizer at end of accumulation
                            if should_sync:
                                if self.use_amp:
                                    self.pathology_scalers[path_idx].unscale_(self.pathology_optimizers[path_idx])
                                    torch.nn.utils.clip_grad_norm_(
                                        [p for name, p in self.model_without_ddp.named_parameters()
                                         if f'pathology_modules.{path_idx}' in name and p.requires_grad],
                                        max_norm=1.0
                                    )
                                    self.pathology_scalers[path_idx].step(self.pathology_optimizers[path_idx])
                                    self.pathology_scalers[path_idx].update()
                                else:
                                    torch.nn.utils.clip_grad_norm_(
                                        [p for name, p in self.model_without_ddp.named_parameters()
                                         if f'pathology_modules.{path_idx}' in name and p.requires_grad],
                                        max_norm=1.0
                                    )
                                    self.pathology_optimizers[path_idx].step()
                        
                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                if is_main_process():
                                    logger.warning(f"OOM in pathology {path_idx} update batch {batch_idx}, skipping")
                                gc.collect()
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e
                
                # ============================================
                # 2. TB Patient Classifier Update
                # ============================================
                for param in self.model_without_ddp.parameters():
                    param.requires_grad = False
                
                # Enable gradients for patient pipeline and task classifiers
                for name, param in self.model_without_ddp.named_parameters():
                    if any(component in name for component in [
                        'site_integration', 'patient_mil', 'task_classifiers',
                        'cross_site_attention', 'tb_classifier'
                    ]):
                        param.requires_grad = True
                
                if batch_idx % accumulation_steps == 0 and self.patient_pipeline_optimizer:
                    self.patient_pipeline_optimizer.zero_grad()
                
                # Use context manager for gradient synchronization
                sync_context = self.model.no_sync if (self.is_distributed and not should_sync) else nullcontext
                
                with sync_context():
                    if self.use_amp and self.patient_pipeline_optimizer:
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(inputs)
                            total_loss, loss_dict = self.model_without_ddp.compute_losses(
                                outputs, targets, self.task_pos_weights
                            )
                            total_loss = total_loss / accumulation_steps
                            
                            if 'TB Label_loss' in loss_dict:
                                running_losses['tb_loss'] += loss_dict['TB Label_loss']
                            if 'attention_entropy' in loss_dict:
                                running_losses['attention_entropy'] += loss_dict['attention_entropy']
                            if 'attention_entropy_penalty' in loss_dict:
                                running_losses['attention_entropy_penalty'] += loss_dict['attention_entropy_penalty']
                            
                            self.patient_pipeline_scaler.scale(total_loss).backward()
                    elif self.patient_pipeline_optimizer:
                        outputs = self.model(inputs)
                        total_loss, loss_dict = self.model_without_ddp.compute_losses(
                            outputs, targets, self.task_pos_weights
                        )
                        total_loss = total_loss / accumulation_steps
                        
                        if 'TB Label_loss' in loss_dict:
                            running_losses['tb_loss'] += loss_dict['TB Label_loss']
                        if 'attention_entropy' in loss_dict:
                            running_losses['attention_entropy'] += loss_dict['attention_entropy']
                        if 'attention_entropy_penalty' in loss_dict:
                            running_losses['attention_entropy_penalty'] += loss_dict['attention_entropy_penalty']
                        
                        # Log attention score statistics
                        if is_main_process() and 'action_logits' in outputs:
                            action_logits = outputs['action_logits']
                            if action_logits is not None:
                                logit_std = action_logits.std().item()
                                logit_min = action_logits.min().item()
                                logit_max = action_logits.max().item()
                                logger.info(f"[Epoch {epoch}, Batch {batch_idx}] Attention logits - "
                                          f"std: {logit_std:.4f}, min: {logit_min:.4f}, max: {logit_max:.4f}")
                                
                                if logit_std < 0.01:
                                    logger.warning(f"⚠️  Attention logits have very low variance: {logit_std:.6f}")
                        
                        total_loss.backward()
                
                # Update optimizer at end of accumulation
                if should_sync and self.patient_pipeline_optimizer:
                    # Log gradients for frame_selector (attention mechanism)
                    if is_main_process():
                        frame_selector_grad_norm = 0.0
                        frame_selector_param_count = 0
                        for name, param in self.model_without_ddp.named_parameters():
                            if 'frame_selector' in name and param.grad is not None:
                                frame_selector_grad_norm += param.grad.norm().item() ** 2
                                frame_selector_param_count += 1
                        
                        if frame_selector_param_count > 0:
                            frame_selector_grad_norm = (frame_selector_grad_norm ** 0.5)
                            logger.info(f"[Epoch {epoch}, Batch {batch_idx}] Frame selector gradient norm: {frame_selector_grad_norm:.6f}")
                            
                            # Warning if gradients are too small
                            if frame_selector_grad_norm < 1e-6:
                                logger.warning(f"⚠️  Frame selector gradients very small: {frame_selector_grad_norm:.2e}")
                        else:
                            logger.warning(f"⚠️  No gradients found for frame_selector!")
                    
                    if self.use_amp:
                        self.patient_pipeline_scaler.unscale_(self.patient_pipeline_optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for name, p in self.model_without_ddp.named_parameters()
                             if any(component in name for component in [
                                 'site_integration', 'patient_mil', 'task_classifiers',
                                 'cross_site_attention', 'tb_classifier'
                             ]) and p.requires_grad],
                            max_norm=1.0
                        )
                        self.patient_pipeline_scaler.step(self.patient_pipeline_optimizer)
                        self.patient_pipeline_scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            [p for name, p in self.model_without_ddp.named_parameters()
                             if any(component in name for component in [
                                 'site_integration', 'patient_mil', 'task_classifiers',
                                 'cross_site_attention', 'tb_classifier'
                             ]) and p.requires_grad],
                            max_norm=1.0
                        )
                        self.patient_pipeline_optimizer.step()
                
                # Collect metrics for TB
                with torch.no_grad():
                    task_logits = outputs.get('task_logits', {})
                    
                    if 'TB Label' in task_logits:
                        logits = task_logits['TB Label']
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        
                        all_tb_targets.append(tb_labels.detach().cpu())
                        all_tb_predictions.append(preds.detach().cpu())
                        all_tb_logits.append(logits.detach().cpu())
                    
                    # Collect pathology metrics if enabled
                    if self.use_pathology_loss and 'pathology_scores' in outputs:
                        path_scores = outputs['pathology_scores']
                        path_labels = targets['pathology_labels']
                        
                        if path_scores.dim() == 3:
                            B, N, P = path_scores.shape
                            path_scores = path_scores.reshape(-1, P)
                            path_labels = path_labels.reshape(-1, P)
                        
                        valid_mask = path_labels >= 0
                        
                        pathology_scores_list.append(path_scores.detach().cpu())
                        pathology_labels_list.append(path_labels.detach().cpu())
                        pathology_masks_list.append(valid_mask.detach().cpu())
                
                # ============================================
                # 3. Backbone Update
                # ============================================
                if self.backbone_optimizer and batch_idx % 2 == 0:  # Update backbone less frequently
                    # Cleanup from previous steps
                    if 'outputs' in locals(): del outputs
                    if 'total_loss' in locals(): del total_loss
                    if 'loss_dict' in locals(): del loss_dict
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Freeze all parameters
                    for param in self.model_without_ddp.parameters():
                        param.requires_grad = False
                    
                    # Unfreeze backbone parameters
                    backbone_components = [
                        'vision_encoder', 'cnn_backbone', 'video_transformer', 'backbone',
                        'multi_feature_extraction', 'multi_scale_extraction',
                        'feature_projection', 'cnn_projection', 'output_projection', 'frame_selector'
                    ]
                    
                    for name, param in self.model_without_ddp.named_parameters():
                        if any(component in name for component in backbone_components):
                            param.requires_grad = True
                    
                    if batch_idx % accumulation_steps == 0:
                        self.backbone_optimizer.zero_grad()
                    
                    try:
                        sync_context = self.model.no_sync if (self.is_distributed and not should_sync) else nullcontext
                        
                        with sync_context():
                            if self.use_amp:
                                with torch.amp.autocast('cuda'):
                                    backbone_outputs = self.model(inputs)
                                    backbone_loss, _ = self.model_without_ddp.compute_losses(
                                        backbone_outputs, targets, self.task_pos_weights
                                    )
                                    backbone_loss = backbone_loss / accumulation_steps
                                    running_losses['total'] += backbone_loss.item() * accumulation_steps
                                    
                                    del backbone_outputs
                                    self.backbone_scaler.scale(backbone_loss).backward()
                                    del backbone_loss
                            else:
                                backbone_outputs = self.model(inputs)
                                backbone_loss, _ = self.model_without_ddp.compute_losses(
                                    backbone_outputs, targets, self.task_pos_weights
                                )
                                backbone_loss = backbone_loss / accumulation_steps
                                running_losses['total'] += backbone_loss.item() * accumulation_steps
                                
                                del backbone_outputs
                                backbone_loss.backward()
                                del backbone_loss
                        
                        # Update optimizer at end of accumulation
                        if should_sync:
                            if self.use_amp:
                                self.backbone_scaler.unscale_(self.backbone_optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    [p for p in self.model_without_ddp.parameters() if p.requires_grad],
                                    max_norm=0.5
                                )
                                self.backbone_scaler.step(self.backbone_optimizer)
                                self.backbone_scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    [p for p in self.model_without_ddp.parameters() if p.requires_grad],
                                    max_norm=0.5
                                )
                                self.backbone_optimizer.step()
                    
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            if is_main_process():
                                logger.warning(f"OOM in backbone update batch {batch_idx}, skipping")
                            if self.backbone_optimizer:
                                self.backbone_optimizer.zero_grad()
                            gc.collect()
                            torch.cuda.empty_cache()
                        else:
                            raise e
                
                # Re-enable all gradients
                for param in self.model_without_ddp.parameters():
                    param.requires_grad = True
                
                # Memory cleanup
                if batch_idx % 2 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Update progress bar (only on main process)
                if is_main_process() and batch_idx % 2 == 0:
                    def to_scalar(value):
                        if isinstance(value, torch.Tensor):
                            return value.item()
                        return value
                    
                    progress_dict = {
                        'total_loss': to_scalar(running_losses['total'] / max(1, batch_idx + 1)),
                        'tb_loss': to_scalar(running_losses['tb_loss'] / max(1, batch_idx + 1)),
                    }
                    
                    if self.use_pathology_loss and 'pathology' in running_losses:
                        progress_dict['path_loss'] = to_scalar(running_losses['pathology'] / max(1, batch_idx + 1))
                    
                    # Add attention entropy metrics to progress display
                    if running_losses['attention_entropy'] > 0:
                        progress_dict['attn_entropy'] = to_scalar(running_losses['attention_entropy'] / max(1, batch_idx + 1))
                        progress_dict['attn_penalty'] = to_scalar(running_losses['attention_entropy_penalty'] / max(1, batch_idx + 1))
                    
                    progress_bar.set_postfix(progress_dict)
            
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    if is_main_process():
                        logger.warning(f"OOM in batch {batch_idx}, skipping")
                    
                    # Reset optimizers
                    if self.backbone_optimizer:
                        self.backbone_optimizer.zero_grad()
                    if self.patient_pipeline_optimizer:
                        self.patient_pipeline_optimizer.zero_grad()
                    for opt in self.pathology_optimizers:
                        opt.zero_grad()
                    
                    # Reset scalers
                    if self.use_amp:
                        if hasattr(self, 'backbone_scaler'):
                            self.backbone_scaler = torch.amp.GradScaler('cuda')
                        self.pathology_scalers = [torch.amp.GradScaler('cuda') for _ in self.pathology_optimizers]
                        if hasattr(self, 'patient_pipeline_scaler'):
                            self.patient_pipeline_scaler = torch.amp.GradScaler('cuda')
                    
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    if is_main_process():
                        logger.error(f"Runtime error: {e}")
                    raise e
        
        progress_bar.close()
        
        # Gather metrics from all processes
        if self.is_distributed:
            # Convert losses to tensors for all_reduce
            loss_tensor = torch.tensor([
                running_losses['total'],
                running_losses['tb_loss'],
                running_losses.get('pathology', 0.0),
                len(self.train_loader)
            ], device=self.device)
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            
            running_losses['total'] = loss_tensor[0].item()
            running_losses['tb_loss'] = loss_tensor[1].item()
            if self.use_pathology_loss:
                running_losses['pathology'] = loss_tensor[2].item()
            
            num_batches = int(loss_tensor[3].item())
        else:
            num_batches = len(self.train_loader)
        
        # Calculate metrics
        all_metrics = {}

        tb_targets_np = None
        tb_preds_np = None
        tb_logits_np = None

        if all_tb_targets and all_tb_predictions and all_tb_logits:
            local_targets = torch.cat(all_tb_targets).cpu().numpy()
            local_preds   = torch.cat(all_tb_predictions).cpu().numpy()
            local_logits  = torch.cat(all_tb_logits).cpu().numpy()

            tb_targets_np = _ddp_concat_numpy(local_targets)
            tb_preds_np   = _ddp_concat_numpy(local_preds)
            tb_logits_np  = _ddp_concat_numpy(local_logits)

            tb_metrics = self._calculate_metrics(
                tb_targets_np, tb_preds_np, tb_logits_np, None, "TB Label"
            )
            for key, value in tb_metrics.items():
                all_metrics[f'TB Label_{key}'] = value

        # Pathology gather for train metrics
        if self.use_pathology_loss and pathology_labels_list and pathology_scores_list:
            local_scores = torch.cat(pathology_scores_list, dim=0).cpu().numpy()
            local_labels = torch.cat(pathology_labels_list, dim=0).cpu().numpy()
            local_masks  = torch.cat(pathology_masks_list,  dim=0).cpu().numpy()

            all_scores_np = _ddp_concat_numpy(local_scores)
            all_labels_np = _ddp_concat_numpy(local_labels)
            all_masks_np  = _ddp_concat_numpy(local_masks)

            path_metrics = self._calculate_pathology_metrics(
                torch.from_numpy(all_scores_np),
                torch.from_numpy(all_labels_np),
                torch.from_numpy(all_masks_np)
            )
            all_metrics.update(path_metrics)
        
        # Log metrics (only on main process)
        if is_main_process():
            # Log attention entropy metrics
            avg_entropy = running_losses['attention_entropy'] / max(1, num_batches)
            avg_penalty = running_losses['attention_entropy_penalty'] / max(1, num_batches)
            if avg_entropy > 0:
                logger.info(f"Train Attention metrics:")
                logger.info(f"  Entropy: {avg_entropy:.4f} | Penalty: {avg_penalty:.4f}")
            
            logger.info(f"Train TB metrics:")
            tb_metrics = {k.replace('TB Label_', ''): v for k, v in all_metrics.items()
                         if k.startswith('TB Label_')}
            if tb_metrics:
                logger.info(f"  TB Label: " + " | ".join([f"{k}: {v:.4f}" for k, v in tb_metrics.items()]))
            
            if self.use_pathology_loss:
                path_metrics = {k: v for k, v in all_metrics.items() if '/' in k}
                if path_metrics:
                    logger.info(f"Train Pathology metrics:")
                    pathology_names = getattr(self.config, 'pathology_classes',
                                            ['A-line', 'Large consolidations', 'Pleural Effusion', 'Other Pathology'])
                    for name in pathology_names:
                        name_metrics = {k.split('/')[-1]: v for k, v in path_metrics.items()
                                      if k.startswith(f'{name}/')}
                        if name_metrics:
                            logger.info(f"  {name}: " + " | ".join([f"{k}: {v:.4f}" for k, v in name_metrics.items()]))
        
        # Step all schedulers
        for scheduler in self.schedulers:
            scheduler.step()
        
        return running_losses['total'] / max(1, num_batches), all_metrics
    
    def validate(self, epoch, loader=None, split_name="val"):
        """Validation with distributed support."""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        running_loss = 0.0
        
        # TB tracking
        all_tb_targets = []
        all_tb_predictions = []
        all_tb_logits = []
        all_tb_probs = []
        
        pathology_labels_list = []
        pathology_scores_list = []
        pathology_masks_list = []
        
        progress_bar = tqdm(
            loader,
            desc=f"{split_name.capitalize()} Evaluation",
            disable=not is_main_process()
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                try:
                    # Move data to device
                    site_videos = batch['site_videos'].to(self.device)
                    site_indices = batch['site_indices'].to(self.device)
                    # Guard for missing site_masks in validation/test
                    if 'site_masks' in batch and batch['site_masks'] is not None:
                        site_masks = batch['site_masks'].to(self.device)
                    else:
                        # Create a default all-valid mask if not provided
                        if 'site_findings' in batch and batch['site_findings'] is not None:
                            sf = batch['site_findings']
                            N = sf.shape[1] if sf.ndim >= 2 else (
                                batch['site_indices'].shape[1] if 'site_indices' in batch else getattr(self.config, 'max_sites', 15)
                            )
                        elif 'site_indices' in batch and batch['site_indices'] is not None:
                            N = batch['site_indices'].shape[1]
                        else:
                            N = getattr(self.config, 'max_sites', 15)
                        site_masks = torch.ones((batch['site_videos'].shape[0], N), dtype=torch.bool, device=self.device)

                    site_findings = batch['site_findings'].to(self.device)
                    tb_labels = batch['tb_labels'].to(self.device).float()
                    pneumonia_labels = torch.full_like(tb_labels, -1)
                    covid_labels = torch.full_like(tb_labels, -1)
                    
                    inputs = {
                        'site_videos': site_videos,
                        'site_indices': site_indices,
                        'site_masks': site_masks,
                        'site_findings': site_findings,
                        'is_patient_level': True
                    }
                    
                    targets = {
                        'tb_labels': tb_labels,
                        'pneumonia_labels': pneumonia_labels,
                        'covid_labels': covid_labels,
                        'pathology_labels': site_findings,
                        'site_masks': site_masks,
                    }
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss, _ = self.model_without_ddp.compute_losses(
                        outputs, targets, self.task_pos_weights
                    )
                    
                    running_loss += loss.item()
                    
                    # Collect predictions for TB (robust extraction)
                    task_logits = outputs.get('task_logits', {})

                    logits = None
                    if isinstance(task_logits, dict) and 'TB Label' in task_logits:
                        logits = task_logits['TB Label']
                    elif 'tb_logits' in outputs:
                        logits = outputs['tb_logits']
                    elif 'logits' in outputs:
                        # Fallback for models that return a generic 'logits'
                        logits = outputs['logits']

                    if logits is not None:
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()

                        all_tb_targets.append(tb_labels.detach().cpu())
                        all_tb_predictions.append(preds.detach().cpu())
                        all_tb_logits.append(logits.detach().cpu())
                        all_tb_probs.append(probs.detach().cpu())

                    # Collect pathology metrics if enabled
                    if self.use_pathology_loss and 'pathology_scores' in outputs:
                        path_scores = outputs['pathology_scores']
                        path_labels = targets['pathology_labels']
                        
                        if path_scores.dim() == 3:
                            B, N, P = path_scores.shape
                            path_scores = path_scores.reshape(-1, P)
                            path_labels = path_labels.reshape(-1, P)
                        
                        valid_mask = path_labels >= 0
                        
                        pathology_scores_list.append(path_scores.detach().cpu())
                        pathology_labels_list.append(path_labels.detach().cpu())
                        pathology_masks_list.append(valid_mask.detach().cpu())
                    
                    # Update progress bar
                    if is_main_process():
                        progress_bar.set_postfix({
                            'loss': running_loss / (progress_bar.n + 1)
                        })
                    
                    # Clean up memory
                    del site_videos, site_indices, site_masks, site_findings, inputs
                    del tb_labels, pneumonia_labels, covid_labels, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        if is_main_process():
                            logger.warning("OOM during validation, skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        # Gather metrics from all processes
        if self.is_distributed:
            loss_tensor = torch.tensor([running_loss, len(loader)], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            running_loss = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())
        else:
            num_batches = len(loader)
        
        # Calculate validation metrics
        val_loss = running_loss / num_batches
        all_metrics = {'loss': val_loss}

        # ---- Gather TB tensors across all ranks before computing metrics ----
        tb_targets_np = None
        tb_preds_np = None
        tb_logits_np = None
        tb_probs_np = None

        if all_tb_targets and all_tb_predictions:
            local_targets = torch.cat(all_tb_targets).cpu().numpy()
            local_preds   = torch.cat(all_tb_predictions).cpu().numpy()
            local_logits  = torch.cat(all_tb_logits).cpu().numpy()
            local_probs   = torch.cat(all_tb_probs).cpu().numpy()

            tb_targets_np = _ddp_concat_numpy(local_targets)
            tb_preds_np   = _ddp_concat_numpy(local_preds)
            tb_logits_np  = _ddp_concat_numpy(local_logits)
            tb_probs_np   = _ddp_concat_numpy(local_probs)

            tb_metrics = self._calculate_metrics(
                tb_targets_np,
                tb_preds_np,
                tb_logits_np,
                tb_probs_np,
                "TB Label"
            )
            for key, value in tb_metrics.items():
                all_metrics[f'TB Label_{key}'] = value

        # ---- Gather pathology tensors across all ranks before computing metrics ----
        if self.use_pathology_loss and pathology_labels_list and pathology_scores_list:
            local_scores = torch.cat(pathology_scores_list, dim=0).cpu().numpy()
            local_labels = torch.cat(pathology_labels_list, dim=0).cpu().numpy()
            local_masks  = torch.cat(pathology_masks_list,  dim=0).cpu().numpy()

            all_scores_np = _ddp_concat_numpy(local_scores)
            all_labels_np = _ddp_concat_numpy(local_labels)
            all_masks_np  = _ddp_concat_numpy(local_masks)

            # Convert back to tensors for the existing helper
            path_metrics = self._calculate_pathology_metrics(
                torch.from_numpy(all_scores_np),
                torch.from_numpy(all_labels_np),
                torch.from_numpy(all_masks_np)
            )
            all_metrics.update(path_metrics)
        
        # Log metrics (only on main process)
        if is_main_process():
            logger.info(f"{split_name} TB metrics:")
            tb_metrics = {k.replace('TB Label_', ''): v for k, v in all_metrics.items()
                         if k.startswith('TB Label_') and '/' not in k}
            if tb_metrics:
                logger.info(f"  TB Label: " + " | ".join([f"{k}: {v:.4f}" for k, v in tb_metrics.items()]))

            if self.use_pathology_loss:
                path_metrics = {k: v for k, v in all_metrics.items() if '/' in k}
                # Macro pathology metrics block (inserted before per-class metrics)
                macro_auroc = path_metrics.get('pathology/macro_auroc')
                macro_auprc = path_metrics.get('pathology/macro_auprc')
                macro_f1 = path_metrics.get('pathology/macro_f1')
                if macro_auroc is not None or macro_auprc is not None or macro_f1 is not None:
                    logger.info(f"{split_name} Pathology metrics:")
                    parts = []
                    if macro_auroc is not None:
                        parts.append(f"auroc: {macro_auroc:.4f}")
                    if macro_auprc is not None:
                        parts.append(f"auprc: {macro_auprc:.4f}")
                    if macro_f1 is not None:
                        parts.append(f"f1: {macro_f1:.4f}")
                    logger.info("  Macro Average: " + " | ".join(parts))
                # Per-class metrics (unchanged)
                pathology_names = getattr(self.config, 'pathology_classes',
                                          ['A-line', 'Large consolidations', 'Pleural Effusion', 'Other Pathology'])
                for name in pathology_names:
                    name_metrics = {k.split('/')[-1]: v for k, v in path_metrics.items()
                                    if k.startswith(f'{name}/')}
                    if name_metrics:
                        logger.info(f"  {name}: " + " | ".join([f"{k}: {v:.4f}" for k, v in name_metrics.items()]))

            # Print confusion matrix on the full (gathered) split
            if tb_targets_np is not None and tb_preds_np is not None:
                try:
                    cm = confusion_matrix(tb_targets_np.flatten(), tb_preds_np.flatten())
                    logger.info(f"\nConfusion Matrix for TB ({split_name}):\n{cm}")
                    report = classification_report(tb_targets_np.flatten(), tb_preds_np.flatten())
                    logger.info(f"\nClassification Report for TB ({split_name}):\n{report}")
                except Exception as e:
                    logger.info(f"Could not compute confusion matrix: {e}")
        
        return val_loss, all_metrics
    
    def _calculate_metrics(self, targets, predictions, logits, probs=None, task_name=""):
        """Calculate performance metrics."""
        metrics = {}
        
        try:
            # Ensure correct shapes
            if targets.ndim == 2 and targets.shape[1] == 1:
                targets = targets.flatten()
            if predictions.ndim == 2 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
            
            # Calculate metrics
            metrics['accuracy'] = accuracy_score(targets, predictions)
            metrics['precision'] = precision_score(targets, predictions, zero_division=0)
            metrics['recall'] = recall_score(targets, predictions, zero_division=0)
            metrics['specificity'] = recall_score(1-targets, 1-predictions, zero_division=0)
            metrics['f1'] = f1_score(targets, predictions, zero_division=0)
            
            # Calculate AUC; if probs not provided, derive from logits via sigmoid
            try:
                if probs is None and logits is not None:
                    # logits is a numpy array here; apply sigmoid safely
                    probs = 1.0 / (1.0 + np.exp(-logits))
                if probs is not None:
                    if probs.ndim == 2 and probs.shape[1] == 1:
                        probs = probs.flatten()
                    metrics['auc'] = roc_auc_score(targets, probs)
                    metrics['auprc'] = average_precision_score(targets, probs)
            except ValueError:
                metrics['auc'] = 0.5
                metrics['auprc'] = 0.5
        
        except Exception as e:
            if is_main_process():
                logger.error(f"Error calculating metrics for {task_name}: {e}")
            if 'accuracy' not in metrics:
                metrics['accuracy'] = 0.0
            if self.config.eval_metric not in metrics:
                metrics[self.config.eval_metric] = 0.0
        
        return metrics
    
    def _calculate_pathology_metrics(self, scores, labels, masks):
        """Calculate metrics for each pathology class."""
        metrics = {}
        
        pathology_names = getattr(self.config, 'pathology_classes', [
            'A-line', 'Large consolidations', 'Pleural Effusion', 'Other Pathology'
        ])
        
        scores_np = scores.numpy()
        labels_np = labels.numpy()
        masks_np = masks.numpy()
        
        auroc_values = []
        auprc_values = []
        f1_values = []
        
        for i, name in enumerate(pathology_names):
            if i < scores_np.shape[1]:
                valid_indices = masks_np[:, i]
                
                if valid_indices.sum() > 0:
                    class_scores = scores_np[valid_indices, i]
                    class_labels = labels_np[valid_indices, i]
                    
                    if len(np.unique(class_labels)) < 2:
                        continue
                    
                    try:
                        class_probs = 1 / (1 + np.exp(-class_scores))
                        class_preds = (class_probs > 0.5).astype(np.float32)
                        
                        auroc = roc_auc_score(class_labels, class_probs)
                        auprc = average_precision_score(class_labels, class_probs)
                        f1 = f1_score(class_labels, class_preds, zero_division=0)
                        
                        metrics[f'{name}/auroc'] = auroc
                        metrics[f'{name}/auprc'] = auprc
                        metrics[f'{name}/f1'] = f1
                        
                        auroc_values.append(auroc)
                        auprc_values.append(auprc)
                        f1_values.append(f1)
                    
                    except Exception as e:
                        if is_main_process():
                            logger.warning(f"Error calculating metrics for {name}: {e}")
        
        # Calculate macro-average metrics
        if auroc_values:
            metrics['pathology/macro_auroc'] = np.mean(auroc_values)
        if auprc_values:
            metrics['pathology/macro_auprc'] = np.mean(auprc_values)
        if f1_values:
            metrics['pathology/macro_f1'] = np.mean(f1_values)
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save checkpoint (only on main process)."""
        if not is_main_process():
            return None
        
        save_dir = pathlib.Path(self.config.experiment_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model state dict (unwrap DDP if needed)
        model_state_dict = self.model_without_ddp.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'backbone_optimizer_state_dict': self.backbone_optimizer.state_dict() if self.backbone_optimizer else None,
            'patient_pipeline_optimizer_state_dict': self.patient_pipeline_optimizer.state_dict() if self.patient_pipeline_optimizer else None,
            'pathology_optimizers_state_dicts': [opt.state_dict() for opt in self.pathology_optimizers],
            'schedulers_state_dicts': [sched.state_dict() for sched in self.schedulers],
            'config': self.config.to_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement,
            'active_tasks': self.active_tasks,
            'use_pathology_loss': self.use_pathology_loss
        }
        
        if self.use_amp:
            checkpoint.update({
                'backbone_scaler_state_dict': self.backbone_scaler.state_dict() if hasattr(self, 'backbone_scaler') else None,
                'patient_pipeline_scaler_state_dict': self.patient_pipeline_scaler.state_dict() if hasattr(self, 'patient_pipeline_scaler') else None,
                'pathology_scalers_state_dicts': [scaler.state_dict() for scaler in self.pathology_scalers],
            })
        
        # Save latest checkpoint
        latest_path = save_dir / "checkpoint_latest.pth"
        print(f"Saving latest checkpoint to {latest_path}...")
        torch.save(checkpoint, latest_path)
        print(f"Saved latest checkpoint to {latest_path}")
        
        # Save periodic checkpoints
        eval_metric_key = f"TB Label_{self.config.eval_metric}"
        
        if epoch % 5 == 0 and eval_metric_key in metrics:
            metric_value = metrics.get(eval_metric_key, metrics.get('loss', float('nan')))
            epoch_path = save_dir / f"checkpoint_epoch_{epoch:03d}_metric_{metric_value:.4f}.pth"
            torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            metric_value = metrics.get(eval_metric_key, metrics.get('loss', float('nan')))
            best_path = save_dir / f"checkpoint_best_metric_{metric_value:.4f}.pth"
            torch.save(checkpoint, best_path)
            
            best_generic_path = save_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_generic_path)
            
            logger.info(f"New best model saved to {best_path}")
        
        return latest_path
    
    def train(self, resume_from_checkpoint=None):
        """Train the model."""
        if is_main_process():
            logger.info(f"Starting TB training with ablation model: {self.model_type}")
            logger.info(f"Active tasks: {self.active_tasks}")
            logger.info(f"Pathology loss enabled: {self.use_pathology_loss}")
        
        start_epoch = 0
        
        if resume_from_checkpoint:
            start_epoch = self.resume_training_from_checkpoint(resume_from_checkpoint)
            if is_main_process():
                logger.info(f"Resuming training from epoch {start_epoch}")
        else:
            if is_main_process():
                logger.info(f"Starting training for {self.config.num_epochs} epochs...")
            self.best_metric = float('-inf') if self.config.eval_metric_goal == 'max' else float('inf')
            self.best_epoch = 0
            self.epochs_without_improvement = 0
        
        for epoch in range(start_epoch, self.config.num_epochs):
            if is_main_process():
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            train_loss, train_metrics = self.train_epoch(epoch)
            
            val_loss, val_metrics = self.validate(epoch)

            # Log concise epoch summary with key metrics
            if is_main_process():
                eval_metric_key = f"TB Label_{self.config.eval_metric}"
                summary_auc = val_metrics.get(eval_metric_key)
                summary_f1 = val_metrics.get('TB Label_f1')
                summary_acc = val_metrics.get('TB Label_accuracy')
                logger.info(
                    "Epoch %d Summary — val_loss: %.4f%s%s%s" % (
                        epoch + 1,
                        val_loss,
                        f", {eval_metric_key}: {summary_auc:.4f}" if summary_auc is not None else "",
                        f", TB Label f1: {summary_f1:.4f}" if summary_f1 is not None else "",
                        f", TB Label acc: {summary_acc:.4f}" if summary_acc is not None else ""
                    )
                )
            
            # Use TB Label metric as primary
            eval_metric_key = f"TB Label_{self.config.eval_metric}"
            current_metric = val_metrics.get(eval_metric_key, val_loss)
            is_best = False
            
            if self.config.eval_metric_goal == 'max':
                if current_metric > self.best_metric:
                    is_best = True
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    if is_main_process():
                        logger.info(f"New best model with {eval_metric_key}: {current_metric:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    if is_main_process():
                        logger.info(f"No improvement. Best {eval_metric_key}: {self.best_metric:.4f} from epoch {self.best_epoch+1}")
            else:
                if current_metric < self.best_metric:
                    is_best = True
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    if is_main_process():
                        logger.info(f"New best model with {eval_metric_key}: {current_metric:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    if is_main_process():
                        logger.info(f"No improvement. Best {eval_metric_key}: {self.best_metric:.4f} from epoch {self.best_epoch+1}")
            
            # Save checkpoint (only on main process)
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Synchronize all processes
            if self.is_distributed:
                dist.barrier()
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                if is_main_process():
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        if is_main_process():
            logger.info(f"Training completed. Best model from epoch {self.best_epoch+1} with {eval_metric_key}: {self.best_metric:.4f}")
        
        # Evaluate the best model if requested
        # IMPORTANT: Run evaluation on ALL ranks to avoid hanging collectives
        if self.config.evaluate_best_valid_model:
            if self.is_distributed:
                # Ensure all ranks finished training and checkpoints are visible
                dist.barrier()
            self._evaluate_best_model()
            if self.is_distributed:
                # Ensure all ranks complete evaluation before teardown
                dist.barrier()
        
        return self.best_metric, self.best_epoch
    
    def _evaluate_best_model(self):
        """Evaluate the best model (simplified version - implement full version as needed)."""
        logger.info("Evaluating best TB ablation model on all splits...")
        
        # Clear GPU cache before evaluation to avoid OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load best model
        best_model_path = os.path.join(self.config.experiment_dir, "checkpoint_best.pth")
        if not os.path.exists(best_model_path):
            logger.warning(f"Best model checkpoint not found at {best_model_path}. Skipping evaluation.")
            return
        
        try:
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_model_path}")
        except Exception as e:
            logger.error(f"Failed to load best model: {e}")
            return
        
        # Evaluate on each split
        self.model.eval()
        
        for split_name, loader in [('test', self.test_loader), ('val', self.val_loader), ('train', self.train_loader)]:
            logger.info(f"Evaluating on {split_name} set...")
            val_loss, metrics = self.validate(0, loader, split_name)
            if is_main_process():
                eval_metric_key = f"TB Label_{self.config.eval_metric}"
                summary_auc = metrics.get(eval_metric_key)
                summary_f1 = metrics.get('TB Label_f1')
                summary_acc = metrics.get('TB Label_accuracy')
                logger.info(
                    f"{split_name.capitalize()} Summary — loss: {val_loss:.4f}"
                    + (f", {eval_metric_key}: {summary_auc:.4f}" if summary_auc is not None else "")
                    + (f", TB Label f1: {summary_f1:.4f}" if summary_f1 is not None else "")
                    + (f", TB Label acc: {summary_acc:.4f}" if summary_acc is not None else "")
                )
            
            # Clear GPU cache between splits to avoid OOM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def resume_training_from_checkpoint(self, checkpoint_path):
        """Resume training from a checkpoint."""
        if is_main_process():
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer states
        if checkpoint.get('backbone_optimizer_state_dict') and self.backbone_optimizer:
            self.backbone_optimizer.load_state_dict(checkpoint['backbone_optimizer_state_dict'])
        
        if checkpoint.get('patient_pipeline_optimizer_state_dict') and self.patient_pipeline_optimizer:
            self.patient_pipeline_optimizer.load_state_dict(checkpoint['patient_pipeline_optimizer_state_dict'])
        
        if checkpoint.get('pathology_optimizers_state_dicts'):
            for opt, state_dict in zip(self.pathology_optimizers, checkpoint['pathology_optimizers_state_dicts']):
                opt.load_state_dict(state_dict)
        
        # Load scheduler states
        if checkpoint.get('schedulers_state_dicts'):
            for scheduler, state_dict in zip(self.schedulers, checkpoint['schedulers_state_dicts']):
                scheduler.load_state_dict(state_dict)
        
        # Load AMP scaler states
        if self.use_amp:
            if checkpoint.get('backbone_scaler_state_dict') and hasattr(self, 'backbone_scaler'):
                self.backbone_scaler.load_state_dict(checkpoint['backbone_scaler_state_dict'])
            
            if checkpoint.get('patient_pipeline_scaler_state_dict') and hasattr(self, 'patient_pipeline_scaler'):
                self.patient_pipeline_scaler.load_state_dict(checkpoint['patient_pipeline_scaler_state_dict'])
            
            if checkpoint.get('pathology_scalers_state_dicts'):
                for scaler, state_dict in zip(self.pathology_scalers, checkpoint['pathology_scalers_state_dicts']):
                    scaler.load_state_dict(state_dict)
        
        # Load training state
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        if is_main_process():
            logger.info(f"Training will resume from epoch {start_epoch}")
            logger.info(f"Best metric so far: {self.best_metric:.4f} (epoch {self.best_epoch + 1})")
        
        return start_epoch


# ============================================================================
# Main Function
# ============================================================================

def parse_args_and_load_config():
    """Parse command line arguments and load configuration."""
    parser = argparse.ArgumentParser(description='Distributed HMV-MIL / ablation training.')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument(
        '--model_type', type=str, help='Ablation model type',
        choices=[
            'no_rl', 'mean_pool', 'attention_pool', 'single_task',
            '3d_cnn', 'cnn_lstm', 'video_transformer', 'Inception', 'R2+1d',
        ],
    )
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--video_folder', type=str, help='Path to video folder')
    parser.add_argument('--output_dir', type=str, help='Override outputs root directory')
    parser.add_argument('--model_weights', type=str, help='Path to model weights')
    parser.add_argument('--best_model_path', type=str, help='Path to best model for evaluation')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--train', action='store_true', default=True, help='Train mode')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only mode')
    parser.add_argument('--debug', action='store_true', help='Verbose per-batch logging')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(config_file=args.config)

    if args.model_type is not None:
        config.model_type = args.model_type
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.seed is not None:
        config.seed = args.seed
    if args.video_folder is not None:
        config.video_folder = args.video_folder
    if args.model_weights is not None:
        config.model_weights = args.model_weights
    if args.best_model_path is not None:
        config.best_model_path = args.best_model_path
    if args.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    if args.output_dir is not None:
        exp_leaf = os.path.basename(str(config.experiment_dir).rstrip('/'))
        config.experiment_dir = os.path.join(args.output_dir, 'ablation_results', exp_leaf)
        config.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        config.log_dir = os.path.join(args.output_dir, 'logs')
        config.save_dir = os.path.join(args.output_dir, 'saves')
        config.pred_save_dir = os.path.join(args.output_dir, 'predictions')
    if args.eval_only:
        config.train = False
    config.debug = args.debug
    return config


def main():
    """Main training function with distributed support."""
    rank, world_size, local_rank = setup_distributed()

    if is_main_process():
        logger.info("Working directory: %s", os.getcwd())

    config = parse_args_and_load_config()
    config.rank = rank
    config.world_size = world_size
    config.local_rank = local_rank
    config.distributed = world_size > 1

    if is_main_process():
        logger.info("Experiment directory: %s", os.path.abspath(config.experiment_dir))
        ensure_output_dirs(config)
        config_path = os.path.join(config.experiment_dir, "config.yaml")
        config.save(config_path)
        logger.info("Configuration saved to %s", config_path)
    
    # Wait for main process to create directories
    if config.distributed:
        dist.barrier()
    
    # GPU/Device information
    if torch.cuda.is_available() and is_main_process():
        logger.info(f"Using {world_size} GPU(s)")
        logger.info(f"Rank {rank}, Local Rank {local_rank}")
        logger.info(f"Device name: {torch.cuda.get_device_name(local_rank)}")
    
    try:
        # Initialize trainer with distributed parameters
        trainer = AblationTrainer(config, rank, world_size, local_rank)
        
        # Check for resume checkpoint
        resume_checkpoint = None
        if hasattr(config, 'resume_from_checkpoint') and config.resume_from_checkpoint is not None:
            if not os.path.exists(config.resume_from_checkpoint):
                if is_main_process():
                    logger.error(f"Resume checkpoint not found: {config.resume_from_checkpoint}")
                cleanup_distributed()
                return
            resume_checkpoint = config.resume_from_checkpoint
        
        # Check if we're in evaluation-only mode
        if not config.train:
            if is_main_process():
                logger.info("Running in evaluation-only mode")
            # Run evaluation on ALL ranks to keep collectives symmetric
            if trainer.is_distributed:
                dist.barrier()
            trainer._evaluate_best_model()
            if trainer.is_distributed:
                dist.barrier()
            cleanup_distributed()
            return
        
        # Start training
        if is_main_process():
            logger.info(f"Starting TB training with ablation model: {config.model_type}...")
        
        best_metric, best_epoch = trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        if is_main_process():
            logger.info(f"Training complete! Best metric: {best_metric:.4f} at epoch {best_epoch+1}")
        
        cleanup_distributed()
        return best_metric, best_epoch
    
    except Exception as e:
        logger.exception(f"Error in training: {e}")
        cleanup_distributed()
        raise

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            cleanup_distributed()
        except Exception:
            pass
    
