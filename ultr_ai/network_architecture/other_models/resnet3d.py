import torch
import torch.nn as nn
import logging
import torchvision.models.video as video_models
import gc

# Import base components from original model
from ultr_ai.network_architecture.components.general_components import (
    PathologyModule, SiteIntegrationModule, DeepAttentionMIL
)
from ultr_ai.network_architecture.components import MultiTaskModel

logger = logging.getLogger(__name__)



class ResNet3DMultiTaskModel(nn.Module):
    """Memory-efficient ResNet3D backbone from torchvision."""
    
    def __init__(self, config):
        super().__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = getattr(config, 'num_classes', 1)
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.num_pathologies = getattr(config, 'num_pathologies', 4)
        self.num_sites = getattr(config, 'num_sites', 21)
        self.device = getattr(config, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.active_tasks = getattr(config, 'active_tasks', ['TB Label'])
        self.use_pathology_loss = getattr(config, 'use_pathology_loss', True)
        self.task_weights = getattr(config, 'task_weights', {'TB Label': 1.0})
        self.selection_strategy = '3d_resnet'
        
        # Memory optimization settings
        self.backbone_frozen = getattr(config, 'backbone_frozen', False)  # NEW: Keep backbone frozen by default
        self.max_sites_per_forward = getattr(config, 'max_sites_per_forward', 4)  # NEW: Process sites in chunks
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', True)  # NEW
        
        logger.info("Using Memory-Efficient ResNet3D backbone")
        
        # Use smaller ResNet3D model for better memory efficiency
        # try:
        #     # Use R(2+1)D ResNet18 - more memory efficient than 3D ResNet
        #     self.backbone = video_models.r2plus1d_18(weights=video_models.R2Plus1D_18_Weights.DEFAULT)
        #     logger.info("Loaded R(2+1)D ResNet18")
        # except:
        #     try:
        #         # Fallback to MC3 ResNet18
        #         self.backbone = video_models.mc3_18(weights=video_models.MC3_18_Weights.DEFAULT) 
        #         logger.info("Loaded MC3 ResNet18")
        #     except:
        #         # Final fallback
        #         self.backbone = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
        #         logger.info("Loaded R3D ResNet18")
        
        self.backbone = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
        logger.info("Loaded R3D ResNet18")
        
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Enable gradient checkpointing if available and requested
        if self.use_gradient_checkpointing and hasattr(self.backbone, 'gradient_checkpointing'):
            self.backbone.gradient_checkpointing = True
            logger.info("Enabled gradient checkpointing on backbone")
        
        # Freeze backbone by default for memory efficiency
        # if self.backbone_frozen:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False
        #     logger.info("Backbone frozen for memory efficiency")
        
        # Get feature dimension
        backbone_dim = 512
        
        projection_dim = min(self.hidden_dim, 512)  
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        if projection_dim != self.hidden_dim:
            self.feature_upscale = nn.Linear(projection_dim, self.hidden_dim)
        else:
            self.feature_upscale = nn.Identity()
        
        # Pathology modules 
        if self.use_pathology_loss:
            pathology_hidden = min(self.hidden_dim // 2, 256)  
            self.pathology_modules = nn.ModuleList([
                PathologyModule(
                    feature_dim=self.hidden_dim,
                    hidden_dim=pathology_hidden,
                    dropout=self.dropout_rate,
                    name=f'pathology_{i}'
                ) for i in range(self.num_pathologies)
            ])
        else:
            self.pathology_modules = None
        
        # Site integration - smaller embedding
        if self.use_pathology_loss:
            self.site_integration = SiteIntegrationModule(
                feature_dim=self.hidden_dim,
                site_embed_dim=256, 
                hidden_dim=self.hidden_dim,
                num_sites=self.num_sites,
                num_pathologies=self.num_pathologies,
                dropout=self.dropout_rate
            )
        else:
            self.site_integration = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            )
        
        # Patient-level MIL - smaller hidden dim
        mil_hidden = min(self.hidden_dim // 2, 512)  
        self.patient_mil = DeepAttentionMIL(
            feature_dim=self.hidden_dim,
            hidden_dim=mil_hidden,
            dropout=self.dropout_rate,
            num_heads=8  
        )
        
        # Task classifiers - smaller hidden layers
        classifier_hidden = min(self.hidden_dim // 2, 512)  
        self.task_classifiers = nn.ModuleDict()
        for task_name in self.active_tasks:
            task_key = task_name.replace(' ', '_').replace('Label', 'label')
            self.task_classifiers[task_key] = nn.Sequential(
                nn.Linear(self.hidden_dim, classifier_hidden),
                nn.LayerNorm(classifier_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(classifier_hidden, self.num_classes)
            )
        
        # TB classifier for backward compatibility
        self.tb_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(classifier_hidden, self.num_classes)
        )
        
        # Dummy frame selector for compatibility
        self.frame_selector = self._create_dummy_selector()
        
        # Report memory optimizations
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    
    def _create_dummy_selector(self):
        """Create dummy frame selector for compatibility."""
        class DummySelector:
            def __init__(self):
                self.saved_actions = []
                self.temperature = 1.0
            
            def get_temperature(self):
                return self.temperature
            
            def clear_history(self):
                self.saved_actions = []
            
            def reset_rewards(self):
                pass
            
            def reset_temperature(self):
                return self.temperature
            
            def update_temperature(self, **kwargs):
                return self.temperature
        
        return DummySelector()
    
    def _extract_video_features(self, video, use_amp=True):
        """Memory-efficient video feature extraction."""
        # Reshape for ResNet3D: [1, C, T, H, W]
        video_input = video.permute(1, 0, 2, 3).unsqueeze(0)
        
        # Use appropriate context for feature extraction
        context = torch.amp.autocast('cuda') if use_amp else torch.enable_grad()
        
        if self.backbone_frozen:
            # Extract features without gradients to save memory
            with torch.no_grad(), context:
                video_features = self.backbone(video_input)
        else:
            # Extract with gradients but use checkpointing
            with context:
                if self.use_gradient_checkpointing and hasattr(self.backbone, 'gradient_checkpointing'):
                    video_features = torch.utils.checkpoint.checkpoint(self.backbone, video_input)
                else:
                    video_features = self.backbone(video_input)
        
        return video_features
    
    def _process_sites_chunked(self, site_videos, site_masks, batch_idx, use_amp=True):
        """Process sites in memory-efficient chunks."""
        batch_size, max_sites = site_videos.shape[0], site_videos.shape[1]
        
        all_site_features = []
        all_pathology_scores = []
        
        # Process each sample in batch
        for b in range(batch_size):
            valid_sites = site_masks[b].sum().item()
            
            if valid_sites == 0:
                # Handle empty case
                site_features = torch.zeros(max_sites, self.hidden_dim, device=site_videos.device)
                pathology_scores = torch.zeros(max_sites, self.num_pathologies, device=site_videos.device)
                all_site_features.append(site_features)
                all_pathology_scores.append(pathology_scores)
                continue
            
            sample_features = []
            sample_pathology_scores = []
            
            # Process sites in chunks to avoid OOM
            for start_idx in range(0, valid_sites, self.max_sites_per_forward):
                end_idx = min(start_idx + self.max_sites_per_forward, valid_sites)
                chunk_size = end_idx - start_idx
                
                chunk_features = []
                chunk_pathology_scores = []
                
                # Process each site in the chunk
                for n in range(start_idx, end_idx):
                    video = site_videos[b, n]  # [T, C, H, W]
                    
                    try:
                        # Extract features with memory-efficient method
                        video_features = self._extract_video_features(video, use_amp)
                        
                        # Project features
                        projected_features = self.feature_projection(video_features)
                        final_features = self.feature_upscale(projected_features)
                        
                        chunk_features.append(final_features)
                        
                        # Process pathology if enabled
                        if self.use_pathology_loss and self.pathology_modules is not None:
                            # Create dummy representation for pathology modules
                            dummy_frames = final_features.unsqueeze(1).repeat(1, 3, 1)
                            dummy_mask = torch.ones(1, 3, dtype=torch.bool, device=video.device)
                            
                            pathology_scores = []
                            for module in self.pathology_modules:
                                score, _, _ = module(dummy_frames, dummy_mask)
                                pathology_scores.append(score)
                            
                            chunk_pathology_scores.append(torch.cat(pathology_scores, dim=1))
                        
                        # Clear intermediate variables
                        del video_features, projected_features, final_features
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            logger.warning(f"OOM processing site {n} in batch {batch_idx}, using zero features")
                            # Use zero features as fallback
                            zero_features = torch.zeros(1, self.hidden_dim, device=site_videos.device)
                            chunk_features.append(zero_features)
                            
                            if self.use_pathology_loss:
                                zero_pathology = torch.zeros(1, self.num_pathologies, device=site_videos.device)
                                chunk_pathology_scores.append(zero_pathology)
                            
                            # Emergency cleanup
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            raise e
                
                # Collect chunk results
                if chunk_features:
                    sample_features.extend(chunk_features)
                if chunk_pathology_scores:
                    sample_pathology_scores.extend(chunk_pathology_scores)
            
            # Pad and collect sample results
            if sample_features:
                sample_features_tensor = torch.cat(sample_features, dim=0)
                padded_features = torch.zeros(max_sites, self.hidden_dim, device=site_videos.device)
                padded_features[:valid_sites] = sample_features_tensor
                all_site_features.append(padded_features)
                
                if self.use_pathology_loss and sample_pathology_scores:
                    sample_pathology_tensor = torch.cat(sample_pathology_scores, dim=0)
                    padded_scores = torch.zeros(max_sites, self.num_pathologies, device=site_videos.device)
                    padded_scores[:valid_sites] = sample_pathology_tensor
                    all_pathology_scores.append(padded_scores)
                else:
                    all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
            else:
                all_site_features.append(torch.zeros(max_sites, self.hidden_dim, device=site_videos.device))
                all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
        
        return all_site_features, all_pathology_scores
    
    def forward(self, inputs):
        """Memory-efficient forward pass."""
        site_videos = inputs['site_videos']
        site_indices = inputs['site_indices']
        site_masks = inputs['site_masks']
        
        batch_size, max_sites = site_videos.shape[0], site_videos.shape[1]
        
        
        # Use chunked processing for memory efficiency
        try:
            use_amp = hasattr(self, 'use_amp') and getattr(self, 'use_amp', True)
            all_site_features, all_pathology_scores = self._process_sites_chunked(
                site_videos, site_masks, batch_idx=0, use_amp=use_amp
            )
            
            # Stack features
            site_features = torch.stack(all_site_features)
            pathology_scores = torch.stack(all_pathology_scores) if self.use_pathology_loss else None
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.error("OOM during chunked processing, falling back to minimal processing")
                # Emergency fallback: use zero features
                site_features = torch.zeros(batch_size, max_sites, self.hidden_dim, device=site_videos.device)
                pathology_scores = torch.zeros(batch_size, max_sites, self.num_pathologies, device=site_videos.device) if self.use_pathology_loss else None
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
        
        # Site integration
        try:
            if self.use_pathology_loss:
                integrated_features = self.site_integration(site_features, site_indices, pathology_scores)
            else:
                integrated_features = self.site_integration(site_features)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning("OOM in site integration, using passthrough")
                integrated_features = site_features
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
        
        # Patient-level MIL
        try:
            patient_features, mil_attention = self.patient_mil(integrated_features, site_masks)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning("OOM in patient MIL, using mean pooling")
                # Fallback to simple mean pooling
                valid_features = integrated_features * site_masks.unsqueeze(-1).float()
                patient_features = valid_features.sum(dim=1) / site_masks.sum(dim=1, keepdim=True).float().clamp(min=1)
                mil_attention = torch.ones_like(site_masks).float() / site_masks.sum(dim=1, keepdim=True).float().clamp(min=1)
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
        
        # Classification
        task_logits = {}
        for task_name in self.active_tasks:
            if task_name == 'TB Label':
                try:
                    tb_logits = self.tb_classifier(patient_features)
                    if self.num_classes == 1:
                        tb_logits = tb_logits.squeeze(-1)
                    task_logits['TB Label'] = tb_logits
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        logger.warning("OOM in TB classifier, using zeros")
                        task_logits['TB Label'] = torch.zeros(batch_size, device=site_videos.device)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise e
        
        return {
            'task_logits': task_logits,
            'pathology_scores': pathology_scores,
            'patient_features': patient_features,
            'mil_attention': mil_attention,
            'site_features': site_features,
            'site_rl_data': []
        }
    
    def compute_losses(self, outputs, targets, pos_weights=None):
        """Reuse loss computation from MultiTaskModel."""
        return MultiTaskModel.compute_losses(self, outputs, targets, pos_weights)
    
    def unfreeze_backbone_gradually(self, epoch, total_epochs):
        """Gradually unfreeze backbone layers as training progresses."""
        if not self.backbone_frozen:
            return
            
            
        layers_to_unfreeze = [ 'layer1','layer2', 'layer3', 'layer4', 'avgpool', 'fc'], # 75% through
        

        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True
                logger.debug(f"Unfroze: {name}")

    
    def get_memory_usage(self):
        """Get current memory usage statistics."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / (1024**3)
            max_memory = torch.cuda.max_memory_allocated() / (1024**3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            return {
                'current_gb': current_memory,
                'max_gb': max_memory, 
                'total_gb': total_memory,
                'usage_percent': (current_memory / total_memory) * 100
            }
        return {'current_gb': 0, 'max_gb': 0, 'total_gb': 0, 'usage_percent': 0}


# Aliases for backward compatibility
CNN3DMultiTaskModel = ResNet3DMultiTaskModel  # Use ResNet3D instead