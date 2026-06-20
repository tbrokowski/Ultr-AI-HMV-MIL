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


class InceptionMultiTaskModel(nn.Module):
    """Ablation: Inception3D (I3D) backbone for video understanding."""
    
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
        self.selection_strategy = 'inception'
        
        # Memory optimization settings
        self.backbone_frozen = getattr(config, 'backbone_frozen', True)
        self.max_sites_per_forward = getattr(config, 'max_sites_per_forward', 4)
        
        logger.info("Using Inception3D (S3D-G) backbone")
        
        # Try to load S3D or similar Inception-based model
        try:
            self.backbone = video_models.s3d(weights=video_models.S3D_Weights.DEFAULT)
            logger.info("Loaded S3D (Inception-based) model")
            backbone_dim = 400
        except:
            # Fallback to MViT or other available model
            try:
                self.backbone = video_models.mvit_v1_b(weights=video_models.MViT_V1_B_Weights.DEFAULT)
                logger.info("Loaded MViT (fallback from Inception)")
                backbone_dim = 768
            except:
                # Final fallback to MC3
                self.backbone = video_models.mc3_18(weights=video_models.MC3_18_Weights.DEFAULT)
                logger.info("Loaded MC3 (fallback from Inception)")
                backbone_dim = 512
        
        # Remove final classification layer
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        
        # Freeze backbone if specified
        if self.backbone_frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Inception backbone frozen for memory efficiency")
        
        # Feature projection
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
        
        # Site integration
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
        
        # Patient-level MIL
        mil_hidden = min(self.hidden_dim // 2, 512)
        self.patient_mil = DeepAttentionMIL(
            feature_dim=self.hidden_dim,
            hidden_dim=mil_hidden,
            dropout=self.dropout_rate,
            num_heads=4
        )
        
        # Task classifiers
        classifier_hidden = min(self.hidden_dim // 2, 256)
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
        
        self.tb_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(classifier_hidden, self.num_classes)
        )
        
        # Dummy frame selector for compatibility
        self.frame_selector = self._create_dummy_selector()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Inception - Total parameters: {total_params:,}")
        logger.info(f"Inception - Trainable parameters: {trainable_params:,}")
    
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
        """Extract features from video using Inception."""
        # Reshape for Inception: [1, C, T, H, W]
        video_input = video.permute(1, 0, 2, 3).unsqueeze(0)
        
        context = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp)

        if self.backbone_frozen:
            with torch.no_grad(), context:
                video_features = self.backbone(video_input)
        else:
            with context:
                video_features = self.backbone(video_input)
        proj_dtype = next(self.feature_projection.parameters()).dtype
        
        return video_features.to(proj_dtype)
    
    def _zeros_like_proj(self, *shape, device):
        dtype = next(self.feature_projection.parameters()).dtype
        return torch.zeros(*shape, device=device, dtype=dtype)

    def forward(self, inputs):
        """Forward pass using Inception backbone."""
        site_videos = inputs['site_videos']
        site_indices = inputs['site_indices']
        site_masks = inputs['site_masks']
        
        batch_size, max_sites = site_videos.shape[0], site_videos.shape[1]
        
        all_site_features = []
        all_pathology_scores = []
        
        # Process each sample in batch
        for b in range(batch_size):
            valid_sites = site_masks[b].sum().item()
            
            if valid_sites == 0:
                site_features = self._zeros_like_proj(max_sites, self.hidden_dim, device=site_videos.device)
                pathology_scores = self._zeros_like_proj(max_sites, self.num_pathologies, device=site_videos.device)
                all_site_features.append(site_features)
                all_pathology_scores.append(pathology_scores)
                continue
            
            sample_features = []
            sample_pathology_scores = []
            
            for n in range(valid_sites):
                video = site_videos[b, n]
                
                try:
                    video_features = self._extract_video_features(video)
                    projected_features = self.feature_projection(video_features)
                    final_features = self.feature_upscale(projected_features)
                    
                    sample_features.append(final_features)
                    
                    if self.use_pathology_loss and self.pathology_modules is not None:
                        dummy_frames = final_features.unsqueeze(1).repeat(1, 3, 1)
                        dummy_mask = torch.ones(1, 3, dtype=torch.bool, device=video.device)
                        
                        pathology_scores = []
                        for module in self.pathology_modules:
                            score, _, _ = module(dummy_frames, dummy_mask)
                            pathology_scores.append(score)
                        
                        sample_pathology_scores.append(torch.cat(pathology_scores, dim=1))
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        logger.warning(f"OOM processing site {n}, using zero features")
                        zero_features = self._zeros_like_proj(1, self.hidden_dim, device=site_videos.device)
                        sample_features.append(zero_features)
                        
                        if self.use_pathology_loss:
                            zero_pathology = self._zeros_like_proj(1, self.num_pathologies, device=site_videos.device)
                            sample_pathology_scores.append(zero_pathology)
                        
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise e
            
            # Pad and collect results
            if sample_features:
                sample_features_tensor = torch.cat(sample_features, dim=0)
                padded_features = self._zeros_like_proj(max_sites, self.hidden_dim, device=site_videos.device)
                padded_features[:valid_sites] = sample_features_tensor
                all_site_features.append(padded_features)
                
                if self.use_pathology_loss and sample_pathology_scores:
                    sample_pathology_tensor = torch.cat(sample_pathology_scores, dim=0)
                    padded_scores = self._zeros_like_proj(max_sites, self.num_pathologies, device=site_videos.device)
                    padded_scores[:valid_sites] = sample_pathology_tensor
                    all_pathology_scores.append(padded_scores)
                else:
                    all_pathology_scores.append(self._zeros_like_proj(max_sites, self.num_pathologies, device=site_videos.device))
            else:
                all_site_features.append(self._zeros_like_proj(max_sites, self.hidden_dim, device=site_videos.device))
                all_pathology_scores.append(self._zeros_like_proj(max_sites, self.num_pathologies, device=site_videos.device))

        site_features = torch.stack(all_site_features)
        pathology_scores = torch.stack(all_pathology_scores) if self.use_pathology_loss else None
        
        # Site integration
        if self.use_pathology_loss:
            integrated_features = self.site_integration(site_features, site_indices, pathology_scores)
        else:
            integrated_features = self.site_integration(site_features)
        
        # Patient-level MIL
        patient_features, mil_attention = self.patient_mil(integrated_features, site_masks)
        
        # Classification
        task_logits = {}
        for task_name in self.active_tasks:
            if task_name == 'TB Label':
                tb_logits = self.tb_classifier(patient_features)
                if self.num_classes == 1:
                    tb_logits = tb_logits.squeeze(-1)
                task_logits['TB Label'] = tb_logits
        
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