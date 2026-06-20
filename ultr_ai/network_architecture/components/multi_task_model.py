import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

from ultr_ai.network_architecture.components.general_components import (
    PathologyModule, SiteIntegrationModule, DeepAttentionMIL
)

logger = logging.getLogger(__name__)

# Optional timm support
try:
    import timm
    _HAS_TIMM = True
except Exception:
    timm = None
    _HAS_TIMM = False


class MultiTaskModel(nn.Module):
    """Hierarchical multi-view MIL for lung ultrasound TB classification."""

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
        
        # For compatibility with new training system
        self.active_tasks = getattr(config, 'active_tasks', ['TB Label'])
        self.use_pathology_loss = getattr(config, 'use_pathology_loss', True)
        self.task_weights = getattr(config, 'task_weights', {'TB Label': 1.0})
        self.selection_strategy = getattr(config, 'selection_strategy', 'attention_pool')
        self.use_patient_mil = getattr(config, 'use_patient_mil', True)
        
        logger.info(f"MultiTaskModel configured for tasks: {self.active_tasks}")
        logger.info(f"Using pathology loss: {self.use_pathology_loss}")
        logger.info(f"Frame selection strategy: {self.selection_strategy}")
        
        # Log attention parameters for debugging
        attention_temp = getattr(config, 'attention_temperature', 0.5)
        entropy_w = getattr(config, 'entropy_weight', 0.001)
        logger.info(f"Attention temperature: {attention_temp} (lower=sharper, higher=softer)")
        logger.info(f"Entropy regularization weight: {entropy_w}")
        
        # Vision Backbone (mobile-friendly options supported)
        self.backbone = getattr(config, 'backbone', 'clip')  # e.g., 'clip' or a timm model name like 'mobilenetv3_large_100'
        self.backbone_model_name = getattr(config, 'backbone_model_name', 'openai/clip-vit-base-patch32')
        self.freeze_backbone = getattr(config, 'freeze_backbone', False)
        self.pretrained = getattr(config, 'pretrained', True)
        self.backbone_image_size = getattr(config, 'backbone_image_size', 224)

        self._init_vision_backbone()
        self.feature_noise_std = 0.05
        self._freeze_backbone(freeze=self.freeze_backbone)
        self.frame_selector = None
        
        # Pathology modules
        self.pathology_names = [
            'a_lines',
            'large_consolidation',
            'pleural_effusion',
            'other_pathology'
        ]
        
        if self.use_pathology_loss:
            self.pathology_modules = nn.ModuleList([
                PathologyModule(
                    feature_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim // 2,
                    dropout=self.dropout_rate,
                    name=name
                ) for name in self.pathology_names
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
            # Simple site integration without pathology
            self.site_integration = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            )
        
        # Patient-level aggregation
        if self.use_patient_mil:
            self.patient_mil = DeepAttentionMIL(
                feature_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim // 2,
                dropout=self.dropout_rate,
                num_heads=8,
            )
        else:
            self.patient_mil = None
        
        # Task classifiers - using ModuleDict for compatibility
        self.task_classifiers = nn.ModuleDict()
        for task_name in self.active_tasks:
            task_key = task_name.replace(' ', '_').replace('Label', 'label')
            self.task_classifiers[task_key] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.num_classes)
            )
        
        # Keep the original TB classifier for backward compatibility
        self.tb_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )

    def _init_vision_backbone(self):
        """
        Initialize a vision backbone.
        Supported:
          - 'clip' (uses HuggingFace CLIPVisionModel with pooler_output)
          - Any timm model name (e.g., 'mobilenetv3_large_100', 'efficientnet_lite0', 'ghostnetv2', 'levit_256', 'deit_tiny_distilled_patch16_224')
        Sets:
          - self.vision_encoder: callable/module that returns a tensor [B*T, D] or feature map
          - self.vision_dim: output feature dimension D
          - self._vision_kind: 'clip', 'timm_features', or 'timm_pooled'
          - self.vision_pool / self.vision_proj if needed (for timm)
        """
        # Default outputs
        self._vision_kind = 'clip'
        # Use CLIP only when backbone explicitly requests 'clip'
        if str(self.backbone).lower() == 'clip':
            from transformers import CLIPVisionModel  # local import to avoid hard dep if using timm-only
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                self.backbone_model_name,
                dtype=torch.float32
            )
            # CLIP ViT-B/32 has 768-d pooler_output
            self.vision_dim = getattr(self.vision_encoder.config, 'hidden_size', 768)
            self._vision_kind = 'clip'
            local_weights_path = os.path.join(
                getattr(self.config, 'local_weights_dir', './CLIP_weights'),
                'model.safetensors'
            )
            if os.path.exists(local_weights_path):
                logger.info(f"Loading CLIP weights from {local_weights_path}")
                try:
                    from safetensors import safe_open as _safe_open
                    with _safe_open(local_weights_path, framework='pt', device='cpu') as f:
                        vision_state_dict = {}
                        model_state_dict = self.vision_encoder.state_dict()
                        matched_keys = 0
                        for key in model_state_dict.keys():
                            safetensors_key = f"vision_model.{key}"
                            if safetensors_key in f.keys():
                                tensor = f.get_tensor(safetensors_key)
                                if tensor.shape == model_state_dict[key].shape:
                                    vision_state_dict[key] = tensor
                                    matched_keys += 1
                        if matched_keys > 0:
                            logger.info(f"Successfully matched {matched_keys}/{len(model_state_dict)} CLIP weights")
                            self.vision_encoder.load_state_dict(vision_state_dict, strict=False)
                        else:
                            logger.info("No weights could be matched from the safetensors file")
                except Exception as e:
                    logger.warning(f"Failed to load local CLIP weights: {e}")
            else:
                logger.info("No local CLIP weights file found, using default pretrained weights")
        else:
            # Use a timm backbone
            if not _HAS_TIMM:
                raise ImportError("timm is not installed but a timm backbone was requested.")
            self._vision_kind = 'timm'
            model_name = str(self.backbone)
            logger.info(f"Initializing timm backbone: {model_name} (pretrained={self.pretrained})")

            # ---- HANDLE LEVIT FIRST (avoid features_only and filter_fn assert) ----
            if 'levit' in model_name.lower():
                try:
                    # Load with classifier intact so pretrained weights can map
                    self.vision_encoder = timm.create_model(
                        model_name,
                        pretrained=self.pretrained
                    )
                except Exception as e:
                    logger.warning(f"LeViT pretrained load failed ({e}); retrying with pretrained=False to bypass filter.")
                    self.vision_encoder = timm.create_model(
                        model_name,
                        pretrained=False
                    )

                # Determine feature dim
                self.vision_dim = getattr(self.vision_encoder, 'num_features', None)
                if not isinstance(self.vision_dim, int) or self.vision_dim <= 0:
                    dummy = torch.zeros(1, 3, self.backbone_image_size, self.backbone_image_size)
                    with torch.no_grad():
                        if hasattr(self.vision_encoder, 'forward_features'):
                            feat_vec = self.vision_encoder.forward_features(dummy)
                        else:
                            feat_vec = self.vision_encoder(dummy)
                    self.vision_dim = int(feat_vec.shape[1])

                # Remove classifier; enforce global pooling for features
                if hasattr(self.vision_encoder, 'reset_classifier'):
                    self.vision_encoder.reset_classifier(0, global_pool='avg')

                self.vision_pool = None
                self.vision_proj = None
                self._vision_kind = 'timm_pooled'
                logger.info(f"Vision backbone: kind=timm_pooled(levit), name={model_name}, vision_dim={self.vision_dim}")
                return
            # ---- END LEVIT EARLY RETURN ----

            # Identify non-convolutional transformer-like models
            nonconv_keys = ['vit', 'deit', 'swin', 'maxvit', 'eva', 'beit', 'convnextv2']
            is_nonconv = any(k in model_name.lower() for k in nonconv_keys)

            if not is_nonconv:
                # Prefer features_only path for CNNs
                try:
                    self.vision_encoder = timm.create_model(
                        model_name,
                        pretrained=self.pretrained,
                        features_only=True,
                        out_indices=[-1]
                    )
                    try:
                        self.vision_dim = int(self.vision_encoder.feature_info[-1]['num_chs'])
                    except Exception:
                        dummy = torch.zeros(1, 3, self.backbone_image_size, self.backbone_image_size)
                        with torch.no_grad():
                            feat = self.vision_encoder(dummy)[0]
                        self.vision_dim = feat.shape[1]
                    self.vision_pool = nn.AdaptiveAvgPool2d(1)
                    self.vision_proj = None
                    self._vision_kind = 'timm_features'
                    logger.info(f"Vision backbone: kind=timm_features, name={model_name}, vision_dim={self.vision_dim}")
                except Exception as e:
                    logger.info(f"features_only path unavailable for {model_name} ({e}). Falling back to pooled forward.")
                    try:
                        self.vision_encoder = timm.create_model(
                            model_name,
                            pretrained=self.pretrained,
                            num_classes=0,
                            global_pool='avg'
                        )
                    except Exception as e2:
                        logger.warning(f"Pooled forward with pretrained=True failed for {model_name} ({e2}). Retrying with pretrained=False.")
                        self.vision_encoder = timm.create_model(
                            model_name,
                            pretrained=False,
                            num_classes=0,
                            global_pool='avg'
                        )
                    self.vision_dim = getattr(self.vision_encoder, 'num_features', None)
                    if not isinstance(self.vision_dim, int) or self.vision_dim <= 0:
                        dummy = torch.zeros(1, 3, self.backbone_image_size, self.backbone_image_size)
                        with torch.no_grad():
                            feat_vec = self.vision_encoder(dummy)
                        self.vision_dim = int(feat_vec.shape[1])
                    self.vision_pool = None
                    self.vision_proj = None
                    self._vision_kind = 'timm_pooled'
                    logger.info(f"Vision backbone: kind=timm_pooled, name={model_name}, vision_dim={self.vision_dim}")
            else:
                # Transformer-like models: pooled forward
                try:
                    self.vision_encoder = timm.create_model(
                        model_name,
                        pretrained=self.pretrained,
                        num_classes=0,
                        global_pool='avg'
                    )
                except Exception as e:
                    logger.warning(f"Transformer pooled forward with pretrained=True failed for {model_name} ({e}). Retrying with pretrained=False.")
                    self.vision_encoder = timm.create_model(
                        model_name,
                        pretrained=False,
                        num_classes=0,
                        global_pool='avg'
                    )
                self.vision_dim = getattr(self.vision_encoder, 'num_features', None)
                if not isinstance(self.vision_dim, int) or self.vision_dim <= 0:
                    dummy = torch.zeros(1, 3, self.backbone_image_size, self.backbone_image_size)
                    with torch.no_grad():
                        feat_vec = self.vision_encoder(dummy)
                    self.vision_dim = int(feat_vec.shape[1])
                self.vision_pool = None
                self.vision_proj = None
                self._vision_kind = 'timm_pooled'
                logger.info(f"Vision backbone: kind=timm_pooled, name={model_name}, vision_dim={self.vision_dim}")
                
                
    def _freeze_backbone(self, freeze: bool = True):
        """
        Freeze or unfreeze the vision backbone. If using CLIP and freeze=False,
        we still keep most of CLIP frozen by default, except the last block or visual projection.
        """
        if self._vision_kind == 'clip':
            # Start by freezing all
            for p in self.vision_encoder.parameters():
                p.requires_grad = not freeze
            if not freeze:
                # Unfreeze only the last block by default (safer for fine-tuning on device)
                if hasattr(self.vision_encoder, 'visual_projection'):
                    for p in self.vision_encoder.visual_projection.parameters():
                        p.requires_grad = True
                elif hasattr(self.vision_encoder, 'vision_model') and hasattr(self.vision_encoder.vision_model, 'encoder'):
                    layers = self.vision_encoder.vision_model.encoder.layers
                    if len(layers) > 0:
                        for p in layers[-1].parameters():
                            p.requires_grad = True
        else:
            # timm backbone
            for p in self.vision_encoder.parameters():
                p.requires_grad = not freeze

    def _extract_vision_features(self, frames):
        """
        Generic feature extractor → returns [B, T, D] where D=self.vision_dim.
        For CLIP, uses pooler_output.
        For timm, supports both features_only and pooled models.
        """
        batch_size, num_frames, channels, height, width = frames.shape
        x = frames.view(-1, channels, height, width)  # [B*T, C, H, W]

        if self._vision_kind == 'clip':
            with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
                outputs = self.vision_encoder(x)
                feats = outputs.pooler_output  # [B*T, D]
        elif self._vision_kind == 'timm_features':
            with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
                feat_maps = self.vision_encoder(x)[0]  # [B*T, C, h, w]
                pooled = self.vision_pool(feat_maps)   # [B*T, C, 1, 1]
                feats = pooled.flatten(1)              # [B*T, C]
                if self.vision_proj is not None:
                    feats = self.vision_proj(feats)
        elif self._vision_kind == 'timm_pooled':
            with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
                feats = self.vision_encoder(x)         # [B*T, D] already pooled
        else:
            raise RuntimeError(f"Unknown vision kind: {self._vision_kind}")

        return feats.view(batch_size, num_frames, -1)

    def extract_clip_features(self, frames):
        # Backward compatibility: keep method name but delegate to generic extractor
        return self._extract_vision_features(frames)
    
    def process_site(self, video, site_idx, mask=None, batch_idx=None, site_pos=None):
        """
        Process a single site's video using soft attention aggregation.
        
        Args:
            video: Video frames [1, T, C, H, W]
            site_idx: Site index
            mask: Frame mask [1, T]
            batch_idx: Batch index for tracking
            site_pos: Site position for tracking
            
        Returns:
            Dictionary with:
                - selected_features: Aggregated site features [1, hidden_dim]
                - pathology_scores: Pathology predictions [1, num_pathologies] if enabled
                - action_logits: Frame attention logits [1, T]
                - state_values: Value estimates for RL
                
        Note:
            Uses temperature-controlled soft attention to aggregate frame features.
            Temperature is controlled by config.attention_temperature (default 0.5).
            Lower temperature = sharper attention (more peaked).
            Higher temperature = softer attention (more uniform).
        """
        # Extract vision features (CLIP or timm)
        clip_features = self._extract_vision_features(video)  # [1, T, vision_dim]
        
        # Select key frames using enhanced frame selector
        action_logits, state_values, enhanced_features = self.frame_selector(
            clip_features, mask, batch_idx, site_pos
        )
        
        # Sample actions (frame indices) - kept for RL compatibility but not used for selection
        actions, _ = self.frame_selector.select_action(
            action_logits, state_values, enhanced_features, batch_idx, site_pos
        )
        
        # Use SOFT attention-based selection with temperature-controlled sharpness
        # This allows gradients to flow back to the attention logits
        tau = getattr(self.config, 'attention_temperature', 0.5)  # Temperature hyperparameter
        
        if mask is not None:
            # Apply mask to logits
            valid_mask = mask[0]
            masked_logits = action_logits[0].clone()
            # Use a safe mask value that works with both FP16 and FP32
            mask_value = -65000.0 if masked_logits.dtype == torch.float16 else -1e9
            masked_logits[~valid_mask] = mask_value
            
            # Compute attention weights with temperature
            attention_weights = F.softmax(masked_logits / tau, dim=0)
            
            # Renormalize after masking
            attention_weights = attention_weights * valid_mask.float()
            attention_weights = attention_weights / (attention_weights.sum() + 1e-9)
        else:
            attention_weights = F.softmax(action_logits[0] / tau, dim=0)
        
        # Weighted sum of features (differentiable)
        selected_features = torch.sum(
            attention_weights.unsqueeze(-1) * enhanced_features[0],  # [T, hidden_dim]
            dim=0
        )  # [hidden_dim]
        
        # Add batch dimension: [1, hidden_dim]
        selected_features = selected_features.unsqueeze(0)
        
        # Add sequence dimension for pathology modules: [1, 1, hidden_dim]
        # PathologyModule expects [B, k, hidden_dim] where k is the number of frames
        selected_features_with_seq = selected_features.unsqueeze(1)
        
        # Mask for single aggregated feature
        selected_mask = torch.ones(1, 1, dtype=torch.bool, device=video.device)
        
        # Process pathologies (if enabled)
        pathology_scores = None
        if self.use_pathology_loss and self.pathology_modules is not None:
            pathology_scores = []
            pathology_attentions = []
            pathology_features = []
            
            for module in self.pathology_modules:
                score, attention, features = module(selected_features_with_seq, selected_mask)
                pathology_scores.append(score)
                pathology_attentions.append(attention)
                pathology_features.append(features)
            
            # Stack pathology outputs
            pathology_scores = torch.cat(pathology_scores, dim=1)  # [1, num_pathologies]
        
        # Return comprehensive output
        # Note: selected_indices is kept for backward compatibility but not used
        # The soft selection doesn't use discrete indices
        return {
            'selected_features': selected_features,
            'selected_indices': None,  # No longer using hard indices
            'pathology_scores': pathology_scores,
            'action_logits': action_logits,
            'state_values': state_values,
            'batch_idx': batch_idx,
            'site_idx': site_pos
        }

    def process_patient(self, site_videos, site_indices, site_masks):
        """
        Process videos from multiple anatomical sites for a patient.
        
        Args:
            site_videos: Videos from different sites [B, N, T, C, H, W]
            site_indices: Anatomical site indices [B, N]
            site_masks: Site masks [B, N]
        """
        batch_size, max_sites = site_videos.shape[0], site_videos.shape[1]
        
        all_site_features = []
        all_pathology_scores = []
        all_site_metadata = []
        
        # Process each patient
        for b in range(batch_size):
            site_features = []
            site_pathology_scores = []
            site_metadata = []
            
            # Process each valid site
            valid_sites = site_masks[b].sum().item()
            for n in range(valid_sites):
                # Get video and site index
                video = site_videos[b, n].unsqueeze(0)  # [1, T, C, H, W]
                site_idx = site_indices[b, n].item()
                
                # Create frame masks (all valid initially)
                frame_mask = torch.ones(1, video.shape[1], dtype=torch.bool, device=video.device)
                
                # Process site
                site_output = self.process_site(
                    video, site_idx, frame_mask, batch_idx=b, site_pos=n
                )
                
                # Get selected features (now already a single vector per site)
                selected_features = site_output['selected_features']  # [1, hidden_dim]
                site_features.append(selected_features)
                
                if self.use_pathology_loss and site_output['pathology_scores'] is not None:
                    site_pathology_scores.append(site_output['pathology_scores'])
                
                # Store site metadata (action logits, state values, etc. for all selection strategies)
                site_metadata.append({
                    'batch_idx': b,
                    'site_idx': n,
                    'selected_indices': site_output['selected_indices'],
                    'action_logits': site_output['action_logits'],
                    'state_values': site_output['state_values']
                })
            
            # Stack outputs for this patient
            if site_features:
                site_features = torch.cat(site_features, dim=0)  # [valid_sites, hidden_dim]
                
                # Create padded tensors
                padded_features = torch.zeros(max_sites, self.hidden_dim, device=site_videos.device)
                padded_features[:valid_sites] = site_features
                all_site_features.append(padded_features)
                
                if self.use_pathology_loss:
                    if site_pathology_scores:
                        site_pathology_scores = torch.cat(site_pathology_scores, dim=0)  # [valid_sites, num_pathologies]
                        padded_scores = torch.zeros(max_sites, self.num_pathologies, device=site_videos.device)
                        padded_scores[:valid_sites] = site_pathology_scores
                        all_pathology_scores.append(padded_scores)
                    else:
                        all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
                
                all_site_metadata.append(site_metadata)
            else:
                # No valid sites
                all_site_features.append(torch.zeros(max_sites, self.hidden_dim, device=site_videos.device))
                if self.use_pathology_loss:
                    all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
                all_site_metadata.append([])
        
        # Stack across batch
        all_site_features = torch.stack(all_site_features)  # [B, N, hidden_dim]
        
        if self.use_pathology_loss:
            all_pathology_scores = torch.stack(all_pathology_scores)  # [B, N, num_pathologies]
        else:
            all_pathology_scores = None
        
        return all_site_features, all_pathology_scores, all_site_metadata
    
    def forward(self, inputs):
        """
        Forward pass through the model.
        
        Args:
            inputs: Dictionary containing:
                - site_videos: Videos from different sites [B, N, T, C, H, W]
                - site_indices: Anatomical site indices [B, N]
                - site_masks: Site masks [B, N]
        """
        # Extract inputs
        site_videos = inputs['site_videos']
        site_indices = inputs['site_indices']
        site_masks = inputs['site_masks']
        
        # Process all sites for all patients
        site_features, pathology_scores, site_metadata = self.process_patient(
            site_videos, site_indices, site_masks
        )
        
        # Integrate site features with anatomical context
        if self.use_pathology_loss:
            integrated_features = self.site_integration(
                site_features, site_indices, pathology_scores
            )
        else:
            integrated_features = self.site_integration(site_features)
        
        # Apply patient-level aggregation
        if self.use_patient_mil and self.patient_mil is not None:
            patient_features, mil_attention = self.patient_mil(integrated_features, site_masks)
        else:
            mask = site_masks.float()
            weights = mask / (mask.sum(dim=1, keepdim=True) + 1e-6)
            patient_features = torch.bmm(weights.unsqueeze(1), integrated_features).squeeze(1)
            mil_attention = weights
        
        if self.training and hasattr(self, 'feature_noise_std'):
            noise = torch.randn_like(patient_features) * self.feature_noise_std
            patient_features = patient_features + noise
        
        # Multi-task classification (for compatibility with new training system)
        task_logits = {}
        for task_name in self.active_tasks:
            if task_name == 'TB Label':
                # Use the original TB classifier
                tb_logits = self.tb_classifier(patient_features)
                if self.num_classes == 1:
                    tb_logits = tb_logits.squeeze(-1)  # [B]
                task_logits['TB Label'] = tb_logits
            else:
                # Use task-specific classifiers for other tasks
                task_key = task_name.replace(' ', '_').replace('Label', 'label')
                if task_key in self.task_classifiers:
                    logits = self.task_classifiers[task_key](patient_features)
                    if self.num_classes == 1:
                        logits = logits.squeeze(-1)
                    task_logits[task_name] = logits

        # Calculate patient-level pathology scores using MIL attention (if enabled)
        patient_pathology_scores = None
        if self.use_pathology_loss and pathology_scores is not None:
            patient_pathology_scores = torch.bmm(
                mil_attention.unsqueeze(1),  # [B, 1, N]
                pathology_scores  # [B, N, num_pathologies]
            ).squeeze(1)  # [B, num_pathologies]
        
        # Flatten site_metadata into site_outputs for entropy loss computation
        # Each element in site_metadata is a list of dicts for one patient
        site_outputs = []
        for patient_sites in site_metadata:
            site_outputs.extend(patient_sites)
        
        # Return comprehensive output (compatible with new training system)
        output = {
            'task_logits': task_logits,  # NEW: Dict of task_name -> logits
            'patient_pathology_scores': patient_pathology_scores,
            'patient_features': patient_features,
            'pathology_scores': pathology_scores,
            'mil_attention': mil_attention,
            'site_features': site_features,
            'site_metadata': site_metadata,  # Metadata for all frame selection strategies
            'site_outputs': site_outputs  # Flattened version for entropy loss computation
        }
        
        # Keep backward compatibility - also include tb_logits
        if 'TB Label' in task_logits:
            output['tb_logits'] = task_logits['TB Label']
        
        return output
    
    def compute_losses(self, outputs, targets, pos_weights=None):
        """Fixed loss computation with proper gradient handling."""
        loss_dict = {}
        total_loss = 0.0
        
        # Default positive weights
        if pos_weights is None:
            pos_weights = {'TB Label': 1.4}
        
        # Task classification losses (existing code is mostly fine)
        for task_name in self.active_tasks:
            if task_name in outputs.get('task_logits', {}):
                # Get target labels
                if task_name == 'TB Label':
                    target_labels = targets['tb_labels']
                elif task_name == 'Pneumonia Label':
                    target_labels = targets['pneumonia_labels']
                elif task_name == 'Covid Label':
                    target_labels = targets['covid_labels']
                else:
                    continue
                
                # Skip if no valid labels
                valid_mask = target_labels >= 0
                if not valid_mask.any():
                    continue
                
                logits = outputs['task_logits'][task_name]
                
                # Get positive weight for this task
                pos_weight = pos_weights.get(task_name, 2.0)
                pos_weight_tensor = torch.tensor(pos_weight, device=logits.device)
                
                # Binary cross-entropy loss
                task_loss = F.binary_cross_entropy_with_logits(
                    logits[valid_mask],
                    target_labels[valid_mask].float(),
                    pos_weight=pos_weight_tensor
                )
                
                # Apply task weight
                task_weight = self.task_weights.get(task_name, 1.0)
                weighted_task_loss = task_loss * task_weight
                
                loss_dict[f'{task_name}_loss'] = task_loss.item()
                total_loss += weighted_task_loss
        
        # Add entropy regularization to encourage discriminative attention
        # This incentivizes the model to NOT have uniform attention weights
        if 'site_outputs' in outputs and self.training:
            entropy_loss = 0.0
            num_sites = 0
            
            # Get temperature from config (same as used in process_site())
            tau = getattr(self.config, 'attention_temperature', 0.5)
            
            for site_output in outputs['site_outputs']:
                if 'action_logits' in site_output and site_output['action_logits'] is not None:
                    action_logits = site_output['action_logits']
                    
                    # Apply temperature scaling (same as in process_site())
                    # This ensures entropy is computed on the ACTUAL distribution used for selection
                    scaled_logits = action_logits / tau
                    
                    # Compute attention weights (probabilities) with temperature
                    attention_probs = F.softmax(scaled_logits, dim=-1)
                    
                    # Compute entropy: H = -sum(p * log(p))
                    # High entropy = uniform distribution (bad - all frames equally weighted)
                    # Low entropy = peaked distribution (good - focuses on few frames)
                    log_probs = F.log_softmax(scaled_logits, dim=-1)
                    entropy = -torch.sum(attention_probs * log_probs, dim=-1).mean()
                    
                    entropy_loss += entropy
                    num_sites += 1
            
            if num_sites > 0:
                # Average entropy across sites
                avg_entropy = entropy_loss / num_sites
                
                # We want to MINIMIZE entropy (encourage peaked distributions)
                # Use entropy_weight from config (default 0.001)
                entropy_weight = getattr(self.config, 'entropy_weight', 0.001)
                entropy_penalty = entropy_weight * avg_entropy
                total_loss += entropy_penalty
                
                # Log both raw entropy and weighted penalty for monitoring
                loss_dict['attention_entropy'] = avg_entropy.item()
                loss_dict['attention_entropy_penalty'] = entropy_penalty.item()
        
        if self.use_pathology_loss and 'pathology_scores' in outputs and outputs['pathology_scores'] is not None:
            pathology_scores = outputs['pathology_scores']
            pathology_labels = targets['pathology_labels']
            for i in range(self.num_pathologies):
                # Extract scores and labels for this pathology
                if pathology_scores.dim() == 3:  # [B, N, num_pathologies]
                    path_score_i = pathology_scores[:, :, i]
                    path_label_i = pathology_labels[:, :, i]
                else:
                    path_score_i = pathology_scores[:, i]
                    path_label_i = pathology_labels[:, i]
                
                # Valid mask
                valid_mask = path_label_i >= 0
                
                if valid_mask.any():
                    # Positive weights for different pathologies
                    pos_weights_path = [1.0, 4.0, 4.0, 4.0, 15.0]
                    pos_weight = torch.tensor(pos_weights_path[i % len(pos_weights_path)], device=pathology_scores.device)
                    
                    # Binary cross-entropy loss
                    loss_name = f'pathology_{i}_loss'
                    p_loss = F.binary_cross_entropy_with_logits(
                        path_score_i[valid_mask],
                        path_label_i[valid_mask].float(),
                        pos_weight=pos_weight
                    )
                    
                    # Store loss
                    loss_dict[loss_name] = p_loss.item()
                    
                    # Add to total loss
                    pathology_weight = 0.2
                    total_loss += pathology_weight * p_loss
        
        # Ensure total_loss is a proper tensor with gradients
        if isinstance(total_loss, (int, float)):
            if total_loss == 0:
                # Create a dummy loss with gradients if needed
                dummy_param = next(self.parameters())
                total_loss = torch.tensor(0.0, device=dummy_param.device, requires_grad=True)
        
        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict
