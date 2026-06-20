import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Import base components from original model
from ultr_ai.network_architecture.components.general_components import (
    PathologyModule, SiteIntegrationModule, DeepAttentionMIL
)
from ultr_ai.network_architecture.components import MultiTaskModel, AttentionPoolSelector

logger = logging.getLogger(__name__)



class VideoTransformerMultiTaskModel(nn.Module):
    """Ablation: Video Vision Transformer (ViViT) for video understanding."""
    
    def __init__(self, config):
        super().__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = getattr(config, 'num_classes', 1)
        self.hidden_dim = getattr(config, 'hidden_dim', 512)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.num_pathologies = getattr(config, 'num_pathologies', 4)
        self.num_sites = getattr(config, 'num_sites', 15)
        self.device = getattr(config, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.active_tasks = getattr(config, 'active_tasks', ['TB Label'])
        self.use_pathology_loss = getattr(config, 'use_pathology_loss', True)
        self.task_weights = getattr(config, 'task_weights', {'TB Label': 1.0})
        self.selection_strategy = 'vivit'
        
        logger.info("Using Video Vision Transformer (ViViT) architecture")
        
        # ViViT configuration
        self.num_frames = 16  # Standard for ViViT
        self.patch_size = 16
        self.image_size = 224
        self.hidden_size = 768
        self.num_heads = 12
        self.num_layers = 12
        
        # Create ViViT model
        self.video_transformer = self._create_vivit_model()
        logger.info("Created ViViT model (unpretrained)")
        
        # Feature projection from transformer output
        transformer_dim = self.hidden_size
        self.feature_projection = nn.Sequential(
            nn.Linear(transformer_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Frame selector for temporal attention
        self.frame_selector = AttentionPoolSelector(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_heads=8
        )
        
        # Same downstream modules
        if self.use_pathology_loss:
            self.pathology_modules = nn.ModuleList([
                PathologyModule(
                    feature_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim // 2,
                    dropout=self.dropout_rate,
                    name=f'pathology_{i}'
                ) for i in range(self.num_pathologies)
            ])
        
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
        
        self.patient_mil = DeepAttentionMIL(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            dropout=self.dropout_rate,
            num_heads=8
        )
        
        # Task classifiers
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
        
        self.tb_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
    
    def _create_vivit_model(self):
        """Create ViViT (Video Vision Transformer) model."""
        class ViViTModel(nn.Module):
            def __init__(self, 
                         image_size=224,
                         patch_size=16, 
                         num_frames=16,
                         hidden_size=768,
                         num_heads=12,
                         num_layers=12,
                         dropout=0.1):
                super().__init__()
                
                self.image_size = image_size
                self.patch_size = patch_size
                self.num_frames = num_frames
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                
                # Calculate number of patches
                self.num_patches_per_frame = (image_size // patch_size) ** 2
                self.total_patches = self.num_patches_per_frame * num_frames
                
                # Patch embedding - convert video patches to embeddings
                self.patch_embedding = nn.Conv3d(
                    in_channels=3,
                    out_channels=hidden_size,
                    kernel_size=(1, patch_size, patch_size),
                    stride=(1, patch_size, patch_size)
                )
                
                # Positional embeddings
                # Spatial position embeddings for each frame
                self.spatial_pos_embedding = nn.Parameter(
                    torch.randn(1, self.num_patches_per_frame, hidden_size) * 0.02
                )
                
                # Temporal position embeddings for frames
                self.temporal_pos_embedding = nn.Parameter(
                    torch.randn(1, num_frames, hidden_size) * 0.02
                )
                
                # Class token
                self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
                
                # Transformer encoder layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True  # Pre-norm like in ViT
                )
                
                self.transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_layers
                )
                
                # Layer norm
                self.layer_norm = nn.LayerNorm(hidden_size)
                
                # Dropout
                self.dropout = nn.Dropout(dropout)
                
                # Initialize weights
                self.apply(self._init_weights)
            
            def _init_weights(self, module):
                """Initialize weights following ViT/ViViT conventions."""
                if isinstance(module, (nn.Linear, nn.Conv3d)):
                    torch.nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    torch.nn.init.zeros_(module.bias)
                    torch.nn.init.ones_(module.weight)
            
            def forward(self, videos):
                """
                Forward pass for ViViT.
                
                Args:
                    videos: [B, T, C, H, W] or [B, C, T, H, W]
                """
                # Ensure correct format [B, C, T, H, W]
                if videos.dim() == 5 and videos.shape[2] == 3:
                    videos = videos.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
                
                batch_size = videos.shape[0]
                
                # Extract patches: [B, hidden_size, T, H//patch_size, W//patch_size]
                patches = self.patch_embedding(videos)
                
                # Reshape to [B, T, num_patches_per_frame, hidden_size]
                patches = patches.permute(0, 2, 3, 4, 1)  # [B, T, H', W', hidden_size]
                patches = patches.reshape(batch_size, self.num_frames, self.num_patches_per_frame, self.hidden_size)
                
                # Add spatial positional embeddings to each frame
                patches = patches + self.spatial_pos_embedding.unsqueeze(1)  # Broadcast across time
                
                # Reshape to [B, T*num_patches_per_frame, hidden_size]
                patches = patches.reshape(batch_size, self.total_patches, self.hidden_size)
                
                # Add temporal positional embeddings
                # Create temporal embedding for each patch
                temporal_emb = self.temporal_pos_embedding.repeat_interleave(self.num_patches_per_frame, dim=1)
                patches = patches + temporal_emb
                
                # Add class token
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                patches = torch.cat([cls_tokens, patches], dim=1)
                
                # Apply dropout
                patches = self.dropout(patches)
                
                # Transformer encoding
                encoded = self.transformer(patches)
                
                # Apply final layer norm
                encoded = self.layer_norm(encoded)
                
                # Split CLS token and patch tokens
                cls_output = encoded[:, 0]  # [B, hidden_size]
                patch_outputs = encoded[:, 1:]  # [B, total_patches, hidden_size]
                
                # Reshape patch outputs back to spatial-temporal format
                patch_outputs = patch_outputs.reshape(
                    batch_size, self.num_frames, self.num_patches_per_frame, self.hidden_size
                )
                
                # Average patch outputs per frame to get frame-level features
                frame_features = patch_outputs.mean(dim=2)  # [B, T, hidden_size]
                
                # Create output similar to other transformer models
                output = type('ViViTOutput', (), {
                    'last_hidden_state': frame_features,
                    'pooler_output': cls_output,
                    'patch_outputs': patch_outputs
                })()
                
                return output
        
        return ViViTModel(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout_rate
        )
    
    def process_video_vivit(self, video):
        """Process video through ViViT."""
        batch_size, num_frames, channels, height, width = video.shape
        
        # Resize if needed
        if height != self.image_size or width != self.image_size:
            video = F.interpolate(
                video.view(-1, channels, height, width),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).view(batch_size, num_frames, channels, self.image_size, self.image_size)
        
        # Sample frames if too many
        if num_frames > self.num_frames:
            # Uniform sampling
            indices = torch.linspace(0, num_frames - 1, self.num_frames, dtype=torch.long, device=video.device)
            video = video[:, indices]
        elif num_frames < self.num_frames:
            # Pad frames if too few by repeating the last frame
            padding_needed = self.num_frames - num_frames
            last_frame = video[:, -1:].repeat(1, padding_needed, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=1)
        
        # Process through ViViT
        try:
            outputs = self.video_transformer(video)
            
            # Get frame-level features
            if hasattr(outputs, 'last_hidden_state'):
                frame_features = outputs.last_hidden_state  # [B, T, hidden_size]
            else:
                # Fallback to pooler output repeated across frames
                frame_features = outputs.pooler_output.unsqueeze(1).repeat(1, self.num_frames, 1)
            
            # Project features
            projected_features = self.feature_projection(frame_features)
            
            return projected_features
            
        except Exception as e:
            logger.warning(f"Error in ViViT processing: {e}")
            # Fallback to zero features
            return torch.zeros(batch_size, self.num_frames, self.hidden_dim, 
                             device=video.device, requires_grad=True)
    
    def forward(self, inputs):
        """Forward pass using ViViT."""
        site_videos = inputs['site_videos']
        site_indices = inputs['site_indices']
        site_masks = inputs['site_masks']
        
        batch_size, max_sites = site_videos.shape[0], site_videos.shape[1]
        
        all_site_features = []
        all_pathology_scores = []
        
        # Process each site
        for b in range(batch_size):
            site_features = []
            site_pathology_scores = []
            
            valid_sites = site_masks[b].sum().item()
            for n in range(valid_sites):
                video = site_videos[b, n].unsqueeze(0)  # [1, T, C, H, W]
                
                try:
                    # Extract features using ViViT
                    frame_features = self.process_video_vivit(video)  # [1, T, hidden_dim]
                    
                    # Select key frames using attention with soft selection
                    attention_scores, _, selected_features = self.frame_selector(frame_features)
                    
                    # Use soft attention pooling instead of hard top-k selection
                    # This allows gradients to flow back to attention_scores
                    num_frames = frame_features.shape[1]
                    if num_frames > 0:
                        # Compute attention weights using softmax
                        attention_weights = F.softmax(attention_scores[0], dim=0)  # [T]
                        
                        # Weighted sum of features (soft selection - differentiable!)
                        selected = torch.sum(
                            attention_weights.unsqueeze(-1) * selected_features[0],  # [T, hidden_dim]
                            dim=0,
                            keepdim=True
                        )  # [1, hidden_dim]
                    else:
                        selected = selected_features[0, :1]
                    
                    site_features.append(selected)
                    
                    # Process pathology
                    if self.use_pathology_loss and self.pathology_modules is not None:
                        dummy_frames = selected.unsqueeze(1).repeat(1, 3, 1)
                        dummy_mask = torch.ones(1, 3, dtype=torch.bool, device=video.device)
                        
                        pathology_scores = []
                        for module in self.pathology_modules:
                            score, _, _ = module(dummy_frames, dummy_mask)
                            pathology_scores.append(score)
                        
                        site_pathology_scores.append(torch.cat(pathology_scores, dim=1))
                
                except Exception as e:
                    logger.warning(f"Error processing video with ViViT: {e}")
                    # Fallback to zeros
                    site_features.append(torch.zeros(1, self.hidden_dim, device=video.device))
                    if self.use_pathology_loss:
                        site_pathology_scores.append(torch.zeros(1, self.num_pathologies, device=video.device))
            
            # Handle padding
            if site_features:
                site_features = torch.cat(site_features, dim=0)
                padded_features = torch.zeros(max_sites, self.hidden_dim, device=site_videos.device)
                padded_features[:valid_sites] = site_features
                all_site_features.append(padded_features)
                
                if self.use_pathology_loss and site_pathology_scores:
                    site_pathology_scores = torch.cat(site_pathology_scores, dim=0)
                    padded_scores = torch.zeros(max_sites, self.num_pathologies, device=site_videos.device)
                    padded_scores[:valid_sites] = site_pathology_scores
                    all_pathology_scores.append(padded_scores)
                else:
                    all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
            else:
                all_site_features.append(torch.zeros(max_sites, self.hidden_dim, device=site_videos.device))
                all_pathology_scores.append(torch.zeros(max_sites, self.num_pathologies, device=site_videos.device))
        
        # Rest same as other models
        site_features = torch.stack(all_site_features)
        pathology_scores = torch.stack(all_pathology_scores) if self.use_pathology_loss else None
        
        if self.use_pathology_loss:
            integrated_features = self.site_integration(site_features, site_indices, pathology_scores)
        else:
            integrated_features = self.site_integration(site_features)
        
        patient_features, mil_attention = self.patient_mil(integrated_features, site_masks)
        
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
