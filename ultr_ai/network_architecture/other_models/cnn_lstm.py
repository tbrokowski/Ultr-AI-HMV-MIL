import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torchvision.models as models

# Import base components from original model
from ultr_ai.network_architecture.components.general_components import (
    PathologyModule, SiteIntegrationModule, DeepAttentionMIL
)
from ultr_ai.network_architecture.components import MultiTaskModel, AttentionPoolSelector

logger = logging.getLogger(__name__)


class CNNLSTMMultiTaskModel(nn.Module):
    """Ablation: CNN-LSTM using PyTorch ResNet + LSTM."""
    
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
        self.selection_strategy = 'cnn_lstm'
        
        logger.info("Using CNN-LSTM with PyTorch ResNet + LSTM")
        
        # CNN backbone (ResNet18)
        self.cnn_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final layers
        self.cnn_backbone = nn.Sequential(*list(self.cnn_backbone.children())[:-2])
        
        # Adaptive pooling to get fixed-size features
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # CNN feature dimension (ResNet18 conv features = 512)
        cnn_dim = 512
        lstm_hidden = 512
        
        # Project CNN features
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_dim, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Output projection
        self.feature_projection = nn.Sequential(
            nn.Linear(lstm_hidden * 2, self.hidden_dim),  # *2 for bidirectional
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Frame selector for selecting key frames from LSTM output
        self.frame_selector = AttentionPoolSelector(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_heads=8
        )
        
        # Same downstream modules as others
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
    
    def process_video_cnn_lstm(self, video):
        """Process video through CNN-LSTM pipeline."""
        batch_size, num_frames, channels, height, width = video.shape
        
        # Reshape to process all frames together
        frames = video.view(-1, channels, height, width)  # [B*T, C, H, W]
        
        # Extract CNN features
        cnn_features = self.cnn_backbone(frames)  # [B*T, 512, H', W']
        cnn_features = self.cnn_pool(cnn_features)  # [B*T, 512, 1, 1]
        cnn_features = cnn_features.view(-1, 512)  # [B*T, 512]
        
        # Project CNN features
        projected_features = self.cnn_projection(cnn_features)  # [B*T, lstm_hidden]
        
        # Reshape for LSTM
        lstm_input = projected_features.view(batch_size, num_frames, -1)  # [B, T, lstm_hidden]
        
        # LSTM processing
        lstm_output, _ = self.lstm(lstm_input)  # [B, T, lstm_hidden*2]
        
        # Project to final output dimension
        output_features = self.feature_projection(lstm_output)  # [B, T, hidden_dim]
        
        return output_features
    
    def forward(self, inputs):
        """Forward pass using CNN-LSTM backbone."""
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
                
                # Extract frame-level features using CNN-LSTM
                frame_features = self.process_video_cnn_lstm(video)  # [1, T, hidden_dim]
                
                # Select key frames using attention with soft selection
                attention_scores, _, selected_features = self.frame_selector(frame_features)
                
                # Use soft attention pooling instead of hard top-k selection
                # This allows gradients to flow back to attention_scores
                num_frames = frame_features.shape[1]
                if num_frames > 0:
                    # Compute attention weights using softmax (already done in forward)
                    attention_weights = F.softmax(attention_scores[0], dim=0)  # [T]
                    
                    # Weighted sum of features (soft selection - differentiable!)
                    selected = torch.sum(
                        attention_weights.unsqueeze(-1) * selected_features[0],  # [T, hidden_dim]
                        dim=0,
                        keepdim=True
                    )  # [1, hidden_dim]
                else:
                    selected = selected_features[0, :1]  # First frame
                
                site_features.append(selected)
                
                # Process pathology
                if self.use_pathology_loss and self.pathology_modules is not None:
                    dummy_frames = selected.unsqueeze(1).repeat(1, 3, 1)  # [1, 3, hidden_dim]
                    dummy_mask = torch.ones(1, 3, dtype=torch.bool, device=video.device)
                    
                    pathology_scores = []
                    for module in self.pathology_modules:
                        score, _, _ = module(dummy_frames, dummy_mask)
                        pathology_scores.append(score)
                    
                    site_pathology_scores.append(torch.cat(pathology_scores, dim=1))
            
            # Handle padding (same as ResNet3D)
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
        
        # Rest same as ResNet3D
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