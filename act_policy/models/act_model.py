"""
ACT (Action Chunking with Transformers) Policy Implementation

Based on: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
Paper: https://arxiv.org/abs/2304.13705
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from torchvision import models


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]


class VisionEncoder(nn.Module):

    def __init__(self, output_dim: int = 512):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, obs_horizon, c, h, w = x.shape
        x = x.view(batch_size * obs_horizon, c, h, w)

        # [batch*obs_horizon, 512, 1, 1]
        features = self.backbone(x)  
        # [batch*obs_horizon, 512]
        features = features.view(batch_size * obs_horizon, -1)  

        # [batch*obs_horizon, output_dim]
        features = self.fc(features) 

        # [batch, obs_horizon, output_dim]
        features = features.view(batch_size, obs_horizon, -1)  

        return features


class ACTPolicy(nn.Module):

    def __init__(
        self,
        proprio_dim: int = 118,
        stage_dim: int = 5,
        action_dim: int = 10,
        obs_horizon: int = 1,
        pred_horizon: int = 16,
        hidden_dim: int = 512,
        nheads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 7,
        latent_dim: int = 32,
        dropout: float = 0.1,
        use_vae: bool = True,
        use_image: bool = True,
        vision_feature_dim: int = 512,
    ):
        super().__init__()

        self.proprio_dim = proprio_dim
        self.stage_dim = stage_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_vae = use_vae
        self.use_image = use_image

        if use_image:
            self.vision_encoder = VisionEncoder(output_dim=vision_feature_dim)
            self.vision_projection = nn.Sequential(
                nn.Linear(vision_feature_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.vision_encoder = None
            self.vision_projection = None

        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.stage_encoder = nn.Sequential(
            nn.Linear(stage_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        
        if use_image:
            # vision + proprio + stage
            concat_dim = 3 * hidden_dim  
        else:
            # proprio + stage
            concat_dim = 2 * hidden_dim  

        self.concat_projection = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_encoder = nn.Linear(action_dim, hidden_dim)

        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max(obs_horizon, pred_horizon))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=False,  # Use [seq, batch, feature] format
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        if use_vae:
            vae_hidden = hidden_dim
            # q(z|obs, gt_actions)
            self.vae_encoder = nn.Sequential(
                nn.Linear(hidden_dim * obs_horizon + action_dim * pred_horizon, vae_hidden),
                nn.ReLU(),
                nn.Linear(vae_hidden, vae_hidden),
                nn.ReLU(),
            )
            self.vae_mu = nn.Linear(vae_hidden, latent_dim)
            self.vae_logvar = nn.Linear(vae_hidden, latent_dim)

            # p(actions|z, obs)
            self.latent_decoder = nn.Sequential(
                nn.Linear(latent_dim + hidden_dim * obs_horizon, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.decoder_input_projection = nn.Linear(2 * hidden_dim, hidden_dim)

        self.action_head = nn.Linear(hidden_dim, action_dim)

        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_observations(
        self,
        proprio: torch.Tensor,
        stage: torch.Tensor,
        image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:


        # [batch, obs_horizon, proprio_dim] -> [obs_horizon, batch, proprio_dim]
        proprio = proprio.transpose(0, 1)

        # [obs_horizon, batch, hidden_dim]
        proprio_features = self.proprio_encoder(proprio)  

        # [batch, hidden_dim]
        stage_features = self.stage_encoder(stage) 
        stage_features = stage_features.unsqueeze(0)

        # [obs_horizon, batch, hidden_dim] 
        stage_features = stage_features.expand(proprio.shape[0], -1, -1) 

        if self.use_image and image is not None:
            # [batch, obs_horizon, vision_feature_dim]
            vision_features = self.vision_encoder(image)  
            # [obs_horizon, batch, vision_feature_dim]
            vision_features = vision_features.transpose(0, 1)  

            # [obs_horizon, batch, hidden_dim]
            vision_features = self.vision_projection(vision_features) 
            combined = torch.cat([vision_features, proprio_features, stage_features], dim=-1)
        else:
            combined = torch.cat([proprio_features, stage_features], dim=-1)
            # [obs_horizon, batch, 2*hidden_dim]

        # [obs_horizon, batch, hidden_dim]
        encoded = self.concat_projection(combined)  
        encoded = self.pos_encoding(encoded)

        # [obs_horizon, batch, hidden_dim]
        memory = self.transformer_encoder(encoded)  

        return memory

    def decode_actions(
        self,
        memory: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        action_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
   
        batch_size = memory.shape[1]

        if action_targets is not None:
            action_seq = action_targets.transpose(0, 1)  
            decoder_input = self.action_encoder(action_seq)  
        else:
            decoder_input = torch.zeros(
                self.pred_horizon, batch_size, self.hidden_dim,
                device=memory.device
            )

        if self.use_vae and latent is not None:
            context = memory.transpose(0, 1).reshape(batch_size, -1)

            latent_features = self.latent_decoder(
                torch.cat([latent, context], dim=-1)
            )

            decoder_input = torch.cat([decoder_input, latent_features.unsqueeze(0).expand(self.pred_horizon, -1, -1)], dim=-1)
            decoder_input = self.decoder_input_projection(decoder_input)

        decoder_input = self.pos_encoding(decoder_input)

        # [pred_horizon, batch, hidden_dim]
        decoded = self.transformer_decoder(
            decoder_input,
            memory,
        )  

        # [pred_horizon, batch, action_dim]
        actions = self.action_head(decoded)  

        # [pred_horizon, batch, action_dim] -> [batch, pred_horizon, action_dim]
        actions = actions.transpose(0, 1)

        return actions

    def encode_vae(
        self,
        proprio_encoded: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = actions.shape[0]

        proprio_flat = proprio_encoded.transpose(0, 1).reshape(batch_size, -1)

        actions_flat = actions.reshape(batch_size, -1)

        vae_input = torch.cat([proprio_flat, actions_flat], dim=-1)

        h = self.vae_encoder(vae_input)
        mu = self.vae_mu(h)
        logvar = self.vae_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
      
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch: dict, inference: bool = False) -> dict:
      
        proprio = batch['proprio']
        stage = batch['stage']
        image = batch.get('image', None) if self.use_image else None
        batch_size = proprio.shape[0]

        memory = self.encode_observations(proprio, stage, image)

        latent = None
        mu, logvar = None, None

        if self.use_vae:
            if not inference and 'action' in batch:
                actions_gt = batch['action']
                mu, logvar = self.encode_vae(memory, actions_gt)
                latent = self.reparameterize(mu, logvar)
            else:
                latent = torch.randn(
                    batch_size, self.latent_dim,
                    device=proprio.device
                ) * 0.1  


        if not inference and 'action' in batch:
            actions_pred = self.decode_actions(
                memory,
                latent=latent,
                # action_targets=batch['action']
                action_targets=None
            )
        else:
            actions_pred = self.decode_actions(
                memory,
                latent=latent,
                action_targets=None
            )

        output = {
            'action': actions_pred,
        }

        if self.use_vae and mu is not None:
            output['mu'] = mu
            output['logvar'] = logvar

        return output

    def compute_loss(self, batch: dict, kl_weight: float = 1.0):
        output = self.forward(batch, inference=False)
        actions_pred = output['action']
        actions_gt = batch['action']

        loss_per_dim = torch.abs(actions_pred - actions_gt)

        recon_loss = loss_per_dim.mean()

        kl_loss = torch.tensor(0.0, device=actions_pred.device)
        mu = None
        logvar = None
        if self.use_vae and 'mu' in output:
            mu = output['mu']
            logvar = output['logvar']
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / mu.shape[0]

        total_loss = recon_loss + kl_weight * kl_loss

        result = {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
        if mu is not None:
            result['mu'] = mu
            result['logvar'] = logvar

        return result

