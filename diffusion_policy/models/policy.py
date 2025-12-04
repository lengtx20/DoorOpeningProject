import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from .model import VisionEncoder, ConditionalUnet1D

class DiffusionPolicy(nn.Module):
    def __init__(self, 
                 action_dim=10, 
                 obs_horizon=2, 
                 pred_horizon=16,
                 vision_feature_dim=256,
                 proprio_dim=29, 
                 use_proprio=True,
                 num_train_timesteps=100,
                 num_inference_steps=16): 
        super().__init__()
        
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.use_proprio = use_proprio
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.vision_encoder = VisionEncoder(embedding_dim=vision_feature_dim)
        vision_out_dim = obs_horizon * vision_feature_dim

        if use_proprio:
            proprio_input_dim = obs_horizon * proprio_dim
            self.proprio_mlp = nn.Sequential(
                nn.Linear(proprio_input_dim, 128),
                nn.Mish(),
                nn.Linear(128, 128)
            )
            global_cond_dim = vision_out_dim + 128
        else:
            global_cond_dim = vision_out_dim


        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
    
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim
        )

    def get_global_cond(self, image, agent_pos):
        vision_feat = self.vision_encoder(image) 
        if self.use_proprio and agent_pos is not None:
            B = agent_pos.shape[0]
            proprio_flat = agent_pos.view(B, -1)
            proprio_feat = self.proprio_mlp(proprio_flat)
            global_cond = torch.cat([vision_feat, proprio_feat], dim=-1)
        else:
            global_cond = vision_feat
        # print(global_cond.shape)
        return global_cond

    def compute_loss(self, batch):
   
        images = batch['image']
        agent_pos = batch['agent_pos']
        actions = batch['action']
        B = actions.shape[0]
        
        global_cond = self.get_global_cond(images, agent_pos)

        noise = torch.randn(actions.shape, device=actions.device)
        
   
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=actions.device
        ).long()


        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        # print(noisy_actions.shape)

        noise_pred = self.model(noisy_actions, timesteps, global_cond)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict_action(self, image, agent_pos=None):

        B = image.shape[0]
        device = image.device
        global_cond = self.get_global_cond(image, agent_pos)



        noisy_action = torch.randn((B, self.pred_horizon, self.action_dim), device=device)
        
        
        self.inference_scheduler.set_timesteps(self.num_inference_steps)


        for k in self.inference_scheduler.timesteps:

            timestep = torch.full(
                (B,),
                fill_value=int(k),
                dtype=torch.long,
                device=device
            )

            noise_pred = self.model(noisy_action, timestep, global_cond)
            
        
            noisy_action = self.inference_scheduler.step(
                model_output=noise_pred, 
                timestep=k, 
                sample=noisy_action
            ).prev_sample
            # print(noisy_action.shape)

        return noisy_action