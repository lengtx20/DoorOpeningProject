import torch
import numpy as np
import sys
from pathlib import Path


current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root.name != "DoorOpeningProject" and project_root.parent != project_root:
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from diffusion_policy.models.diffusion_unet import DiffusionUNet
from diffusion_policy.utils.normalizer import LinearNormalizer


class DiffusionController:

    def __init__(self, model_path, normalizer_path, device='cuda:0', use_ddim=True, ddim_steps=10):
        self.device = torch.device(device)
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        self.normalizer = self._load_normalizer(normalizer_path)
        self.model = self._load_model(model_path)
        self.model.eval()
        
        self.obs_history = []
        self.action_buffer = None
        self.action_counter = 0

    def _load_normalizer(self, path):
        data = np.load(path)
        normalizer = LinearNormalizer(mode='limits')
        normalizer.params['obs'] = {
            'min': data['obs_min'], 'max': data['obs_max'],
            'scale': data['obs_max'] - data['obs_min'],
        }
        
        # Avoid div by zero (Explicit check)
        scale = normalizer.params['obs']['scale']
        scale[scale < 1e-8] = 1.0
        
        normalizer.params['actions'] = {
            'min': data['actions_min'], 'max': data['actions_max'],
            'scale': data['actions_max'] - data['actions_min'],
        }
        act_scale = normalizer.params['actions']['scale']
        act_scale[act_scale < 1e-8] = 1.0
        
        return normalizer

    def _load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        obs_encoder_weight = state_dict['obs_encoder.0.weight']
        obs_dim_total = obs_encoder_weight.shape[1]
        input_proj_weight = state_dict['noise_pred_net.input_proj.block.0.weight']
        action_dim = input_proj_weight.shape[1]

        model = DiffusionUNet(
            obs_dim=obs_dim_total // 2,
            action_dim=action_dim,
            obs_horizon=2,
            pred_horizon=16,
            action_horizon=8,
            num_diffusion_iters=100,
            down_dims=(256, 512, 1024),
            obs_encoder_layers=(256, 256),
        ).to(self.device)
        model.load_state_dict(state_dict)
        return model

    def get_action(self, obs: np.ndarray):

        self.obs_history.append(obs)
        if len(self.obs_history) > self.model.obs_horizon:
            self.obs_history.pop(0)

        if len(self.obs_history) < self.model.obs_horizon:
            return np.zeros(self.model.action_dim, dtype=np.float32)

        if self.action_buffer is None or self.action_counter >= self.model.action_horizon:
            self.action_buffer = self._generate_actions()
            self.action_counter = 0

        action = self.action_buffer[self.action_counter]
        self.action_counter += 1
        return action

    def _generate_actions(self):
        obs_seq = np.stack(self.obs_history, axis=0) 
        obs_tensor = torch.from_numpy(obs_seq).float().to(self.device).unsqueeze(0)
        
        B, T, D = obs_tensor.shape
        obs_flat = obs_tensor.reshape(B * T, D)
        obs_norm = self.normalizer.normalize(obs_flat, 'obs').reshape(B, T, D)

        with torch.no_grad():
            action_pred = self.model.conditional_sample(
                obs_norm,
                use_ddim=self.use_ddim,
                ddim_steps=self.ddim_steps,
            )

        B, T_act, D_act = action_pred.shape
        action_flat = action_pred.reshape(B * T_act, D_act)
        action_denorm = self.normalizer.denormalize(action_flat, 'actions').reshape(B, T_act, D_act)
        
        return action_denorm[0].cpu().numpy()


class HierarchicalController:
    def __init__(self, env, model_path, normalizer_path, device="cuda"):
        self.env = env
        self.device = device
        
        self.brain = DiffusionController(
            model_path=model_path,
            normalizer_path=normalizer_path,
            device=device,
            use_ddim=True,
            ddim_steps=10
        )

        self.cmd_vel = "target_base_velocity"
        self.cmd_hand_l = "target_left_hand_pos_in_base"
        self.cmd_hand_r = "target_right_hand_pos_in_base"
        
        self.robot = env.unwrapped.scene["robot"]
        

        all_joints = self.robot.joint_names
        gripper_matches = [name for name in all_joints if "left_gripper" in name]
        
        if len(gripper_matches) > 0:
            self.left_gripper_idxs, _ = self.robot.find_joints("left_gripper_.*")
            self.has_gripper = True
        else:
            self.has_gripper = False

    def _inject_command(self, term_name, value_3d):

        cmd_man = self.env.unwrapped.command_manager
        term = cmd_man.get_term(term_name)

        target_tensor = term.command
        
   
        if target_tensor.shape[-1] == 7:
            target_tensor[:, :3] = value_3d
        elif target_tensor.shape[-1] == 3:
            target_tensor[:] = value_3d
        else:

            if target_tensor.shape[-1] == value_3d.shape[-1]:
                target_tensor[:] = value_3d

    def step(self, state_dict):
  
        obs_tensor = torch.cat([
            state_dict['l_hand'], 
            state_dict['r_hand'], 
            state_dict['vel'], 
            state_dict['grasp']
        ])
        
        obs_numpy = obs_tensor.cpu().numpy()

  
        action_numpy = self.brain.get_action(obs_numpy)

    
        cmd = torch.from_numpy(action_numpy).to(self.device)

        self._inject_command(self.cmd_hand_l, cmd[0:3].unsqueeze(0))

        self._inject_command(self.cmd_hand_r, cmd[3:6].unsqueeze(0))

        self._inject_command(self.cmd_vel,    cmd[6:9].unsqueeze(0))
        

        if self.has_gripper:
            grasp_prob = cmd[9]
            grip_pos = 0.0 if grasp_prob > 0.5 else 0.1
            self.robot.write_data.joint_pos_target[:, self.left_gripper_idxs] = grip_pos