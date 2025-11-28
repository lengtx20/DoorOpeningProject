# Diffusion Policy for G1 Door Opening

This directory contains a diffusion policy implementation for training the Unitree G1 humanoid robot to walk through doors using demonstration data and simulation.

## Overview

The diffusion policy learns to generate robot actions by denoising random Gaussian noise conditioned on observations. This approach provides several advantages:

- **Multimodal action distribution**: Can represent complex, multi-modal behaviors
- **Temporal coherence**: Predicts action sequences rather than single actions
- **Robust training**: Stable training dynamics with the diffusion objective

## Architecture

```
Observation Sequence → Observation Encoder (MLP)
                              ↓
                       Global Condition
                              ↓
Random Noise → Conditional U-Net (1D) → Denoised Actions
                     ↑
              Diffusion Timestep
```

### Key Components

1. **Observation Encoder**: Multi-layer perceptron (MLP) that encodes observation sequences
2. **Conditional U-Net 1D**: Denoising network with skip connections and FiLM conditioning
3. **DDPM/DDIM Sampling**: Iterative denoising for action generation

## Directory Structure

```
diffusion_policy/
├── models/
│   ├── conditional_unet1d.py   # 1D U-Net architecture
│   └── diffusion_unet.py       # Full diffusion policy model
├── utils/
│   ├── dataset.py              # Trajectory data loader
│   └── normalizer.py           # Data normalization utilities
├── scripts/
│   ├── train.py                # Training script
│   └── eval_sim.py             # Simulation evaluation script
├── configs/
│   └── default.yaml            # Default configuration
└── README.md                   # This file
```

## Installation

Install required dependencies:

```bash
pip install torch numpy pandas scipy tqdm tensorboard pyyaml
```

For simulation evaluation, you also need the unitree_rl_gym environment (already in parent directory).

## Data Preparation

The policy expects trajectory data in CSV format with the following columns:

### Required columns:
- **Base state**: `pos_x`, `pos_y`, `pos_z`, `quat_w`, `quat_x`, `quat_y`, `quat_z`
- **Base velocity**: `vel_x`, `vel_y`, `vel_z`, `gyro_x`, `gyro_y`, `gyro_z`
- **Joint positions**: `q_0` to `q_28` (29 joints)
- **Joint velocities**: `dq_0` to `dq_28` (29 joints)

### Optional columns (if available):
- **Wrist positions**: `left_wrist_pos_x/y/z`, `right_wrist_pos_x/y/z`
- **Wrist rotations**: `left_wrist_rot_0` to `left_wrist_rot_8`, etc.

Place your preprocessed trajectory CSV files in the `data/` folder.

## Training

### Basic Training

Train with default configuration:

```bash
python diffusion_policy/scripts/train.py \
    --data_dir data \
    --output_dir outputs \
    --batch_size 64 \
    --epochs 500 \
    --lr 1e-4
```

### Advanced Training Options

```bash
python diffusion_policy/scripts/train.py \
    --data_dir data \
    --output_dir outputs \
    --obs_horizon 2 \
    --pred_horizon 16 \
    --action_horizon 8 \
    --num_diffusion_iters 100 \
    --down_dims 256 512 1024 \
    --obs_encoder_layers 256 256 \
    --batch_size 64 \
    --epochs 500 \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    --augment \
    --device cuda:0 \
    --num_workers 4 \
    --save_freq 50
```

### Key Hyperparameters

- `obs_horizon`: Number of past observations to condition on (default: 2)
- `pred_horizon`: Number of future actions to predict (default: 16)
- `action_horizon`: Number of actions to execute before replanning (default: 8)
- `num_diffusion_iters`: Diffusion denoising steps (default: 100)
- `down_dims`: U-Net channel dimensions (default: [256, 512, 1024])

### Training Outputs

Training produces the following outputs in `outputs/diffusion_policy_TIMESTAMP/`:

- `best_model.pt`: Best model checkpoint (lowest validation loss)
- `final_model.pt`: Final model checkpoint
- `checkpoint_epoch_N.pt`: Periodic checkpoints
- `normalizer.npz`: Data normalization statistics
- `logs/`: TensorBoard logs

### Monitor Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir outputs/diffusion_policy_TIMESTAMP/logs
```

## Evaluation in Simulation

Evaluate a trained model in the unitree_rl_gym simulation:

```bash
python diffusion_policy/scripts/eval_sim.py \
    --model_path outputs/diffusion_policy_TIMESTAMP/best_model.pt \
    --normalizer_path outputs/diffusion_policy_TIMESTAMP/normalizer.npz \
    --task g1 \
    --num_episodes 10 \
    --use_ddim \
    --ddim_steps 10
```

### Evaluation Options

- `--use_ddim`: Use DDIM sampling (faster, deterministic)
- `--ddim_steps`: Number of DDIM steps (fewer = faster but less accurate)
- `--num_episodes`: Number of episodes to run
- `--max_steps`: Maximum steps per episode
- `--headless`: Run without visualization
- `--save_results`: Save evaluation statistics

### Sampling Methods

1. **DDPM** (Denoising Diffusion Probabilistic Models):
   - Uses all diffusion timesteps (100 steps by default)
   - Stochastic sampling
   - Slower but theoretically more accurate

2. **DDIM** (Denoising Diffusion Implicit Models):
   - Uses subset of timesteps (10-20 steps typical)
   - Deterministic sampling
   - Much faster, minimal quality loss

For real-time control, use `--use_ddim --ddim_steps 10`.

## Model Details

### Diffusion Process

The diffusion policy uses a forward diffusion process to add noise to actions, and a reverse denoising process to generate actions:

**Forward process** (training):
```
q(x_t | x_0) = N(x_t; √(α_t) * x_0, (1 - α_t) * I)
```

**Reverse process** (inference):
```
p_θ(x_{t-1} | x_t, c) = N(x_{t-1}; μ_θ(x_t, t, c), σ_t * I)
```

where `c` is the observation condition and `θ` are the U-Net parameters.

### Noise Schedule

Uses a cosine schedule for better sample quality:

```python
α_t = cos²((t/T + s) / (1 + s) * π/2)
```

### Normalization

Data is normalized to [-1, 1] range using min-max normalization:

```python
x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
```

## Performance Tips

1. **Batch Size**: Larger batches (64-128) generally improve training stability
2. **Horizons**:
   - Longer `pred_horizon` (16-32) helps with temporal coherence
   - Shorter `action_horizon` (8-16) allows more frequent replanning
3. **Diffusion Steps**:
   - Training: 100 steps provides good coverage
   - Inference: 10-20 DDIM steps is usually sufficient
4. **Data Augmentation**: Small Gaussian noise helps generalization

## Troubleshooting

### Common Issues

1. **Model predicts constant actions**:
   - Check data normalization
   - Verify action diversity in training data
   - Increase learning rate or model capacity

2. **Training loss not decreasing**:
   - Reduce learning rate
   - Check data loading and preprocessing
   - Verify model architecture matches data dimensions

3. **Simulation evaluation fails**:
   - Ensure observation building matches training data format
   - Check normalization statistics are loaded correctly
   - Verify action dimensions match environment

4. **Actions too jerky/unstable**:
   - Increase `action_horizon` for smoother execution
   - Use more DDIM steps during inference
   - Add temporal smoothing to actions

## Citation

If you use this code, please cite:

```bibtex
@article{chi2023diffusionpolicy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  journal={RSS},
  year={2023}
}
```

## License

This implementation is for educational and research purposes.
