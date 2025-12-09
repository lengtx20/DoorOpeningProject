# Diffusion Policy for Door Opening

Implementation of **Diffusion Policy** for the door opening task with the G1 humanoid robot.

## Overview

Diffusion Policy uses denoising diffusion probabilistic models to generate action sequences. Instead of directly predicting actions, it learns to denoise random noise into valid action trajectories conditioned on observations.

**Key Features:**
- Iterative denoising process for action generation
- Multi-modal behavior capability
- Smooth, stable action trajectories
- State-of-the-art performance on manipulation tasks

## Installation

### Required Dependencies

```bash
# Install diffusers library (required for noise schedulers)
pip install diffusers

# Other dependencies (should already be installed)
pip install torch numpy pyyaml tqdm matplotlib tensorboard
```

## Architecture

```
Observations (proprio) → Observation Encoder → Latent
                                                   ↓
Random Noise ─────────────────────→ U-Net Denoiser → Predicted Actions
      ↑                                    ↑
      └────── Iterative Denoising ─────────┘
```

### Components

1. **Observation Encoder**: MLP that maps observations to latent space
2. **Conditional U-Net**: 1D convolutional network that predicts noise
3. **Diffusion Process**: DDPM training + DDIM inference
4. **FiLM Conditioning**: Feature-wise affine modulation for obs conditioning

### Model Configuration

- **Input**: Proprio (29-dim) × obs_horizon (2)
- **Output**: Action chunk (16 steps × 10-dim actions)
- **Obs embedding**: 256-dim
- **U-Net channels**: [256, 512, 1024]
- **Training steps**: 100 denoising iterations
- **Inference steps**: 10 denoising iterations (DDIM for speed)

## Directory Structure

```
diffusion_policy/
├── configs/
│   └── diffusion_config.yaml         # Training configuration
├── models/
│   ├── __init__.py
│   └── diffusion_model.py             # Diffusion model implementation
├── scripts/
│   ├── train.py                       # Training script
│   └── eval.py                        # Evaluation script
├── utils/
│   ├── __init__.py
│   └── diffusion_controller.py        # Inference controller
├── checkpoints/                       # Saved models (created during training)
└── README.md                          # This file
```

## Usage

### 1. Training

Train the diffusion policy from scratch:

```bash
cd /home/jason/DoorOpeningProject
python diffusion_policy/scripts/train.py --config diffusion_policy/configs/diffusion_config.yaml
```

Resume training from a checkpoint:

```bash
python diffusion_policy/scripts/train.py \
    --config diffusion_policy/configs/diffusion_config.yaml \
    --resume diffusion_policy/checkpoints/last.pth
```

### 2. Evaluation

Evaluate a trained model on the validation set:

```bash
python diffusion_policy/scripts/eval.py \
    --checkpoint diffusion_policy/checkpoints/best.pth \
    --batch_size 32
```

### 3. Deployment (Robot/Simulation)

Use the diffusion controller for real-time inference:

```python
from diffusion_policy.utils import DiffusionController, load_diffusion_policy_and_normalizer

# Load model and normalizer
model, normalizer, config = load_diffusion_policy_and_normalizer(
    checkpoint_path='diffusion_policy/checkpoints/best.pth',
    device='cuda:0'
)

# Create controller
controller = DiffusionController(
    model=model,
    normalizer=normalizer,
    obs_horizon=2,
    pred_horizon=16,
    action_horizon=8,  # Replan every 8 steps
    device='cuda:0',
)

# In control loop
for t in range(num_steps):
    # Get current observation
    proprio = get_robot_state()  # [29,] tensor

    # Get action (denoising happens internally)
    action = controller.get_action(proprio)  # [10,] tensor

    # Execute action
    execute_action(action)
```

### 4. Isaac Lab Deployment

Deploy via replay.py:

```bash
cd g1_wbc/scripts/rsl_rl

# Run with Diffusion policy
python replay.py \
    --mode diffusion \
    --diffusion_checkpoint /home/jason/DoorOpeningProject/diffusion_policy/checkpoints/best.pth \
    --num_envs 1

# With video recording
python replay.py \
    --mode diffusion \
    --diffusion_checkpoint /home/jason/DoorOpeningProject/diffusion_policy/checkpoints/best.pth \
    --video \
    --video_length 500 \
    --num_envs 1
```

## Configuration

Key hyperparameters in `diffusion_config.yaml`:

```yaml
# Model architecture
obs_embedding_dim: 256
down_dims: [256, 512, 1024]
kernel_size: 5
n_groups: 8

# Diffusion process
num_diffusion_iters: 100      # Training denoising steps
num_inference_steps: 10        # Inference denoising steps (DDIM)
beta_schedule: "squaredcos_cap_v2"

# Action chunking
obs_horizon: 2
pred_horizon: 16
action_horizon: 8

# Training
batch_size: 64
learning_rate: 1.0e-4
num_epochs: 300
ema_decay: 0.995              # Exponential moving average
```

## How Diffusion Policy Works

### Training

1. **Forward diffusion**: Add noise to ground truth actions
2. **Noise prediction**: Train U-Net to predict the added noise
3. **Loss**: MSE between predicted and actual noise

### Inference

1. **Start**: Sample random noise
2. **Denoise**: Iteratively denoise using trained U-Net
3. **Output**: Clean action trajectory after all denoising steps

### Key Advantages

1. **Multi-modality**: Can generate diverse behaviors
2. **Stability**: Iterative refinement produces smooth actions
3. **Expressiveness**: Can represent complex action distributions
4. **Robustness**: Less prone to overfitting than direct prediction

## Comparison with Other Policies

| Feature | BC | ACT | Diffusion |
|---------|----|----|-----------|
| Architecture | MLP | Transformer | U-Net |
| Action Prediction | Direct | Direct (chunked) | Iterative denoising |
| Stochasticity | No | VAE | Inherent |
| Temporal Model | None | Self-attention | Convolution |
| Parameters | ~1M | ~7.6M | ~15-20M |
| Training Time | Fast | Medium | Slow |
| Inference Time | Very Fast (~1ms) | Fast (~5ms) | Slow (~50-100ms) |
| Multi-modal | No | Yes | Yes |
| Best For | Baselines | General use | Complex tasks |

## Training Tips

1. **Denoising Steps**:
   - Training: 100 steps (good quality)
   - Inference: 10 steps (DDIM for speed)

2. **EMA**: Use exponential moving average of weights (0.995) for better stability

3. **Batch Size**: Larger is better (64) for stable training

4. **Learning Rate**: Lower than BC/ACT (1e-4) due to complexity

5. **Training Time**: Expect 4-6 hours for 300 epochs on single GPU

## Performance Monitoring

Monitor these metrics during training:

1. **Training Loss**: Should steadily decrease
2. **Validation Loss**: Check for overfitting
3. **Action Statistics**: Ensure reasonable ranges

Use TensorBoard:

```bash
tensorboard --logdir diffusion_policy/checkpoints/tensorboard
```

## Troubleshooting

### Issue: Slow inference
**Cause**: Too many denoising steps

**Solution**: Reduce `num_inference_steps` (try 5-10)

### Issue: Poor action quality
**Cause**: Not enough denoising steps

**Solution**: Increase `num_inference_steps` or `num_diffusion_iters`

### Issue: OOM during training
**Cause**: Large model + large batch

**Solution**: Reduce `batch_size` or `down_dims`

### Issue: Training instability
**Cause**: Learning rate too high

**Solution**: Reduce `learning_rate` to 5e-5

## Citation

If you use this implementation, please cite:

```bibtex
@article{chi2023diffusionpolicy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  journal={Robotics: Science and Systems (RSS)},
  year={2023}
}
```

## References

- [Diffusion Policy Paper](https://arxiv.org/abs/2303.04137)
- [Project Page](https://diffusion-policy.cs.columbia.edu/)
- [Original Implementation](https://github.com/real-stanford/diffusion_policy)
- [Diffusers Library](https://github.com/huggingface/diffusers)

## Notes

- **Inference Speed**: Diffusion is slower than BC/ACT due to iterative denoising
- **Quality**: Often produces smoother, more stable actions
- **Use Case**: Best for complex manipulation requiring multi-modal behaviors
- **Recommendation**: Start with ACT, use Diffusion if you need better quality/multi-modality
