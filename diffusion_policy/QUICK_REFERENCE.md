# Quick Reference Guide

## 📊 Project Status

✅ **Completed Components:**
1. ✅ Diffusion policy model implementation
2. ✅ Dataset loader for G1 trajectories  
3. ✅ Training pipeline with TensorBoard logging
4. ✅ Simulation evaluation with unitree_rl_gym
5. ✅ Data normalization and preprocessing utilities
6. ✅ Test scripts and documentation
7. ✅ Quick start script

## 🎯 What the System Does

**Input**: Robot trajectory demonstrations (55 CSV files with ~54,000 samples)
**Output**: Trained diffusion policy that can control the G1 robot to walk through doors
**Method**: Diffusion-based behavior cloning with temporal action sequences

## 📁 Key Files

| File | Purpose |
|------|---------|
| `diffusion_policy/scripts/train.py` | Train the policy |
| `diffusion_policy/scripts/eval_sim.py` | Evaluate in simulation |
| `diffusion_policy/test_pipeline.py` | Verify installation |
| `diffusion_policy/quick_start.sh` | Interactive menu |
| `GETTING_STARTED.md` | Step-by-step tutorial |
| `PROJECT_OVERVIEW.md` | Complete documentation |

## ⚡ Common Commands

```bash
# 1. Test everything works
cd diffusion_policy && python test_pipeline.py

# 2. Train (quick test)
python scripts/train.py --data_dir ../data --epochs 10 --batch_size 32 --device cpu

# 3. Train (full, GPU)
python scripts/train.py --data_dir ../data --epochs 500 --device cuda:0

# 4. Monitor training
tensorboard --logdir outputs/

# 5. Evaluate
python scripts/eval_sim.py \
    --model_path outputs/diffusion_policy_*/best_model.pt \
    --normalizer_path outputs/diffusion_policy_*/normalizer.npz
```

## 🔑 Key Numbers

- **Trajectories**: 55 demonstration sequences
- **Samples**: 54,322 training samples
- **Observation dim**: 77 (base state + joints + wrists)
- **Action dim**: 29 (joint positions)
- **Model parameters**: ~38M (default config)
- **Training time**: ~2-4 hours (GPU), ~1-2 days (CPU)

## 🎓 Model Architecture

```
Observations (77-dim) → MLP Encoder (256→256)
                              ↓
                        Global Condition
                              ↓
Noise (29-dim) → Conditional U-Net 1D → Denoised Actions (29-dim)
                     ↑
              Diffusion Timestep
```

## 📈 Expected Performance

| Metric | Target Value |
|--------|--------------|
| Training loss | < 0.01 (after 500 epochs) |
| Validation loss | < 0.02 |
| Samples/sec (GPU) | ~100-200 |
| Inference time (DDIM) | ~20-50 Hz |

## 🔧 Hyperparameters

**Most Important**:
- `--lr 1e-4`: Learning rate (lower if unstable)
- `--batch_size 64`: Batch size (higher = more stable, lower = less memory)
- `--pred_horizon 16`: Action sequence length (longer = smoother)
- `--action_horizon 8`: Replanning frequency (lower = more reactive)

**Model Size**:
- `--down_dims 256 512 1024`: U-Net channels (bigger = more capacity)
- `--obs_encoder_layers 256 256`: Encoder size

**Diffusion**:
- `--num_diffusion_iters 100`: Training denoising steps
- `--ddim_steps 10`: Inference steps (fewer = faster)

## 🐛 Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| OOM error | `--batch_size 16` or `--device cpu` |
| Slow training | `--device cuda:0` and `--num_workers 4` |
| Loss not decreasing | `--lr 5e-5` or `--batch_size 128` |
| Isaac Gym error | Install from https://developer.nvidia.com/isaac-gym |
| No data error | Check `data/` folder has CSV files |

## 📚 Documentation

1. **GETTING_STARTED.md**: Step-by-step tutorial ⭐ Start here!
2. **PROJECT_OVERVIEW.md**: Complete technical overview
3. **diffusion_policy/README.md**: API and model details
4. **This file**: Quick reference

## 🚀 Typical Workflow

1. **Verify setup**: `python test_pipeline.py`
2. **Quick test**: Train 10 epochs to check everything works
3. **Full training**: Train 500 epochs on GPU
4. **Monitor**: Watch TensorBoard for convergence
5. **Evaluate**: Run in simulation to test behavior
6. **Iterate**: Tune hyperparameters if needed

## 💡 Tips

- Start with a **quick test** (10 epochs) before full training
- Use **TensorBoard** to monitor training
- **DDIM sampling** is much faster than DDPM for inference
- **Data augmentation** (`--augment`) helps generalization
- **GPU training** is 10-100x faster than CPU
- Save checkpoints frequently (`--save_freq 50`)

## 📊 What Success Looks Like

**Training**:
- Loss decreases smoothly
- Validation tracks training loss
- No overfitting (train/val gap small)

**Evaluation**:
- Robot moves smoothly (not jerky)
- Follows demonstration patterns
- Successfully navigates to door
- Episode completes without falling

## 🎯 Next Steps

1. Train your first model (10 epochs)
2. Evaluate in simulation
3. Train full model (500 epochs)
4. Tune hyperparameters for your task
5. Deploy to real robot (sim-to-real transfer)
