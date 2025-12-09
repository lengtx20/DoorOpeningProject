# Door Opening Project - ACT Policy


### 1. Train ACT Policy

```bash
python act_policy/scripts/train.py \
--config act_policy/configs/act_config.yaml

```

### 2. Visualize Validation Predictions

```bash
python act_policy/scripts/visualize_pred.py \
--checkpoint act_policy/checkpoints_stride_1_obs_2_pred_8_stage_5/best.pth \
--mode full_stage 

```

### 3. Run Open-Loop Evaluation

```bash
python g1_wbc/scripts/rsl_rl/replay_act_openloop.py \
--traj_path data/processed_data/traj_55_3 \
--act_checkpoint act_policy/checkpoints_stride_1_obs_2_pred_16_stage_5/best.pth \
--exec_horizon 8 \
--obs_horizon 2 \
--enable_cameras
```

