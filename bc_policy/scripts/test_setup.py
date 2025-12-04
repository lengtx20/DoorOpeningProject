
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from bc_policy.models import BCPolicy
from bc_policy.utils import BCDataset, get_data_stats, Normalizer
def test_dataset():
    print("\n" + "="*60)
    print("TEST 1: Dataset Loading")
    print("="*60)
    config_path = 'bc_policy/configs/bc_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    try:
        dataset = BCDataset(
            dataset_root=config['data_root'],
            mode='train',
            obs_horizon=config['obs_horizon'],
            pred_horizon=config['pred_horizon'],
            sample_stride=config['sample_stride'],
        )
        print(f"✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
        sample = dataset[0]
        print(f"  Sample proprio shape: {sample['proprio'].shape}")
        print(f"  Sample action shape: {sample['action'].shape}")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        batch = next(iter(dataloader))
        print(f"  Batch proprio shape: {batch['proprio'].shape}")
        print(f"  Batch action shape: {batch['action'].shape}")
        return True, dataset, dataloader, config
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None
def test_model(config):
    print("\n" + "="*60)
    print("TEST 2: Model Instantiation")
    print("="*60)
    try:
        model = BCPolicy(
            proprio_dim=config['proprio_dim'],
            action_dim=config['action_dim'],
            obs_horizon=config['obs_horizon'],
            pred_horizon=config['pred_horizon'],
            hidden_dims=config['hidden_dims'],
            activation=config['activation'],
            use_layer_norm=config['use_layer_norm'],
        )
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {num_params:,}")
        return True, model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None
def test_forward(model, dataloader):
    print("\n" + "="*60)
    print("TEST 3: Forward Pass")
    print("="*60)
    try:
        batch = next(iter(dataloader))
        output = model(batch['proprio'])
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {batch['proprio'].shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        loss = model.compute_loss(batch)
        print(f"  Loss: {loss.item():.6f}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
def test_normalization(dataloader):
    print("\n" + "="*60)
    print("TEST 4: Normalization")
    print("="*60)
    try:
        batch = next(iter(dataloader))
        proprio = batch['proprio'].reshape(-1, 29)
        action = batch['action'].reshape(-1, 10)
        stats = {
            'proprio': {
                'min': proprio.min(dim=0)[0],
                'max': proprio.max(dim=0)[0],
                'mean': proprio.mean(dim=0),
                'std': proprio.std(dim=0),
            },
            'action': {
                'min': action.min(dim=0)[0],
                'max': action.max(dim=0)[0],
                'mean': action.mean(dim=0),
                'std': action.std(dim=0),
            }
        }
        normalizer = Normalizer(stats)
        print(f"✓ Normalizer created")
        batch_norm = normalizer.normalize(batch)
        print(f"  Normalized proprio range: [{batch_norm['proprio'].min():.3f}, {batch_norm['proprio'].max():.3f}]")
        print(f"  Normalized action range: [{batch_norm['action'].min():.3f}, {batch_norm['action'].max():.3f}]")
        action_denorm = normalizer.denormalize(batch_norm['action'], 'action')
        print(f"  Denormalized action range: [{action_denorm.min():.3f}, {action_denorm.max():.3f}]")
        print(f"  Reconstruction error: {(action_denorm - batch['action']).abs().max():.6f}")
        return True, normalizer
    except Exception as e:
        print(f"✗ Normalization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None
def test_training(model, dataloader, normalizer, config):
    print("\n" + "="*60)
    print("TEST 5: Training Loop (1 epoch)")
    print("="*60)
    try:
        device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"  Using device: {device}")
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        model.train()
        total_loss = 0.0
        num_batches = 0
        for i, batch in enumerate(dataloader):
            if i >= 5:  
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            batch = normalizer.normalize(batch)
            loss = model.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            print(f"  Batch {i+1}/5: loss = {loss.item():.6f}")
        avg_loss = total_loss / num_batches
        print(f"✓ Training loop successful")
        print(f"  Average loss: {avg_loss:.6f}")
        return True
    except Exception as e:
        print(f"✗ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False
def main():
    print("\n" + "="*60)
    print("BC POLICY SETUP TEST")
    print("="*60)
    all_passed = True
    success, dataset, dataloader, config = test_dataset()
    if not success:
        print("\n✗ Setup test FAILED at dataset loading")
        return
    all_passed &= success
    success, model = test_model(config)
    if not success:
        print("\n✗ Setup test FAILED at model creation")
        return
    all_passed &= success
    success = test_forward(model, dataloader)
    if not success:
        print("\n✗ Setup test FAILED at forward pass")
        return
    all_passed &= success
    success, normalizer = test_normalization(dataloader)
    if not success:
        print("\n✗ Setup test FAILED at normalization")
        return
    all_passed &= success
    success = test_training(model, dataloader, normalizer, config)
    if not success:
        print("\n✗ Setup test FAILED at training loop")
        return
    all_passed &= success
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYou can now start training with:")
        print("  python bc_policy/scripts/train.py --config bc_policy/configs/bc_config.yaml")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
if __name__ == "__main__":
    main()
