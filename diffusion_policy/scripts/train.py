import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
import yaml
import wandb  

import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from diffusion_policy.utils.dataset import G1Dataset
from diffusion_policy.models.policy import DiffusionPolicy
from diffusion_policy.utils.normalizer import Normalizer

def load_config(path=None):
    if path is None:
        script_dir = Path(__file__).parent
        path = script_dir.parent / "configs" / "default.yaml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(path, policy, optimizer, stats, config, epoch, loss):
    torch.save({
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats,
        'config': config,
        'epoch': epoch,
        'loss': loss
    }, path)

def main():

    config = load_config()
    
   
    wandb.init(
        project=config.get("wandb_project", "diffusion_policy"),
        name=config.get("run_name", None),
        config=config
    )


    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")


    train_ds = G1Dataset(
        dataset_root=config["data_root"],
        mode='train',
        obs_horizon=config["obs_horizon"],
        pred_horizon=config["pred_horizon"],
        use_proprio=config["use_proprio"],
        image_resize_size=config.get("image_resize_size", None),
        sample_stride=config["sample_stride"]
    )
    
    val_ds = G1Dataset(
        dataset_root=config["data_root"],
        mode='val',
        obs_horizon=config["obs_horizon"],
        pred_horizon=config["pred_horizon"],
        use_proprio=config["use_proprio"],
        image_resize_size=config.get("image_resize_size", None),
        sample_stride=config["sample_stride"]
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config["num_workers"],
        pin_memory=True
    )


    stats = train_ds.get_normalizer()
    normalizer = Normalizer(stats)


    policy = DiffusionPolicy(
        action_dim=config["action_dim"],
        obs_horizon=config["obs_horizon"],
        pred_horizon=config["pred_horizon"],
        vision_feature_dim=config["vision_feature_dim"],
        proprio_dim=config["proprio_dim"],
        use_proprio=config["use_proprio"]
    ).to(device)


    optimizer = torch.optim.AdamW(policy.parameters(), lr=config["learning_rate"], weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )

    best_val_loss = float('inf')
    global_step = 0


    for epoch in range(config["num_epochs"]):
        policy.train()
        total_train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch = normalizer.normalize(batch)

            
            optimizer.zero_grad()
            loss = policy.compute_loss(batch)
            loss.backward()
            optimizer.step()
            
      
            loss_val = loss.item()
            total_train_loss += loss_val
            global_step += 1
            

            wandb.log({
                "train_loss": loss_val,
                "lr": optimizer.param_groups[0]['lr'],
                "global_step": global_step
            })
            
            loop.set_postfix(loss=loss_val)
        
        avg_train_loss = total_train_loss / len(train_loader)
        

        policy.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch = normalizer.normalize(batch)
                loss = policy.compute_loss(batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()

       
        wandb.log({
            "val_loss": avg_val_loss,
            "avg_train_loss": avg_train_loss,
            "epoch": epoch + 1
        })

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.5f} | Val Loss={avg_val_loss:.5f}")


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(config["checkpoint_dir"], "best.pth")
            save_checkpoint(best_path, policy, optimizer, stats, config, epoch, avg_val_loss)
            
         
            wandb.log({"best_val_loss": best_val_loss})
            print(f"New Best Model Saved (Val Loss: {best_val_loss:.5f})")


        epoch_path = os.path.join(config["checkpoint_dir"], f"epoch_{epoch+1}.pth")
        save_checkpoint(epoch_path, policy, optimizer, stats, config, epoch, avg_val_loss)


    last_path = os.path.join(config["checkpoint_dir"], "last.pth")
    save_checkpoint(last_path, policy, optimizer, stats, config, config["num_epochs"], avg_val_loss)
    
    wandb.finish()


if __name__ == "__main__":
    main()