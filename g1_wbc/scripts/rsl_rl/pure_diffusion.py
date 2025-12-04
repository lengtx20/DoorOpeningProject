import argparse
import sys
import os
import torch
import time
from pathlib import Path

# --- 1. Path Setup (Preserved from original to ensure imports work) ---
# Resolve paths relative to where this script is placed (assuming same depth as original)
script_path = Path(__file__).resolve()
# Adjust these .parents indices if you move this script to a different folder depth
g1_wbc_root = script_path.parents[2] 
project_root = g1_wbc_root.parent

source_path = g1_wbc_root / "source"
if str(source_path) not in sys.path:
    sys.path.insert(0, str(source_path))

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- 2. Imports (Diffusion specific) ---
try:
    from g1_wbc.tasks.manager_based.g1_wbc.agents.diffusion_controller import HierarchicalController
except ImportError as e:
    print(f"[ERROR] Could not import HierarchicalController. Check your python paths.\n{e}")
    sys.exit(1)

# --- 3. Mock Environment ---
# Since we aren't running Isaac Lab, we need a dummy class to satisfy the 
# controller's requirement for an 'env' object.
class MockCommandManager:
    def __init__(self):
        self.commands = {}
    
    # Add methods here if the controller calls set_command or similar
    def set_command(self, name, value):
        self.commands[name] = value

class MockEnv:
    def __init__(self, num_envs=1, device="cuda"):
        self.device = device
        self.num_envs = num_envs
        self.command_manager = MockCommandManager()
        self.step_dt = 0.02 # Standard physics step
        
        # Mock scene/robot if controller accesses them specifically
        self.scene = {
            "robot": type('MockRobot', (), {"device": device})
        }
        
    @property
    def unwrapped(self):
        return self

def main():
    parser = argparse.ArgumentParser(description="Test Diffusion Policy Inference")
    parser.add_argument("--model_path", type=str, default=None, help="Override path to model.pt")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    args = parser.parse_args()

    # --- 4. Configuration ---
    # Default paths from your snippet
    default_diff_dir = project_root / "diffusion_policy/scripts/outputs/door_opening_20251129_193711"
    
    diff_model_path = args.model_path if args.model_path else str(default_diff_dir / "best_model.pt")
    diff_norm_path = str(default_diff_dir / "normalizer.npz")

    print(f"[INFO] Loading Diffusion Model from: {diff_model_path}")
    print(f"[INFO] Loading Normalizer from: {diff_norm_path}")

    # --- 5. Initialization ---
    device = args.device
    env = MockEnv(num_envs=1, device=device)

    try:
        controller = HierarchicalController(
            env,
            model_path=diff_model_path, 
            normalizer_path=diff_norm_path,
            device=device
        )
        print("[INFO] Controller loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load controller: {e}")
        return

    # --- 6. Inference Loop ---
    print("[INFO] Starting Dummy Inference Loop...")
    
    # Run for a few steps to test speed/stability
    for i in range(10):
        start_time = time.time()

        # Generate Dummy Data (simulating the extraction logic in your original script)
        # Shapes based on: l_hand_torso[0], formatted_vel[0]
        state_dict = {
            'l_hand': torch.randn(3, device=device), # Relative pos of left hand
            'r_hand': torch.randn(3, device=device), # Relative pos of right hand
            'vel': torch.randn(3, device=device),    # Base Lin XY + Ang Z
            'grasp': torch.zeros(1, device=device)   # Grasp state
        }
        
        # Step the controller
        # This typically updates the env.command_manager internally
        controller.step(state_dict)
        
        end_time = time.time()
        print(f"Step {i+1}: Inference time = {(end_time - start_time)*1000:.2f} ms")
        
        # Optional: Print result if we knew where controller stored it.
        # e.g., print(env.command_manager.commands)

    print("[INFO] Test complete.")

if __name__ == "__main__":
    main()