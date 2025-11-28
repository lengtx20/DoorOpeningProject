#!/bin/bash
# Quick start script for diffusion policy training and evaluation

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Diffusion Policy Quick Start${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if data directory exists
if [ ! -d "../data" ]; then
    echo -e "${RED}Error: data/ directory not found!${NC}"
    echo "Please run the preprocessing script first to generate data."
    exit 1
fi

# Count CSV files
num_files=$(ls -1 ../data/merged_50hz_log*.csv 2>/dev/null | wc -l)
if [ "$num_files" -eq 0 ]; then
    echo -e "${RED}Error: No trajectory CSV files found in data/!${NC}"
    echo "Please run the preprocessing script first."
    exit 1
fi

echo -e "${GREEN}Found $num_files trajectory files in data/${NC}"
echo ""

# Menu
echo "Select an option:"
echo "1) Train diffusion policy (default settings)"
echo "2) Train diffusion policy (custom settings)"
echo "3) Evaluate trained model in simulation"
echo "4) Quick test (train for 10 epochs)"
echo "5) Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo -e "${BLUE}Starting training with default settings...${NC}"
        python scripts/train.py \
            --data_dir ../data \
            --output_dir outputs \
            --batch_size 64 \
            --epochs 500 \
            --lr 1e-4 \
            --augment \
            --device cuda:0
        ;;
    2)
        echo -e "${BLUE}Custom training settings:${NC}"
        read -p "Batch size [64]: " batch_size
        batch_size=${batch_size:-64}

        read -p "Epochs [500]: " epochs
        epochs=${epochs:-500}

        read -p "Learning rate [1e-4]: " lr
        lr=${lr:-1e-4}

        read -p "Device [cuda:0]: " device
        device=${device:-cuda:0}

        python scripts/train.py \
            --data_dir ../data \
            --output_dir outputs \
            --batch_size $batch_size \
            --epochs $epochs \
            --lr $lr \
            --augment \
            --device $device
        ;;
    3)
        echo -e "${BLUE}Available model checkpoints:${NC}"
        find outputs -name "best_model.pt" -o -name "final_model.pt" | nl
        echo ""
        read -p "Enter path to model checkpoint: " model_path

        # Find normalizer in same directory
        model_dir=$(dirname "$model_path")
        normalizer_path="$model_dir/normalizer.npz"

        if [ ! -f "$normalizer_path" ]; then
            echo -e "${RED}Error: normalizer.npz not found in model directory!${NC}"
            exit 1
        fi

        read -p "Number of episodes [10]: " num_episodes
        num_episodes=${num_episodes:-10}

        read -p "Headless mode? [y/N]: " headless
        headless_flag=""
        if [ "$headless" = "y" ] || [ "$headless" = "Y" ]; then
            headless_flag="--headless"
        fi

        echo -e "${GREEN}Starting evaluation...${NC}"
        python scripts/eval_sim.py \
            --model_path "$model_path" \
            --normalizer_path "$normalizer_path" \
            --num_episodes $num_episodes \
            --use_ddim \
            --ddim_steps 10 \
            $headless_flag \
            --save_results
        ;;
    4)
        echo -e "${BLUE}Quick test (10 epochs)...${NC}"
        python scripts/train.py \
            --data_dir ../data \
            --output_dir outputs \
            --batch_size 32 \
            --epochs 10 \
            --lr 1e-4 \
            --device cuda:0
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice!${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
