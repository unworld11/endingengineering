#!/usr/bin/env python3
"""
Example usage of the new Adaptive FracBNN (Ada-FracBNN) features:

1. Learnable, per-channel fractionalization (Adaptive PG)
2. Knowledge Distillation from compact FP teacher

This script demonstrates how to use the new features implemented in the FracBNN codebase.
"""

import subprocess
import sys
import os

def run_experiment(experiment_name, model_id, additional_args=""):
    """Run an experiment with the given configuration"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {experiment_name}")
    print(f"{'='*60}")
    
    cmd = f"python cifar10.py -id {model_id} -e 50 -b 128 {additional_args}"
    print(f"Command: {cmd}")
    
    # Note: In a real environment, you would run this:
    # subprocess.run(cmd.split(), check=True)
    print("(Command would be executed here)")

def main():
    """Demonstrate the new Ada-FracBNN features"""
    
    print("Ada-FracBNN: Adaptive Fractional Binary Neural Networks")
    print("=" * 60)
    print()
    print("This script demonstrates the new features:")
    print("1. Adaptive PG: Learnable per-channel fractionalization")
    print("2. Knowledge Distillation from compact FP teacher")
    print()
    
    # Experiment 1: Original FracBNN (baseline)
    run_experiment(
        "Baseline FracBNN",
        model_id=0,  # 'binput-pg'
        additional_args="-g 0.0"
    )
    
    # Experiment 2: Adaptive PG (Ada-FracBNN)
    run_experiment(
        "Ada-FracBNN (Adaptive PG)",
        model_id=1,  # 'adaptive-pg'
        additional_args="-ts 0.15 -sw 0.01"
    )
    
    # Experiment 3: Adaptive PG with Knowledge Distillation
    run_experiment(
        "Ada-FracBNN + Knowledge Distillation",
        model_id=2,  # 'adaptive-pg-kd'
        additional_args="-ts 0.15 -sw 0.01 -temp 4.0 -alpha 0.7 -tp teacher_model.pth"
    )
    
    print(f"\n{'='*60}")
    print("PARAMETER EXPLANATIONS")
    print(f"{'='*60}")
    print()
    print("Model IDs:")
    print("  0: 'binput-pg'     - Original FracBNN")
    print("  1: 'adaptive-pg'   - Adaptive PG (Ada-FracBNN)")
    print("  2: 'adaptive-pg-kd' - Adaptive PG + Knowledge Distillation")
    print()
    print("New Parameters:")
    print("  -ts, --target_sparsity: Target sparsity for adaptive PG (0.1-0.2, default: 0.15)")
    print("  -sw, --sparsity_weight: Weight for sparsity regularization (default: 0.01)")
    print("  -temp, --kd_temperature: Temperature for knowledge distillation (default: 4.0)")
    print("  -alpha, --kd_alpha: Alpha for KD loss weighting (default: 0.7)")
    print("  -tp, --teacher_path: Path to pretrained teacher model")
    print()
    print("Key Features:")
    print("1. Learnable Gates: Each output channel has a learnable gate g ∈ [0,1]")
    print("2. Sparsity Regularization: L1 penalty enforces target sparsity (10-20%)")
    print("3. Temperature Annealing: Gates start soft (T=5.0) and become hard (T=1.0)")
    print("4. Knowledge Distillation: Optional KL divergence loss from FP teacher")
    print("5. Automatic Metrics: Reports 2-bit fraction and compute overhead")
    print()
    print("Expected Results:")
    print("- Ada-FracBNN: +0.5-1.0% accuracy vs baseline at same compute budget")
    print("- Ada-FracBNN+KD: +0.5-1.5% additional accuracy improvement")
    print("- Learned sparsity: Network chooses where 2-bit upgrades are most valuable")

if __name__ == "__main__":
    main()
