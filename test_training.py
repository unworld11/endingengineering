#!/usr/bin/env python3
"""
Quick training test to verify models can actually train
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys

sys.path.append('/Users/vedantasp/endingengineering')

import model.fracbnn_cifar10 as m
from utils.quantization import KnowledgeDistillationLoss

def quick_training_test():
    """Test that models can actually train for a few iterations"""
    print("=" * 80)
    print("Quick Training Test")
    print("=" * 80)
    
    # Prepare small dataset
    print("\nPreparing mini-dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    trainset = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10_data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    # Use only 32 samples for quick test
    trainset = torch.utils.data.Subset(trainset, range(32))
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=8,
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test cases
    test_cases = [
        {
            'name': 'Binary Input PG ResNet-20',
            'model': m.resnet20(batch_size=8, num_gpus=1, adaptive_pg=False),
            'use_kd': False,
        },
        {
            'name': 'Adaptive PG ResNet-20 (Ada-FracBNN)',
            'model': m.resnet20(batch_size=8, num_gpus=1, adaptive_pg=True, target_sparsity=0.15),
            'use_kd': False,
        },
        {
            'name': 'Adaptive PG with Knowledge Distillation',
            'model': m.resnet20(batch_size=8, num_gpus=1, adaptive_pg=True, target_sparsity=0.15),
            'use_kd': True,
        },
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing: {test_case['name']}")
        print(f"{'='*80}")
        
        try:
            model = test_case['model'].to(device)
            model.train()
            
            # Setup optimizer
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Setup teacher if using KD
            teacher_model = None
            kd_criterion = None
            
            if test_case['use_kd']:
                print("Setting up teacher model for KD...")
                teacher_model = m.fp_resnet20(num_classes=10).to(device)
                teacher_model.eval()
                kd_criterion = KnowledgeDistillationLoss(temperature=4.0, alpha=0.7)
                
                # Normalize transform for teacher
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            
            # Training loop
            num_iters = 3
            losses = []
            
            print(f"\nTraining for {num_iters} iterations...")
            
            for iteration, (inputs, labels) in enumerate(trainloader):
                if iteration >= num_iters:
                    break
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Compute loss
                if test_case['use_kd'] and teacher_model is not None:
                    # Prepare normalized input for teacher
                    inputs_normalized = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )(inputs)
                    
                    with torch.no_grad():
                        teacher_outputs = teacher_model(inputs_normalized)
                    
                    loss = kd_criterion(outputs, teacher_outputs, labels)
                else:
                    loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                print(f"  Iteration {iteration + 1}/{num_iters}: Loss = {loss.item():.4f}")
            
            # Verify loss is finite
            avg_loss = sum(losses) / len(losses)
            print(f"\nAverage loss: {avg_loss:.4f}")
            
            assert all(torch.isfinite(torch.tensor(l)) for l in losses), "Loss became NaN or Inf!"
            
            print(f"✅ PASSED: {test_case['name']}")
            
        except Exception as e:
            print(f"❌ FAILED: {test_case['name']}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_inference_speed():
    """Test inference speed of different models"""
    print("\n" + "=" * 80)
    print("Inference Speed Comparison")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_runs = 10
    
    models = {
        'Binary Input PG': m.resnet20(batch_size=batch_size, num_gpus=1, adaptive_pg=False),
        'Adaptive PG': m.resnet20(batch_size=batch_size, num_gpus=1, adaptive_pg=True, target_sparsity=0.15),
        'FP Teacher': m.fp_resnet20(num_classes=10),
    }
    
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Number of runs: {num_runs}")
    print(f"Device: {device}")
    print()
    
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            _ = model(x)
        
        # Timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
        
        print(f"{name:.<40} {avg_time:>8.2f} ms/batch")
    
    return True


def main():
    print("=" * 80)
    print("Training and Inference Test Suite")
    print("=" * 80)
    
    results = {
        'Quick Training Test': quick_training_test(),
        'Inference Speed': test_inference_speed(),
    }
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(results.values())
    
    print("=" * 80)
    if all_passed:
        print("🎉 ALL TRAINING TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

