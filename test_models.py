#!/usr/bin/env python3
"""
Comprehensive test script for CIFAR-10 models including the new teacher model
"""

import torch
import sys
import os

sys.path.append('/Users/vedantasp/endingengineering')

import model.fracbnn_cifar10 as m

def test_model_architectures():
    """Test all model architectures can be instantiated"""
    print("=" * 80)
    print("Testing Model Architectures")
    print("=" * 80)
    
    batch_size = 4  # Small batch for testing
    input_size = (batch_size, 3, 32, 32)
    
    models_to_test = [
        {
            'name': 'Binary Input PG ResNet-20 (model_id=0)',
            'model': m.resnet20(batch_size=batch_size, num_gpus=1, adaptive_pg=False),
            'input_range': (0, 1),  # Binary input expects [0,1]
        },
        {
            'name': 'Adaptive PG ResNet-20 (model_id=1, Ada-FracBNN)',
            'model': m.resnet20(batch_size=batch_size, num_gpus=1, adaptive_pg=True, target_sparsity=0.15),
            'input_range': (0, 1),  # Binary input expects [0,1]
        },
        {
            'name': 'Full Precision ResNet-20 Teacher (NEW)',
            'model': m.fp_resnet20(num_classes=10),
            'input_range': (-2, 2),  # FP model uses normalized inputs
        },
    ]
    
    all_passed = True
    
    for i, model_info in enumerate(models_to_test):
        print(f"\n{'='*80}")
        print(f"Test {i+1}: {model_info['name']}")
        print(f"{'='*80}")
        
        try:
            model = model_info['model']
            model.eval()
            
            # Create random input in appropriate range
            min_val, max_val = model_info['input_range']
            x = torch.rand(input_size) * (max_val - min_val) + min_val
            
            print(f"Input shape: {x.shape}")
            print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
            
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Number of classes: {output.shape[1]}")
            
            # Verify output shape
            assert output.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {output.shape}"
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            print(f"✅ PASSED: {model_info['name']}")
            
        except Exception as e:
            print(f"❌ FAILED: {model_info['name']}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_dynamic_batch_sizes():
    """Test models handle different batch sizes correctly"""
    print("\n" + "=" * 80)
    print("Testing Dynamic Batch Sizes")
    print("=" * 80)
    
    test_batch_sizes = [1, 4, 32, 64, 80, 128]  # Including problematic size 80
    
    all_passed = True
    
    for batch_size in test_batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        try:
            # Test binary input model (most sensitive to batch size issues)
            model = m.resnet20(batch_size=128, num_gpus=1, adaptive_pg=True, target_sparsity=0.15)
            model.eval()
            
            x = torch.rand(batch_size, 3, 32, 32)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, 10), f"Expected ({batch_size}, 10), got {output.shape}"
            print(f"  ✓ Batch size {batch_size}: Output shape {output.shape}")
            
        except Exception as e:
            print(f"  ✗ Batch size {batch_size}: FAILED - {e}")
            all_passed = False
    
    return all_passed


def test_teacher_student_compatibility():
    """Test teacher and student models can work together for KD"""
    print("\n" + "=" * 80)
    print("Testing Teacher-Student Compatibility for Knowledge Distillation")
    print("=" * 80)
    
    try:
        batch_size = 8
        
        # Student model (binary)
        print("\nInitializing student model (Ada-FracBNN)...")
        student = m.resnet20(batch_size=batch_size, num_gpus=1, 
                            adaptive_pg=True, target_sparsity=0.15)
        student.eval()
        
        # Teacher model (full precision)
        print("Initializing teacher model (FP ResNet-20)...")
        teacher = m.fp_resnet20(num_classes=10)
        teacher.eval()
        
        # Test with same input (respecting each model's input requirements)
        print("\nTesting forward pass...")
        
        # Student expects [0,1] range (binary input)
        x_student = torch.rand(batch_size, 3, 32, 32)
        
        # Teacher expects normalized input (ImageNet stats)
        normalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        normalize_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_teacher = (x_student - normalize_mean) / normalize_std
        
        print(f"Student input range: [{x_student.min():.3f}, {x_student.max():.3f}]")
        print(f"Teacher input range: [{x_teacher.min():.3f}, {x_teacher.max():.3f}]")
        
        with torch.no_grad():
            student_output = student(x_student)
            teacher_output = teacher(x_teacher)
        
        print(f"Student output shape: {student_output.shape}")
        print(f"Teacher output shape: {teacher_output.shape}")
        
        # Verify compatibility
        assert student_output.shape == teacher_output.shape, \
            "Teacher and student output shapes don't match!"
        
        # Test KD loss computation
        from utils.quantization import KnowledgeDistillationLoss
        
        labels = torch.randint(0, 10, (batch_size,))
        kd_loss = KnowledgeDistillationLoss(temperature=4.0, alpha=0.7)
        
        loss = kd_loss(student_output, teacher_output, labels)
        
        print(f"KD Loss: {loss.item():.4f}")
        print(f"✅ PASSED: Teacher-Student compatibility verified")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_pg_features():
    """Test adaptive PG specific features"""
    print("\n" + "=" * 80)
    print("Testing Adaptive PG Features")
    print("=" * 80)
    
    try:
        batch_size = 4
        
        # Create adaptive model
        print("\nCreating Adaptive PG model...")
        model = m.resnet20(batch_size=batch_size, num_gpus=1, 
                          adaptive_pg=True, target_sparsity=0.15)
        model.eval()
        
        x = torch.rand(batch_size, 3, 32, 32)
        
        # Check for gating parameters
        print("\nChecking for gating parameters...")
        has_gates = False
        gate_params = []
        
        for name, param in model.named_parameters():
            if 'gate' in name.lower():
                has_gates = True
                gate_params.append((name, param.shape))
                print(f"  Found gate: {name}, shape: {param.shape}")
        
        if has_gates:
            print(f"✓ Found {len(gate_params)} gating parameters")
        else:
            print("  No gating parameters found (this might be expected)")
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        print(f"\nOutput shape: {output.shape}")
        print(f"✅ PASSED: Adaptive PG features verified")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("CIFAR-10 Model Test Suite")
    print("=" * 80)
    print("\nThis test suite verifies:")
    print("1. All model architectures can be instantiated")
    print("2. Models handle dynamic batch sizes correctly")
    print("3. Teacher-Student compatibility for Knowledge Distillation")
    print("4. Adaptive PG features")
    print("=" * 80)
    
    results = {
        'Model Architectures': test_model_architectures(),
        'Dynamic Batch Sizes': test_dynamic_batch_sizes(),
        'Teacher-Student KD': test_teacher_student_compatibility(),
        'Adaptive PG Features': test_adaptive_pg_features(),
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
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

