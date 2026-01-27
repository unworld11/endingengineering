#!/usr/bin/env python3
"""
Test script to verify the InputEncoder fix works with dynamic batch sizes
"""

import torch
import sys
import os

# Add the project path
sys.path.append('/Users/vedantasp/endingengineering')

from utils.quantization import InputEncoder

def test_input_encoder():
    print("Testing InputEncoder with dynamic batch sizes...")
    
    # Create encoder assuming batch_size=128
    encoder = InputEncoder(input_size=(128, 3, 32, 32), resolution=8)
    
    # Test with different batch sizes
    test_cases = [128, 64, 80, 32, 1]  # Including the problematic size 80
    
    for batch_size in test_cases:
        print(f"\nTesting batch size: {batch_size}")
        
        # Create random input
        x = torch.randn(batch_size, 3, 32, 32)
        
        try:
            # Forward pass
            output = encoder(x)
            expected_channels = encoder.b * 3  # b * c
            
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected channels: {expected_channels}")
            print(f"  ✓ Success!")
            
            # Verify output shape is correct
            assert output.shape == (batch_size, expected_channels, 32, 32), \
                f"Wrong output shape: {output.shape}"
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    print(f"\n✅ All tests passed! InputEncoder now handles dynamic batch sizes.")
    return True

if __name__ == "__main__":
    test_input_encoder()
