# Batch Size Issue Fix

## Problem

The error occurred because the `InputEncoder` was designed with a fixed batch size assumption, but during training the actual batch size can vary:

1. **Last batch**: When the dataset size doesn't divide evenly by batch_size
2. **Multi-GPU training**: DataParallel splits batches across GPUs
3. **Dynamic batch sizes**: Different batch sizes during validation/testing

```
RuntimeError: The size of tensor a (128) must match the size of tensor b (80) at non-singleton dimension 0
```

## Root Cause

The original `InputEncoder` created a fixed-size `placeholder` tensor during initialization:
```python
# OLD CODE - FIXED SIZE
placeholder = torch.ones(self.n, self.c, self.b, self.h, self.w, dtype=torch.float32)
```

This caused issues when the actual input batch size didn't match `self.n`.

## Solution

### 1. Dynamic InputEncoder

Modified the `InputEncoder` to handle variable batch sizes:

```python
class InputEncoder(nn.Module):
    def __init__(self, input_size, resolution):
        # Store template that can be expanded
        template = torch.arange(self.b, dtype=torch.float32).view(1, 1, -1, 1, 1)
        self.register_buffer('template', template)

    def forward(self, x):
        batch_size = x.size(0)  # Get actual batch size
        # Expand template to match current batch size
        placeholder = self.template.expand(batch_size, self.c, self.b, self.h, self.w)
        # ... rest of forward pass
```

### 2. Correct drop_last Setting

Fixed the `drop_last` logic to apply to all binary input models:

```python
# OLD: drop_last = True if 'binput' in _ARCH else False
# NEW: All our models use binary input encoder
drop_last = True if ('binput' in _ARCH or 'adaptive' in _ARCH) else False
```

## Files Modified

1. **`utils/quantization.py`**: Updated `InputEncoder` class
2. **`cifar10.py`**: Fixed `drop_last` logic

## Testing

Run the test script to verify the fix:
```bash
python test_encoder_fix.py
```

## Usage

The fix is backward compatible. All existing commands will work:

```bash
# Original FracBNN
python cifar10.py -id 0 -e 50 -b 128

# Ada-FracBNN (should now work without batch size errors)
python cifar10.py -id 1 -e 50 -b 128 -ts 0.15 -sw 0.01

# Ada-FracBNN + KD
python cifar10.py -id 2 -e 50 -b 128 -ts 0.15 -sw 0.01 -temp 4.0 -alpha 0.7
```

## Key Benefits

- ✅ Handles variable batch sizes (including last batch)
- ✅ Works with multi-GPU training (DataParallel)
- ✅ Backward compatible with existing code
- ✅ More robust during validation/testing phases
- ✅ Memory efficient (uses expand() instead of creating large tensors)
