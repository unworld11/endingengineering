# Ada-FracBNN Testing Notebook Guide

## Overview

The `test_ada_fracbnn.ipynb` notebook provides an interactive environment to test and explore the Adaptive FracBNN (Ada-FracBNN) implementation.

## Getting Started

### Prerequisites

Make sure you have the following installed:
```bash
pip install torch torchvision numpy matplotlib seaborn tqdm jupyter
```

### Launch the Notebook

```bash
jupyter notebook test_ada_fracbnn.ipynb
```

Or if using JupyterLab:
```bash
jupyter lab test_ada_fracbnn.ipynb
```

## Notebook Structure

### 1. Setup and Imports
- Imports all necessary libraries
- Sets up matplotlib and seaborn for visualization
- Verifies CUDA availability

### 2. Configuration
- Defines model and training parameters
- Configure target sparsity (default: 0.15)
- Set knowledge distillation parameters

### 3. Load CIFAR-10 Dataset
- Automatically downloads CIFAR-10 if not present
- Creates train and test data loaders
- Applies appropriate data augmentation

### 4. Model Creation and Testing
- Creates Baseline FracBNN model
- Creates Adaptive PG model with learnable gates
- Tests forward pass to verify models work correctly
- Shows parameter counts

### 5. Analyze Adaptive PG Gates
- Visualizes gate statistics per layer
- Shows 2-bit fraction distribution
- Plots gate mean and standard deviation
- Compares actual sparsity vs target

### 6. Quick Evaluation Functions
- Evaluates model accuracy on test set
- Quick evaluation on subset (first 10 batches) for fast testing
- Can be used with any model

### 7. Summary and Full Training Commands
- Provides command-line examples for full training
- Explains all parameters
- Shows how to run different model configurations

## What Gets Tested

✅ **Model Instantiation**
- Baseline FracBNN (binput-pg)
- Adaptive PG (adaptive-pg)
- Teacher FP model

✅ **Forward Pass**
- Verifies models can process CIFAR-10 images
- Checks output shapes are correct

✅ **Gate Statistics**
- Analyzes learnable gates in Adaptive PG
- Visualizes 2-bit fraction per layer
- Compares target vs actual sparsity

✅ **Quick Evaluation**
- Tests inference on random initialization
- Provides baseline accuracy metrics

## Expected Outputs

### Random Initialization (Untrained)
- Baseline/Adaptive models: ~10% accuracy (random chance for 10 classes)
- Forward pass should complete without errors
- Gate statistics should show roughly uniform distribution

### After Training (250 epochs)
- Baseline FracBNN: ~90-91% accuracy
- Adaptive PG: ~90.5-92% accuracy (+0.5-1% improvement)
- Adaptive PG + KD: ~91-93% accuracy (additional +0.5-1.5%)

## Full Training

For production training, use the command-line interface:

### 1. Baseline FracBNN
```bash
python cifar10.py -id 0 -e 250 -b 128 -g 0.0 -s
```
Expected: ~90-91% accuracy, 250 epochs (~8-12 hours on single GPU)

### 2. Adaptive PG (Ada-FracBNN)
```bash
python cifar10.py -id 1 -e 250 -b 128 -ts 0.15 -sw 0.01 -s
```
Expected: ~90.5-92% accuracy with 15% 2-bit channels

### 3. Adaptive PG + Knowledge Distillation
First train a teacher model:
```bash
# Train FP teacher (optional, you can use any pretrained model)
python cifar10.py -id 0 -e 250 -b 128 -s
```

Then train with KD:
```bash
python cifar10.py -id 2 -e 250 -b 128 -ts 0.15 -sw 0.01 \
    -temp 4.0 -alpha 0.7 -tp save_CIFAR10_model/teacher.pth -s
```
Expected: ~91-93% accuracy

## Parameter Tuning Guide

### Target Sparsity (`-ts`)
- **0.10**: More aggressive sparsity (10% use 2-bit, 90% use 1-bit)
- **0.15**: Balanced (default, recommended)
- **0.20**: Less sparse (20% use 2-bit, 80% use 1-bit)

Higher sparsity = less compute but potentially lower accuracy.

### Sparsity Weight (`-sw`)
- **0.001**: Weak regularization (gates may not reach target)
- **0.01**: Default (recommended)
- **0.1**: Strong regularization (may over-constrain)

Controls how strictly the model enforces target sparsity.

### KD Temperature (`-temp`)
- **1.0**: Hard targets (sharp probability distribution)
- **4.0**: Default (recommended, soft targets)
- **10.0**: Very soft targets (may be too smooth)

Higher temperature = softer probability distributions from teacher.

### KD Alpha (`-alpha`)
- **0.5**: Equal weight to hard and soft targets
- **0.7**: Default (70% soft, 30% hard)
- **0.9**: Mostly soft targets

Balance between learning from teacher (soft) vs ground truth (hard).

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` (try 64 or 32)
- Reduce `num_workers` (try 0 or 1)

### Slow Data Loading
- Increase `num_workers` (try 4 or 8)
- Ensure data is on fast storage (SSD)

### Models Not Training
- Check learning rate (try 1e-4 or 5e-4)
- Verify data augmentation is working
- Check for NaN losses (may need gradient clipping)

### Gate Statistics Not Showing
- Only works with Adaptive PG models (`-id 1` or `-id 2`)
- Requires `adaptive_pg=True` in model creation
- Check model has `get_gate_statistics()` method

## Key Features Demonstrated

1. **Learnable Gates**: Each output channel has a learnable gate g ∈ [0,1]
2. **Sparsity Regularization**: L1 penalty enforces target sparsity (10-20%)
3. **Temperature Annealing**: Gates start soft (T=5.0) and become hard (T=1.0)
4. **Knowledge Distillation**: Optional KL divergence loss from FP teacher
5. **Automatic Metrics**: Reports 2-bit fraction and compute overhead

## Next Steps

After running the notebook:
1. ✅ Verify all models instantiate correctly
2. ✅ Check forward pass works
3. ✅ Analyze gate statistics
4. 🚀 Run full training (250 epochs)
5. 📊 Compare baseline vs adaptive vs KD
6. 🔬 Analyze learned gate patterns
7. ⚡ Measure actual compute savings

## References

- See `example_usage.py` for command-line examples
- See `ADA_FRACBNN_README.md` for detailed documentation
- See `cifar10.py` for full training implementation
- See `model/fracbnn_cifar10.py` for model architecture

## Questions?

If you have any doubts or need clarification, just ask!

