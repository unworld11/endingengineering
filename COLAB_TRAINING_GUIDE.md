# Google Colab Training Guide (Updated)

## 🎯 Quick Start - Recommended Training Order

### Setup (Run Once)
```python
# 1. Clone your repo or upload files
!git clone https://github.com/YOUR_USERNAME/endingengineering.git
%cd endingengineering

# 2. Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

---

## 📊 Model Options

| Model ID | Architecture | Description | Expected Accuracy |
|----------|-------------|-------------|-------------------|
| **0** | `binput-pg` | Original FracBNN | ~91% |
| **1** | `adaptive-pg` | Ada-FracBNN (Adaptive PG) | ~91% |
| **2** | `adaptive-pg-kd` | Ada-FracBNN + Knowledge Distillation | ~92%+ |

---

## 🚀 Training Scenarios

### Scenario 1: Train Original FracBNN (Baseline)
```bash
# Fast test (10 epochs, ~20-30 min)
!python cifar10.py -id 0 -e 10 -b 128 -s

# Full training (250 epochs, ~8-12 hours)
!python cifar10.py -id 0 -e 250 -b 128 -s
```

**Expected Result:** ~91% accuracy after 250 epochs

---

### Scenario 2: Train Ada-FracBNN (Adaptive PG)
```bash
# Fast test (10 epochs)
!python cifar10.py -id 1 -e 10 -b 128 -ts 0.15 -sw 0.01 -s

# Full training (250 epochs)
!python cifar10.py -id 1 -e 250 -b 128 -ts 0.15 -sw 0.01 -s
```

**Parameters:**
- `-ts 0.15`: Target sparsity (15% channels upgraded to 2-bit)
- `-sw 0.01`: Sparsity regularization weight

**Expected Result:** ~91% accuracy with 15% compute overhead

---

### Scenario 3: Train Ada-FracBNN + KD (Best Performance)

**⚠️ IMPORTANT:** Knowledge Distillation requires a pretrained teacher!

#### Step 3.1: Train Teacher Model (Full Precision ResNet-20)

**Option A: Train from scratch** (Recommended for best KD results)
```bash
# You need to create a separate script for teacher training
# Or train a full-precision ResNet-20 elsewhere and upload teacher.pth
```

**Option B: Use randomly initialized teacher** (Not recommended, but works)
```bash
# The model will warn you but continue training
!python cifar10.py -id 2 -e 10 -b 128 -ts 0.15 -sw 0.01 \
    -temp 4.0 -alpha 0.7 -s
```

#### Step 3.2: Train with Knowledge Distillation
```bash
# With pretrained teacher
!python cifar10.py -id 2 -e 250 -b 128 -ts 0.15 -sw 0.01 \
    -temp 4.0 -alpha 0.7 -tp teacher.pth -s

# OR shorter test (10 epochs)
!python cifar10.py -id 2 -e 10 -b 128 -ts 0.15 -sw 0.01 \
    -temp 4.0 -alpha 0.7 -tp teacher.pth -s
```

**KD Parameters:**
- `-tp teacher.pth`: Path to pretrained teacher weights
- `-temp 4.0`: Temperature for distillation (higher = softer targets)
- `-alpha 0.7`: KD loss weight (0.7 KD + 0.3 CE)

**Expected Result:** ~92%+ accuracy

---

## 💾 Saving Models to Google Drive

### Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Create directory for models
!mkdir -p /content/drive/MyDrive/ada_fracbnn_models
```

### Train with Drive Storage
```bash
# Train and save to Drive
!python cifar10.py -id 1 -e 250 -b 128 -ts 0.15 -sw 0.01 -s

# After training, copy to Drive
!cp save_CIFAR10_model/* /content/drive/MyDrive/ada_fracbnn_models/
```

### Resume Training from Drive
```bash
!python cifar10.py -id 1 -e 250 -b 128 -ts 0.15 -sw 0.01 \
    -r /content/drive/MyDrive/ada_fracbnn_models/model_adaptive-pg.pt
```

---

## 🔧 Key Improvements (Now Fixed!)

### ✅ Input Preprocessing Fix
- **Binary models (id 0, 1, 2):** Use raw [0,1] inputs (no normalization)
- **Teacher model:** Uses normalized inputs automatically
- **You don't need to change anything!** The code now handles this correctly.

### ✅ Knowledge Distillation Fix
- Teacher receives normalized inputs internally
- Student receives unnormalized inputs
- Input mismatch is now automatically handled

### ✅ Batch Size Handling
- `drop_last=True` automatically applied for binary input models
- Handles last batch correctly (e.g., batch_size=128 with 10000 samples)

---

## 📈 Monitoring Training

The training will automatically print:
```
Using raw [0,1] inputs for binary input encoder (no normalization)
Binary Input PG PreAct RPrelu ResNet-20 BNN
Adaptive Binary Input PG PreAct RPrelu ResNet-20 BNN (Ada-FracBNN)
Available GPUs: 1

Epoch: [1]
current learning rate = 0.001
Loss: 1.23 | Acc: 54.2% | Time/batch: 0.15s
...
Accuracy of the network on the 10000 test images: 85.4 %
Average 2-bit fraction: 0.152 (target: 0.150)
Sparsity of the update phase: 84.8 %
The best test accuracy so far: 85.4
```

---

## 🎛️ Advanced Options

### Fine-tuning
```bash
# Resume and fine-tune with lower learning rate
!python cifar10.py -id 1 -e 50 -b 128 -ts 0.15 \
    -r saved_model.pt -f -lr 0.0001
```

### Entropy Regularization (Crisper Gates)
```bash
# Add entropy weight to push gates toward 0 or 1
!python cifar10.py -id 1 -e 250 -b 128 -ts 0.15 -sw 0.01 \
    -ew 0.001 -s
```

### PG-only Feature KD (Experimental)
```bash
# Add feature-level KD on upgraded channels only
!python cifar10.py -id 2 -e 250 -b 128 -ts 0.15 -sw 0.01 \
    -temp 4.0 -alpha 0.7 -tp teacher.pth \
    --pg_kd --pg_kd_weight 0.5 -s
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size
```bash
!python cifar10.py -id 1 -e 250 -b 64 -ts 0.15 -sw 0.01 -s
```

### Issue: "Teacher model not found"
**Solution:** Either train a teacher first or remove `-tp` flag
```bash
# Train without pretrained teacher (will use random init)
!python cifar10.py -id 2 -e 10 -b 128 -ts 0.15 -sw 0.01 \
    -temp 4.0 -alpha 0.7 -s
```

### Issue: Session timeout (Colab free tier)
**Solution:** Use Google Drive to save checkpoints
```python
# Save models periodically
from google.colab import drive
drive.mount('/content/drive')

# Run training with save flag
!python cifar10.py -id 1 -e 250 -b 128 -ts 0.15 -sw 0.01 -s

# Models auto-save to save_CIFAR10_model/ folder
```

---

## 📋 Complete Example Workflow

```python
# === SETUP ===
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repo
!git clone https://github.com/YOUR_USERNAME/endingengineering.git
%cd endingengineering

# 3. Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# === TRAINING ===
# 4. Train Ada-FracBNN (10 epochs for testing)
!python cifar10.py -id 1 -e 10 -b 128 -ts 0.15 -sw 0.01 -s

# 5. Check results
!ls -lh save_CIFAR10_model/

# 6. Copy to Drive
!mkdir -p /content/drive/MyDrive/ada_fracbnn_models
!cp save_CIFAR10_model/* /content/drive/MyDrive/ada_fracbnn_models/

# 7. Test the model
!python cifar10.py -id 1 -t -r save_CIFAR10_model/model_adaptive-pg.pt -b 128
```

---

## 🎯 Recommended Settings for Best Results

### For Quick Testing (1-2 hours)
```bash
!python cifar10.py -id 1 -e 50 -b 128 -ts 0.15 -sw 0.01 -lr 0.001 -s
```
**Expected:** ~85-88% accuracy

### For Full Training (8-12 hours)
```bash
!python cifar10.py -id 1 -e 250 -b 128 -ts 0.15 -sw 0.01 -s
```
**Expected:** ~91% accuracy

### For Best Accuracy with KD (requires pretrained teacher)
```bash
!python cifar10.py -id 2 -e 250 -b 128 -ts 0.15 -sw 0.01 \
    -temp 4.0 -alpha 0.7 -tp teacher.pth -s
```
**Expected:** ~92%+ accuracy

---

## 📚 All Command-Line Arguments

```
Required:
  -id, --model_id        Model ID (0=binput-pg, 1=adaptive-pg, 2=adaptive-pg-kd)
  -e, --num_epoch        Number of training epochs

Common:
  -b, --batch_size       Batch size (default: 128)
  -lr, --init_lr         Initial learning rate (default: 0.001)
  -s, --save            Save the trained model
  -t, --test            Test only mode
  -r, --resume          Resume from checkpoint path

Adaptive PG:
  -ts, --target_sparsity    Target sparsity (0.1-0.2, default: 0.15)
  -sw, --sparsity_weight    Sparsity regularization weight (default: 0.01)
  -ew, --entropy_weight     Entropy regularization weight (default: 0.0)

Knowledge Distillation:
  -tp, --teacher_path       Path to pretrained teacher model
  -temp, --kd_temperature   KD temperature (default: 4.0)
  -alpha, --kd_alpha        KD alpha weight (default: 0.7)
  --pg_kd                   Enable PG-only feature KD
  --pg_kd_weight           PG-KD weight (default: 0.5)

Other:
  -d, --data_dir        Dataset directory (default: /tmp/cifar10_data)
  -gpu, --which_gpus    Which GPUs to use (default: '0')
  -wd, --weight_decay   Weight decay (default: 1e-5)
  -g, --gtarget         PG threshold target (default: 0.0)
```

---

## ✨ What's New?

1. **Fixed input preprocessing bug** - Models now get correct input ranges
2. **Fixed KD input mismatch** - Teacher receives normalized inputs automatically
3. **Better error messages** - Clear warnings and validation
4. **Improved documentation** - Clear input requirements in model docstrings
5. **Performance optimizations** - Cached normalization tensors

Your models should now achieve the expected 91%+ accuracy! 🎉

