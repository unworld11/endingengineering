# Running Ada-FracBNN in Google Colab

This guide shows how to run the `test_ada_fracbnn.ipynb` notebook in Google Colab.

## Quick Start (3 Methods)

### Method 1: Direct Upload to Colab (Easiest)

1. **Go to Google Colab**: https://colab.research.google.com/

2. **Upload notebook**:
   - Click `File` → `Upload notebook`
   - Select `test_ada_fracbnn.ipynb` from your local machine

3. **Upload project files**:
   - In Colab, add a new code cell at the top:
   ```python
   # Upload all project files
   from google.colab import files
   import zipfile
   import os
   
   # Upload your project as a zip file
   uploaded = files.upload()  # Select your zipped project folder
   
   # Extract
   for filename in uploaded.keys():
       if filename.endswith('.zip'):
           with zipfile.ZipFile(filename, 'r') as zip_ref:
               zip_ref.extractall('.')
   
   # List contents
   !ls -la
   ```

4. **Run the notebook cells** as normal

### Method 2: Clone from GitHub (Recommended)

1. **Go to Google Colab**: https://colab.research.google.com/

2. **Add setup cell** at the very top of the notebook:
   ```python
   # Clone your repository (replace with your repo URL)
   !git clone https://github.com/YOUR_USERNAME/endingengineering.git
   %cd endingengineering
   
   # Install dependencies
   !pip install -q tqdm seaborn
   
   # Verify GPU
   import torch
   print("PyTorch version:", torch.__version__)
   print("CUDA available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print("GPU:", torch.cuda.get_device_name(0))
   ```

3. **Run the notebook cells**

### Method 3: Direct Link (If repo is public)

Share this link with others (replace with your GitHub username/repo):
```
https://colab.research.google.com/github/YOUR_USERNAME/endingengineering/blob/main/test_ada_fracbnn.ipynb
```

## Detailed Setup Instructions

### Step 1: Enable GPU Runtime

**Important**: Enable GPU for faster training!

1. Click `Runtime` → `Change runtime type`
2. Select `GPU` from the Hardware accelerator dropdown
3. Choose `T4` or `A100` (if available)
4. Click `Save`

### Step 2: Add Colab Setup Cell

Add this as the **FIRST CELL** in your notebook:

```python
# ============================================
# GOOGLE COLAB SETUP
# ============================================

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("✓ Running in Google Colab")
except:
    IN_COLAB = False
    print("✓ Running locally")

if IN_COLAB:
    # Clone repository (replace with your repo URL)
    !git clone https://github.com/YOUR_USERNAME/endingengineering.git
    
    # Change to project directory
    %cd endingengineering
    
    # Install additional dependencies
    !pip install -q tqdm seaborn
    
    print("\n✓ Setup complete!")
    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device:", torch.cuda.get_device_name(0))
```

### Step 3: Modify Configuration

Update the `config` dictionary in Cell 2 for Colab:

```python
# Configuration
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 256 if torch.cuda.is_available() else 128,  # Larger batch for Colab GPU
    'num_workers': 2,  # Colab works well with 2
    'data_dir': '/content/data/cifar10',  # Colab data directory
    
    # Adaptive PG parameters
    'target_sparsity': 0.15,
    'sparsity_weight': 0.01,
    
    # Knowledge Distillation parameters
    'kd_temperature': 4.0,
    'kd_alpha': 0.7,
    
    # Training parameters
    'learning_rate': 1e-3,
    'num_epochs': 5,
}

print("Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
```

### Step 4: Run All Cells

Click `Runtime` → `Run all` or run cells sequentially.

## Important Colab Considerations

### 1. Runtime Limits
- **Free Tier**: ~12 hours, then disconnects
- **Colab Pro**: Longer sessions, priority GPU access
- **Solution**: Save models frequently with checkpoint code:

```python
# Add this to save checkpoints
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Save model
    save_path = '/content/drive/MyDrive/ada_fracbnn_models/'
    import os
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), save_path + 'model.pth')
```

### 2. Data Persistence
CIFAR-10 data downloads every session. To persist:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Use Drive for data
config['data_dir'] = '/content/drive/MyDrive/cifar10_data'
```

### 3. File Uploads (Alternative to Git)

If you don't have a GitHub repo yet:

```python
from google.colab import files
import zipfile

# Upload project zip
print("Please upload your project as a .zip file:")
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/endingengineering')

%cd /content/endingengineering
!ls -la
```

### 4. Visualization in Colab

Matplotlib plots work automatically in Colab! Just run the cells with visualization code.

## Full Training in Colab

For full 250-epoch training:

```python
# Add this cell for full training
!python cifar10.py -id 1 -e 250 -b 256 -ts 0.15 -sw 0.01 -s

# Monitor training in real-time
# Note: Colab may disconnect after 12 hours (free tier)
```

### Training with Checkpointing

```python
# Train with periodic saving
!python cifar10.py -id 1 -e 250 -b 256 -ts 0.15 -sw 0.01 -s -r checkpoint.pth

# If disconnected, resume with:
!python cifar10.py -id 1 -e 250 -b 256 -ts 0.15 -sw 0.01 -s -r save_CIFAR10_model/adaptive-pg.pth
```

## Downloading Results

Download trained models and results:

```python
from google.colab import files

# Download model
files.download('save_CIFAR10_model/adaptive-pg.pth')

# Download all results as zip
!zip -r results.zip save_CIFAR10_model/
files.download('results.zip')
```

## Common Issues & Solutions

### Issue 1: "Module not found" errors
```python
# Add at the top of notebook
import sys
sys.path.insert(0, '/content/endingengineering')
```

### Issue 2: GPU not available
- Check: `Runtime` → `Change runtime type` → Select `GPU`
- Verify with: `torch.cuda.is_available()`

### Issue 3: Out of memory
```python
# Reduce batch size in config
config['batch_size'] = 128  # or 64
```

### Issue 4: Colab disconnects
- Enable: `Tools` → `Settings` → `Miscellaneous` → `Automatically reconnect`
- Use checkpointing (save every N epochs)

### Issue 5: Slow data loading
```python
# Reduce num_workers in Colab
config['num_workers'] = 2  # Colab optimal
```

## Pro Tips

### 1. Use Colab Pro for Long Training
- Longer sessions (24h+)
- Better GPUs (A100, V100)
- Background execution
- Cost: ~$10/month

### 2. Use Tensor Board
```python
# Add to training code
%load_ext tensorboard
%tensorboard --logdir runs/

# In training loop, log metrics
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('Loss/train', loss, epoch)
```

### 3. Save to Google Drive Automatically
```python
# Mount drive first
from google.colab import drive
drive.mount('/content/drive')

# Modify save path in cifar10.py or use symbolic link
!ln -s /content/drive/MyDrive/ada_fracbnn_saves /content/endingengineering/save_CIFAR10_model
```

### 4. Monitor GPU Usage
```python
# Check GPU memory
!nvidia-smi

# Check in Python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

### 5. Parallel Experiments
Open multiple Colab tabs to run different configurations simultaneously:
- Tab 1: Baseline model
- Tab 2: Adaptive PG
- Tab 3: Adaptive PG + KD

## Complete Colab-Ready Notebook

Here's the complete setup for a Colab-ready version:

```python
# ============================================
# CELL 0: COLAB SETUP (Add this as first cell)
# ============================================

# Detect Colab environment
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    print("🚀 Setting up Google Colab environment...\n")
    
    # Enable GPU
    import torch
    assert torch.cuda.is_available(), "⚠️ GPU not enabled! Go to Runtime → Change runtime type → GPU"
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    
    # Clone repository (REPLACE WITH YOUR REPO URL)
    print("\n📦 Cloning repository...")
    !git clone https://github.com/YOUR_USERNAME/endingengineering.git
    %cd endingengineering
    
    # Install dependencies
    print("\n📚 Installing dependencies...")
    !pip install -q tqdm seaborn
    
    # Mount Google Drive (optional, for saving models)
    print("\n💾 Mounting Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    
    print("\n✅ Setup complete! Ready to run.")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   PyTorch: {torch.__version__}")
else:
    print("✓ Running locally")

# ============================================
# Continue with rest of notebook cells...
# ============================================
```

## Quick Start Checklist

- [ ] Open https://colab.research.google.com/
- [ ] Upload `test_ada_fracbnn.ipynb` OR clone from GitHub
- [ ] Enable GPU (`Runtime` → `Change runtime type` → `GPU`)
- [ ] Add Colab setup cell at the top
- [ ] Update config with Colab paths
- [ ] Run all cells
- [ ] Monitor training and save results

## Need Help?

If you encounter issues:
1. Check GPU is enabled: `torch.cuda.is_available()`
2. Verify files are in place: `!ls -la`
3. Check imports work: `import utils.quantization as q`
4. Monitor GPU memory: `!nvidia-smi`
5. Check Colab logs for error messages

Happy training in Colab! 🚀

