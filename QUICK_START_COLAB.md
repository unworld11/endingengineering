# Quick Start Guide: Running in Google Colab

## 🚀 3-Minute Setup

### Option 1: GitHub (Recommended)

1. **Upload to GitHub**:
   ```bash
   cd /path/to/endingengineering
   git add .
   git commit -m "Add Colab notebook"
   git push
   ```

2. **Open in Colab**:
   - Go to: https://colab.research.google.com/
   - Click `File` → `Open notebook` → `GitHub` tab
   - Enter: `YOUR_USERNAME/endingengineering`
   - Select: `test_ada_fracbnn_colab.ipynb`

3. **Enable GPU**:
   - Click `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Click `Save`

4. **Run Setup Cell**:
   - In Cell 2, uncomment this line:
     ```python
     !git clone https://github.com/YOUR_USERNAME/endingengineering.git
     %cd endingengineering
     ```
   - Run the cell (takes ~30 seconds)

5. **Run All Other Cells**:
   - Click `Runtime` → `Run all`
   - Done! ✅

### Option 2: Direct Upload (No GitHub)

1. **Zip your project**:
   ```bash
   cd /path/to
   zip -r endingengineering.zip endingengineering/ \
       -x "*.pyc" -x "__pycache__/*" -x ".git/*" -x "*.pth"
   ```

2. **Open Colab**: https://colab.research.google.com/

3. **Upload notebook**:
   - Click `File` → `Upload notebook`
   - Select `test_ada_fracbnn_colab.ipynb`

4. **Enable GPU**: `Runtime` → `Change runtime type` → `GPU`

5. **In Cell 2, uncomment upload code**:
   ```python
   from google.colab import files
   import zipfile
   uploaded = files.upload()  # Select your endingengineering.zip
   for f in uploaded.keys():
       if f.endswith('.zip'):
           !unzip -q {f}
   %cd endingengineering
   ```

6. **Run the cell** → Upload your zip when prompted

7. **Run remaining cells** ✅

## 📊 Quick Test Commands

Once setup is complete:

```python
# Quick model test (in notebook)
# Just run all cells sequentially

# Full training (add new cell):
!python cifar10.py -id 1 -e 250 -b 256 -ts 0.15 -sw 0.01 -s

# Monitor GPU:
!nvidia-smi
```

## 💾 Save Results to Google Drive

Add this cell after training:

```python
from google.colab import files

# Download model
files.download('save_CIFAR10_model/adaptive-pg.pth')

# Or save to Drive (if mounted)
!cp save_CIFAR10_model/*.pth /content/drive/MyDrive/ada_fracbnn_models/
```

## 🔧 Common Issues

**"Module not found"**
```python
import sys
sys.path.insert(0, '/content/endingengineering')
```

**"No GPU"**
- Runtime → Change runtime type → GPU → Save
- Check: `torch.cuda.is_available()`

**"Out of memory"**
```python
config['batch_size'] = 128  # reduce from 256
```

## 📱 Direct Colab Link

After uploading to GitHub, share this link:
```
https://colab.research.google.com/github/YOUR_USERNAME/endingengineering/blob/main/test_ada_fracbnn_colab.ipynb
```

Anyone can click and run immediately! 🎉

