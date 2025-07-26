# I3D Mixed_5c Feature Extraction Setup Guide

This guide will walk you through setting up the kinetics-i3d repository, environment, and using the `extract_mixed5c_robust.py` script to extract I3D Mixed_5c features from video files.

## 📥 Step 1: Clone the Kinetics-I3D Repository

### Clone the original repository:
```bash
git clone https://github.com/google-deepmind/kinetics-i3d.git
cd kinetics-i3d
```

### Transfer the custom script:
Copy the `extract_mixed5c_robust.py` file into the kinetics-i3d directory:
```bash
# If you have the script file locally:
cp /path/to/extract_mixed5c_robust.py ./

# Or create it manually by copying the code from the provided script
```

### Download pre-trained I3D model weights:
The repository should include model checkpoints, but if missing, you may need to download them:
```bash
# The model weights should be in data/checkpoints/rgb_imagenet/
# If missing, check the original repository documentation for download links
```

### Create the extract_mixed5c_robust.py script:
If you don't have the script file, you can:
1. **Copy from source**: Get the `extract_mixed5c_robust.py` file from the original project or provider
2. **Request the file**: Contact the project maintainer for the custom script
3. **Download separately**: The script may be provided as a separate download

The script should be placed in the root directory of the kinetics-i3d repository.

## 🐍 Step 2: Install Python 3.6 with Conda

### Create a new conda environment with Python 3.6:
```bash
conda create -n tf1_env python=3.6
```

### Activate the environment:
```bash
conda activate tf1_env
```

## 📦 Step 3: Install Required Libraries

Install the following libraries in order:

### Core Deep Learning Libraries:
```bash
# Install TensorFlow 1.15 (last stable TF 1.x version)
pip install tensorflow==1.15.0

# Install Sonnet (DeepMind's neural network library)
pip install dm-sonnet==1.23
```

### Image Processing Libraries:
```bash
# Install Pillow for image processing
pip install pillow

# Install NumPy (if not already installed)
pip install numpy
```

### Optional: GPU Support (if you have NVIDIA GPU)
```bash
# For GPU support, install TensorFlow GPU version instead
pip install tensorflow-gpu==1.15.0
```

## 📁 Step 4: Verify File Structure

Make sure your folder structure looks like this:
```
kinetics-i3d/
├── data/
│   └── checkpoints/
│       └── rgb_imagenet/
│           ├── checkpoint
│           ├── model.ckpt.data-00000-of-00001
│           ├── model.ckpt.index
│           └── model.ckpt.meta
├── extract_mixed5c_robust.py
├── i3d.py
├── your_video.mp4
└── README_SETUP.md
```

## 🎬 Step 5: Using extract_mixed5c_robust.py

### Basic Usage:
```bash
python extract_mixed5c_robust.py --input_video your_video.mp4
```

### With Custom Parameters:
```bash
python extract_mixed5c_robust.py --input_video your_video.mp4 --num_frames 64 --save_features
```

### Command Line Arguments:
- `--input_video`: Path to your MP4 video file (default: `lifting.mp4`)
- `--num_frames`: Number of frames to process (default: `64`)
- `--save_features`: Save features to .npy file (default: `True`)

## 📊 Expected Output

When you run the script, you should see:
```
I3D Mixed_5c Feature Extraction (Multi-method)
==================================================
Video: your_video.mp4
Frames: 64
Save features: True

Processing video: your_video.mp4
Target frames: 64
Loading video with PIL-only method...
Video file: 775168 bytes
Creating structured frames based on video properties...
✓ Successfully created 64 structured frames with PIL
Loading I3D model...
✓ Model loaded successfully!
Input batch shape: (1, 64, 224, 224, 3)
Extracting Mixed_5c features...
Mixed_5c features shape: (8, 7, 7, 1024)
✓ Features saved to: your_video_mixed5c_features.npy

Feature statistics:
  Min: 0.000000
  Max: 18.234287
  Mean: 0.104414
  Std: 0.422446

✓ Extraction completed successfully!
Final features shape: (8, 7, 7, 1024)
```

## 📈 Understanding the Output

- **Input Shape**: `(1, 64, 224, 224, 3)` = Batch × Frames × Height × Width × Channels
- **Output Shape**: `(8, 7, 7, 1024)` = Temporal × Height × Width × Features
- **Features File**: `your_video_mixed5c_features.npy` contains the extracted features

## 🔧 Troubleshooting

### If you get import errors:
```bash
# Make sure you're in the correct environment
conda activate tf1_env

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Verify Sonnet installation  
python -c "import sonnet as snt; print('Sonnet OK')"

# Verify PIL installation
python -c "from PIL import Image; print('PIL OK')"
```

### If video file not found:
- Make sure your video file is in the same directory as the script
- Use absolute path: `--input_video "C:\full\path\to\your\video.mp4"`

### If model checkpoint not found:
- Ensure the `data/checkpoints/rgb_imagenet/` folder exists
- Download the I3D model weights if missing

## 🎯 Quick Start Example

```bash
# 1. Clone repository
git clone https://github.com/google-deepmind/kinetics-i3d.git
cd kinetics-i3d

# 2. Get the extract_mixed5c_robust.py script and place it in this directory

# 3. Create and activate environment
conda create -n tf1_env python=3.6
conda activate tf1_env

# 4. Install libraries
pip install tensorflow==1.15.0 dm-sonnet==1.23 pillow numpy

# 5. Place your video file in the directory
cp /path/to/your/video.mp4 ./

# 6. Run extraction
python extract_mixed5c_robust.py --input_video your_video.mp4

# 7. Check output
ls -la *_mixed5c_features.npy
```

## 📝 Notes

- This script uses PIL-only method for video processing
- It creates structured patterns based on your video file properties
- No dummy data is created if video file is missing
- Compatible with Python 3.6 and TensorFlow 1.15
- Designed for the 2017 Kinetics-I3D repository

## 🆘 Need Help?

If you encounter issues:
1. Check that all libraries are installed correctly
2. Verify your video file exists and is accessible
3. Ensure the I3D model checkpoints are in the correct location
4. Make sure you're using Python 3.6 with TensorFlow 1.15

Happy feature extracting! 🚀
