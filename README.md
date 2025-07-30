# I3D Mixed_5c Feature Extraction Setup Guide

This guide will walk you through setting up the kinetics-i3d repository, environment, and using the `extract_mixed5c_robust.py` script to extract I3D Mixed_5c features from video files.

## 📥 Step 1: Clone the Kinetics-I3D Repository

### Clone the original repository:
```bash
git clone https://github.com/google-deepmind/kinetics-i3d.git
```

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

## 🎬 Step 4: Using extract_mixed5c.py

### Basic Usage:
```bash
python extract_mixed5c_robust.py your_video.mp4
```

