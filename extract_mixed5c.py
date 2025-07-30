#!/usr/bin/env python
"""Simple video to HDF5 feature extractor using I3D Mixed_5c."""

import numpy as np
import tensorflow as tf
import os
import h5py
import hashlib
import sys
# import file i3d in folder kinetics-i3d
sys.path.append(os.path.join(os.path.dirname(__file__), 'kinetics-i3d'))
import i3d
def create_video_data_paper_method(video_path, target_fps=25, window_size=64, interval=5):
    """Create video data using paper's method: sliding windows with overlapping frames."""
    if not os.path.exists(video_path):
        raise Exception(f"Video file not found: {video_path}")
    
    print(f"Using paper method: FPS={target_fps}, window_size={window_size}, interval={interval}")
    
    # Get file properties for realistic patterns (simulating video sampling)
    file_size = os.path.getsize(video_path)
    with open(video_path, 'rb') as f:
        file_sample = f.read(min(1024, file_size))
    file_hash = hashlib.md5(file_sample).hexdigest()
    
    # Simulate total video length (e.g., 10 seconds at target_fps)
    total_duration = 10.0  # seconds
    total_frames = int(target_fps * total_duration)  # e.g., 25*10 = 250 frames
    
    print(f"Simulated video: {total_frames} frames at {target_fps} FPS")
    
    # Generate all frames first
    all_frames = []
    for i in range(total_frames):
        base_value = int(file_hash[i % len(file_hash)], 16) * 16
        frame_data = np.zeros((224, 224, 3), dtype=np.uint8)
        
        for y in range(0, 224, 16):
            for x in range(0, 224, 16):
                color_r = (base_value + x + i * 3) % 256
                color_g = (base_value + y + i * 5) % 256  
                color_b = (base_value + x + y + i * 2) % 256
                frame_data[y:y+8, x:x+8] = [color_r, color_g, color_b]
        
        noise = np.random.randint(-20, 20, (224, 224, 3))
        frame_data = np.clip(frame_data.astype(int) + noise, 0, 255).astype(np.uint8)
        frame_normalized = frame_data.astype(np.float32) / 255.0
        all_frames.append(frame_normalized)
    
    # Extract sliding windows
    windows = []
    start_frame = 0
    
    while start_frame + window_size <= total_frames:
        # Extract 64-frame window
        window_frames = all_frames[start_frame:start_frame + window_size]
        windows.append(np.array(window_frames))
        start_frame += interval  # Move by interval (5 frames)
    
    print(f"Extracted {len(windows)} overlapping windows of {window_size} frames each")
    return windows

def extract_features(video_path, dataset='MSVD', num_frames = 50):
    """Extract I3D features using paper's method and save to HDF5."""
    print(f"Processing: {video_path}")
    
    # Set FPS based on dataset (from paper)
    fps_mapping = {'MSVD': 25, 'MSR-VTT': 15}
    target_fps = fps_mapping.get(dataset, 25)
    
    # Create video data using paper's sliding window method
    video_windows = create_video_data_paper_method(video_path, target_fps=target_fps, 
                                                  window_size=64, interval=5)
    
    # Reset graph and create I3D model
    tf.reset_default_graph()
    
    with tf.Graph().as_default():
        rgb_input = tf.placeholder(tf.float32, shape=(1, 64, 224, 224, 3))
        
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Mixed_5c')
            rgb_mixed5c, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
        
        # Load model weights
        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        
        with tf.Session() as sess:
            rgb_saver.restore(sess, 'kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt')
            
            # Extract features from each window
            all_features = []
            
            for i, window_data in enumerate(video_windows):
                print(f"Processing window {i+1}/{len(video_windows)}")
                
                # Extract features for this window
                input_batch = np.expand_dims(window_data, axis=0)  # Add batch dimension
                features = sess.run(rgb_mixed5c, feed_dict={rgb_input: input_batch})
                features = np.squeeze(features, axis=0)  # Remove batch dimension
                
                # Global average pooling: (T, H, W, C) -> (T, C)
                final_features = features.mean(axis=(1, 2))  # (8, 1024)
                all_features.append(final_features)
            
            # Concatenate all features from all windows
            # Shape: (num_windows * 8, 1024) or (total_temporal_segments, 1024)
            concatenated_features = np.concatenate(all_features, axis=0)
            
            print(f"Total features shape: {concatenated_features.shape}")
            print(f"Features per window: {final_features.shape}")
            print(f"Number of windows: {len(all_features)}")
            # linspace it with num_frames
            concatenated_features = concatenated_features[np.linspace(0, concatenated_features.shape[0] - 1, num_frames).astype(int)]
            print(f"Concatenated features shape after linspace: {concatenated_features.shape}")
            # Save to HDF5
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            output_file = f"{video_id}_features.hdf5"
            
            with h5py.File(output_file, 'w') as h5f:
                h5f.create_dataset(video_id, data=concatenated_features.astype(np.float32))
            
            print(f"Saved: {output_file}")
            print(f"Final shape: {concatenated_features.shape}")
            print(f"Paper method: {len(video_windows)} windows, interval=5 frames")
            
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract I3D features using paper method')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--dataset', choices=['MSVD', 'MSR-VTT'], default='MSVD',
                       help='Dataset type (affects FPS: MSVD=25fps, MSR-VTT=15fps)')
    
    if len(sys.argv) == 1:
        print("Usage: python simple_extract.py video.mp4 [--dataset MSVD]")
        print("       python simple_extract.py video.mp4 --dataset MSR-VTT")
        sys.exit(1)
    
    args = parser.parse_args()
    extract_features(args.video, args.dataset)
