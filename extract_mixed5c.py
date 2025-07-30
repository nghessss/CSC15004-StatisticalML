#!/usr/bin/env python
"""Simple video to HDF5 feature extractor using I3D Mixed_5c."""

import numpy as np
import tensorflow as tf
import cv2
import os
import h5py
import sys
# import file i3d in folder kinetics-i3d
sys.path.append(os.path.join(os.path.dirname(__file__), 'kinetics-i3d'))
import i3d
def sample_frames(video_path, target_fps):
    """Sample frames from video at target FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(orig_fps / target_fps)))
    frames = []
    idx = 0
    
    print(f"Original FPS: {orig_fps}, Target FPS: {target_fps}, Sampling interval: {interval}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            # Convert BGR to RGB and resize to 224x224
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            frames.append(frame_normalized)
        idx += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames from video")
    return frames

def create_video_clips(frames, clip_len=64, stride=5):
    """Create overlapping video clips from frames."""
    clips = []
    
    for start_idx in range(0, len(frames) - clip_len + 1, stride):
        clip = frames[start_idx:start_idx + clip_len]
        if len(clip) == clip_len:  # Ensure we have exactly clip_len frames
            clips.append(np.array(clip))
    
    print(f"Created {len(clips)} clips of {clip_len} frames each with stride {stride}")
    return clips

def extract_features(video_path, dataset='MSVD', num_frames=50):
    """Extract I3D features using real video processing and save to HDF5."""
    print(f"Processing: {video_path}")
    
    if not os.path.exists(video_path):
        raise Exception(f"Video file not found: {video_path}")
    
    # Set FPS based on dataset (from paper)
    fps_mapping = {'MSVD': 25, 'MSR-VTT': 15}
    target_fps = fps_mapping.get(dataset, 25)
    
    # Sample frames from video at target FPS
    frames = sample_frames(video_path, target_fps)
    
    if len(frames) < 64:
        print(f"Warning: Video has only {len(frames)} frames, less than required 64 frames")
        # Pad with repeated frames if needed
        while len(frames) < 64:
            frames.extend(frames[:min(64-len(frames), len(frames))])
    
    # Create overlapping video clips using paper's method
    video_clips = create_video_clips(frames, clip_len=64, stride=5)
    
    if not video_clips:
        raise Exception("No valid clips could be created from video")
    
    # Reset graph and create I3D model (TensorFlow 1.x compatibility)
    tf.compat.v1.reset_default_graph()
    
    with tf.Graph().as_default():
        rgb_input = tf.compat.v1.placeholder(tf.float32, shape=(1, 64, 224, 224, 3))
        
        with tf.compat.v1.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Mixed_5c')
            rgb_mixed5c, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
        
        # Load model weights
        rgb_variable_map = {}
        for variable in tf.compat.v1.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        
        rgb_saver = tf.compat.v1.train.Saver(var_list=rgb_variable_map, reshape=True)
        
        with tf.compat.v1.Session() as sess:
            rgb_saver.restore(sess, 'kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt')
            
            # Extract features from each clip
            all_features = []
            
            for i, clip_data in enumerate(video_clips):
                print(f"Processing clip {i+1}/{len(video_clips)}")
                
                # Extract features for this clip
                input_batch = np.expand_dims(clip_data, axis=0)  # Add batch dimension
                features = sess.run(rgb_mixed5c, feed_dict={rgb_input: input_batch})
                features = np.squeeze(features, axis=0)  # Remove batch dimension
                
                # Global average pooling: (T, H, W, C) -> (T, C)
                final_features = features.mean(axis=(1, 2))  # (8, 1024)
                all_features.append(final_features)
            
            # Concatenate all features from all clips
            # Shape: (num_clips * 8, 1024) or (total_temporal_segments, 1024)
            concatenated_features = np.concatenate(all_features, axis=0)
            
            print(f"Total features shape: {concatenated_features.shape}")
            print(f"Features per clip: {final_features.shape}")
            print(f"Number of clips: {len(all_features)}")
            
            # Resample to fixed number of frames using linspace
            if concatenated_features.shape[0] > 0:
                indices = np.linspace(0, concatenated_features.shape[0] - 1, num_frames).astype(int)
                final_features_resampled = concatenated_features[indices]
            else:
                final_features_resampled = np.zeros((num_frames, 1024), dtype=np.float32)
            
            print(f"Final resampled features shape: {final_features_resampled.shape}")
            
            # Save to HDF5
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            output_file = f"{video_id}_features.hdf5"
            
            with h5py.File(output_file, 'w') as h5f:
                h5f.create_dataset(video_id, data=final_features_resampled.astype(np.float32))
            
            print(f"Saved: {output_file}")
            print(f"Final shape: {final_features_resampled.shape}")
            print(f"Real video processing: {len(video_clips)} clips, stride=5 frames")
            
            return final_features_resampled
            
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
