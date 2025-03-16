import os
import argparse
import numpy as np
from PIL import Image
import subprocess

def generate_overlays(video):
    frames_dir = os.path.join('data', f'{video}_frames')
    masks_dir = os.path.join('data', f'{video}_saliency')
    overlay_dir = os.path.join('data', f'{video}_saliency_overlay')
    
    if not os.path.isdir(frames_dir):
        print(f"Frames directory {frames_dir} does not exist.")
        return
    if not os.path.isdir(masks_dir):
        print(f"Saliency masks directory {masks_dir} does not exist.")
        return
    
    os.makedirs(overlay_dir, exist_ok=True)
    
    frame_files = sorted(os.listdir(frames_dir))
    for fname in frame_files:
        frame_path = os.path.join(frames_dir, fname)
        mask_path = os.path.join(masks_dir, fname)
        overlay_path = os.path.join(overlay_dir, fname)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask {fname} not found. Skipping.")
            continue
        
        try:
            original = Image.open(frame_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            # Resize mask to match original frame dimensions
            mask = mask.resize(original.size, Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error opening {fname}: {e}")
            continue
        
        original_np = np.array(original, dtype=np.float32) / 255.0
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        mask_np = mask_np * 0.5 + 0.5  # Normalize to [0, 1]
        
        # Create pure red background (in RGB)
        red = np.zeros_like(original_np)
        red[..., 0] = 1.0  # Red channel
        
        # Blend with original image using mask
        overlay_np = red * (1 - mask_np[..., np.newaxis]) + (original_np * mask_np[..., np.newaxis])
        
        overlay_np = (overlay_np * 255).clip(0, 255).astype(np.uint8)
        overlay_img = Image.fromarray(overlay_np)
        
        overlay_img.save(overlay_path)
        print(f"Processed {fname}")

def visualize_overlays(video):
    overlay_dir = os.path.join('data', f'{video}_saliency_overlay')
    video_path = os.path.join('data', f'{video}_saliency_video.mp4')
    
    print(f"Overlay directory {overlay_dir}.")
    if not os.path.isdir(overlay_dir):
        print(f"Overlay directory {overlay_dir} does not exist.")
        return
    
    command = [
        "ffmpeg", 
        "-framerate", "4",
        "-i", os.path.join(overlay_dir, "%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        video_path
    ]
    subprocess.run(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saliency Overlay Tool')
    parser.add_argument('action', choices=['get', 'vis'], 
                       help='Action: generate overlays (get) or create video (vis)')
    parser.add_argument('--vid', required=True, 
                       help='Video name (without _frames or _saliency)')
    args = parser.parse_args()
    
    if args.action == 'get':
        generate_overlays(args.vid)
        visualize_overlays(args.vid)
    elif args.action == 'vis':
        visualize_overlays(args.vid)