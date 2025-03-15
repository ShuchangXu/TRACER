import os
import cv2
import numpy as np
import subprocess
import py360convert
from tqdm import tqdm

# Configuration
FACE_ORDER = ['U', 'D', 'F', 'B', 'R', 'L']
PAIRS = ['BD', 'BL', 'FD', 'FR', 'LD', 'LF', 'RB', 'RD', 'UB', 'UF', 'UL', 'UR']
PAIR_COMPONENTS = {
    'BD': ('B', 'D'), 'BL': ('B', 'L'), 'FD': ('F', 'D'), 'FR': ('F', 'R'),
    'LD': ('L', 'D'), 'LF': ('L', 'F'), 'RB': ('R', 'B'), 'RD': ('R', 'D'),
    'UB': ('U', 'B'), 'UF': ('U', 'F'), 'UL': ('U', 'L'), 'UR': ('U', 'R')
}
PAIR_ROTATIONS = {
    'BD': {'B': 0, 'D': 180},   'BL': {'B': 0, 'L': 0},
    'FD': {'F': 0, 'D': 0},   'FR': {'F': 0, 'R': 0},
    'LD': {'L': 0, 'D': 270},   'LF': {'L': 0, 'F': 0},
    'RB': {'R': 0, 'B': 0},   'RD': {'R': 0, 'D': 90},
    'UB': {'U': 180, 'B': 0}, 'UF': {'U': 0, 'F': 0},
    'UL': {'U': 90, 'L': 0}, 'UR': {'U': 270, 'R': 0}
}

def rotate_image(image, angle):
    """Rotate image by exact 90-degree increments without quality loss"""
    if angle == 0:
        return image
    
    # Convert angle to positive equivalent and get rotation count
    angle = angle % 360
    if angle not in {90, 180, 270}:
        raise ValueError("Angle must be one of 0, 90, 180, 270 degrees")
    
    # Calculate number of 90 CCW rotations needed
    k = angle // 90
    return np.rot90(image, k=k)

def video_frame_sampling(video_path, sample_rate):
    video = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("data", f"{video}_frames")
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(round(fps / sample_rate))
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"{saved_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    return output_dir

def get_faces(frame_dir):
    video = os.path.basename(frame_dir).replace('_frames', '')
    output_dir = os.path.join("data", f"{video}_faces")
    os.makedirs(output_dir, exist_ok=True)
    
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    
    for frame_name in tqdm(frame_files, desc="Extracting cubemap faces"):
        img = cv2.imread(os.path.join(frame_dir, frame_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            cube = py360convert.e2c(img, face_w=256, mode='bilinear', cube_format='dict')
        except Exception as e:
            print(f"Error processing {frame_name}: {str(e)}")
            continue
        
        frame_num = os.path.splitext(frame_name)[0]
        for face in ['F', 'R', 'B', 'L', 'U', 'D']:
            face_img = cube[face]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"{frame_num}_{face}.png"), face_img)
    
    return output_dir

def get_face_pairs(face_dir):
    video = os.path.basename(face_dir).replace('_faces', '')
    output_dir = os.path.join("data", f"{video}_pairs")
    os.makedirs(output_dir, exist_ok=True)
    
    frame_nums = sorted(list({f.split('_')[0] for f in os.listdir(face_dir)}))
    
    for frame in tqdm(frame_nums, desc="Generating face pairs"):
        for pair, components in PAIR_ROTATIONS.items():
            f1, f2 = pair[:1], pair[1:]
            img1 = cv2.imread(os.path.join(face_dir, f"{frame}_{f1}.png"))
            img2 = cv2.imread(os.path.join(face_dir, f"{frame}_{f2}.png"))
            
            if img1 is None or img2 is None:
                continue
                
            # Apply rotations
            img1_rot = rotate_image(img1, components[f1])
            img2_rot = rotate_image(img2, components[f2])
            
            # Vertical concatenation for U/D pairs
            if f1 in ['U', 'D'] or f2 in ['U', 'D']:
                pair_img = cv2.vconcat([img1_rot, img2_rot])
            else:
                pair_img = cv2.hconcat([img1_rot, img2_rot])
            
            cv2.imwrite(os.path.join(output_dir, f"{frame}_{pair}.png"), pair_img)
    
    return output_dir

def saliency_detect(video):
    command = [
        "python", 
        "main.py",
        "inference",
        "--dataset", os.path.join(f"{video}_pairs"), 
        "--arch", "7",
        "--img_size", "640",
        "--save_map", "True",
        "--batch_size", "1",
    ]
    subprocess.run(command)

def saliency_stitch(video):
    face_dir = os.path.join("data", f"{video}_faces")
    pair_dir = os.path.join("mask", f"{video}_pairs")
    output_dir = os.path.join("data", f"{video}_saliency")
    os.makedirs(output_dir, exist_ok=True)
    
    frame_nums = sorted(list({f.split('_')[0] for f in os.listdir(pair_dir)}))
    
    for frame in tqdm(frame_nums, desc="Stitching saliency maps"):
        face_contributions = {f: [] for f in FACE_ORDER}
        
        for pair, components in PAIR_ROTATIONS.items():
            pair_path = os.path.join(pair_dir, f"{frame}_{pair}.png")
            if not os.path.exists(pair_path):
                continue
                
            saliency = cv2.imread(pair_path, cv2.IMREAD_GRAYSCALE)
            if saliency is None:
                continue
                
            f1, f2 = pair[:1], pair[1:]
            h, w = saliency.shape
            
                        # Split and rotate back with exact inverses
            if f1 in ['U', 'D'] or f2 in ['U', 'D']:
                part1 = saliency[:h//2, :]
                part2 = saliency[h//2:, :]
            else:
                part1 = saliency[:, :w//2]
                part2 = saliency[:, w//2:]

            # Reverse rotations using numpy's optimised rotations
            part1 = rotate_image(part1, -components[f1])
            part2 = rotate_image(part2, -components[f2])
            
            face_contributions[f1].append(part1)
            face_contributions[f2].append(part2)
        
        # Combine using max projection
        cube_faces = []
        for face in FACE_ORDER:
            if face_contributions[face]:
                # Use maximum value across all contributions
                max_face = np.max(np.stack(face_contributions[face]), axis=0)
                cube_faces.append(max_face.astype(np.uint8))
            else:
                cube_faces.append(np.zeros((256, 256), dtype=np.uint8))
        
        try:
            cube_dict = {face: cube_faces[i] for i, face in enumerate(FACE_ORDER)}
            h, w = 540, 960
            equirect = py360convert.c2e(cube_dict, h, w, cube_format='dict')
            equirect = cv2.cvtColor(equirect, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(output_dir, f"{frame}.png"), equirect)
        except Exception as e:
            print(f"Error stitching {frame}: {str(e)}")

def main(video_path, sample_rate, start_with):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if start_with <= 0:
        frames_dir = video_frame_sampling(video_path, sample_rate)
    else:
        frames_dir = os.path.join("data", f"{video_name}_frames")
    
    if start_with <= 1:
        faces_dir = get_faces(frames_dir)
    else:
        faces_dir = os.path.join("data", f"{video_name}_faces")
    
    if start_with <= 2:
        pairs_dir = get_face_pairs(faces_dir)
    else:
        pairs_dir = os.path.join("data", f"{video_name}_pairs")
    
    if start_with <= 3:
        saliency_detect(video_name)
    
    if start_with <= 4:
        saliency_stitch(video_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--start_with", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    
    main(args.video, args.sample_rate, args.start_with)