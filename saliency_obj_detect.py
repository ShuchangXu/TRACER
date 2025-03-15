import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from py360convert import e2c, c2e
from torchvision.transforms import transforms
from model.TRACER import TRACER
from util.utils import load_pretrained

class SaliencyProcessor:
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.test_transform = self.get_test_augmentation()
        
        # Initialize model
        self.model = TRACER(args).to(self.device)
        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)
        
        # Load pretrained weights
        path = load_pretrained(f'TE-{args.arch}')
        self.model.load_state_dict(path)
        self.model.eval()
        print('###### TRACER model loaded #####')

    def get_test_augmentation(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def process_image(self, image):
        # Apply transformations
        image = self.test_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs, _, _ = self.model(image)
            output = F.interpolate(outputs, size=image.shape[2:], mode='bilinear')
        
        # Post-processing
        saliency = (output.squeeze().cpu().numpy() * 255).astype(np.uint8)
        return saliency

    def post_process(self, original, saliency):
        # Combine with original image
        original = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
        saliency_rgba = cv2.cvtColor(saliency, cv2.COLOR_GRAY2BGRA)
        saliency_rgba[:, :, 3] = saliency  # Alpha channel
        return cv2.addWeighted(original, 0.7, saliency_rgba, 0.3, 0)

def main():
    # Configuration (modify as needed)
    class Args:
        img_size = 352
        arch = 7
        multi_gpu = False
        save_map = True

    args = Args()
    processor = SaliencyProcessor(args)

    # Load equirectangular frame
    erp_image = cv2.imread('frames/frame_0001.png')
    h, w = erp_image.shape[:2]

    # Step 2: Convert to cubemap
    cubemap = e2c(erp_image, face_w=256, cube_format='dict')

    # Define face pairs
    pairs = [
        {'faces': ['B', 'D'], 'dir': 'vertical'},
        {'faces': ['B', 'L'], 'dir': 'horizontal'},
        {'faces': ['F', 'D'], 'dir': 'vertical'},
        {'faces': ['F', 'R'], 'dir': 'horizontal'},
        {'faces': ['L', 'D'], 'dir': 'vertical'},
        {'faces': ['L', 'F'], 'dir': 'horizontal'},
        {'faces': ['R', 'B'], 'dir': 'horizontal'},
        {'faces': ['R', 'D'], 'dir': 'vertical'},
        {'faces': ['U', 'B'], 'dir': 'vertical'},
        {'faces': ['U', 'F'], 'dir': 'vertical'},
        {'faces': ['U', 'L'], 'dir': 'horizontal'},
        {'faces': ['U', 'R'], 'dir': 'horizontal'}
    ]

    # Process each pair
    saliency = {face: [] for face in cubemap.keys()}
    for pair in pairs:
        f1, f2 = pair['faces']
        img1, img2 = cubemap[f1], cubemap[f2]
        
        # Concatenate faces
        if pair['dir'] == 'horizontal':
            concat = cv2.hconcat([img1, img2])
        else:
            concat = cv2.vconcat([img1, img2])
        
        # Process with TRACER
        saliency_map = processor.process_image(concat)
        
        # Split and store results
        if pair['dir'] == 'horizontal':
            h_split = saliency_map.shape[1] // 2
            saliency[f1].append(saliency_map[:, :h_split])
            saliency[f2].append(saliency_map[:, h_split:])
        else:
            v_split = saliency_map.shape[0] // 2
            saliency[f1].append(saliency_map[:v_split, :])
            saliency[f2].append(saliency_map[v_split:, :])

    # Average saliency maps
    saliency_cube = {face: np.mean(saliency[face], axis=0) for face in saliency}
    
    # Convert to equirectangular
    cube_list = [saliency_cube['F'], saliency_cube['R'], saliency_cube['B'],
                 saliency_cube['L'], saliency_cube['U'], saliency_cube['D']]
    erp_saliency = c2e(cube_list, h, w, cube_format='list')

    # Post-process and save
    final_result = processor.post_process(erp_image, erp_saliency)
    cv2.imwrite('final_saliency.png', final_result)

if __name__ == '__main__':
    main()