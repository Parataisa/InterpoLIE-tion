import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

class FileHandler:
    def __init__(self, downscale_size=512, downscale=True, crop_center=False):
        self.downscale_size = downscale_size
        self.downscale = downscale
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
        self.crop_center = crop_center
    
    def scan_folder(self, folder_path):
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        images = []
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                images.append(file_path)
        
        return sorted(images)
    
    def load_image(self, img_path, apply_downscale=None):
        try:
            img = np.array(Image.open(img_path).convert('L'))
            
            should_downscale = apply_downscale if apply_downscale is not None else self.downscale
            
            if should_downscale:
                h, w = img.shape[:2]
                
                if self.crop_center:
                    print(f"    Cropping center of image {img_path} to size {self.downscale_size}x{self.downscale_size}")
                    crop_size = self.downscale_size
                    
                    if h >= crop_size and w >= crop_size:
                        center_x, center_y = w // 2, h // 2
                        
                        left = center_x - crop_size // 2
                        right = center_x + crop_size // 2
                        top = center_y - crop_size // 2
                        bottom = center_y + crop_size // 2
                        
                        img = img[top:bottom, left:right]
                else:
                    if w > self.downscale_size:
                        print(f"    Downscaling image {img_path} from {w} to {self.downscale_size} pixels wide")
                        scale_factor = self.downscale_size / w
                        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            return img.astype(np.float32)
            
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")
    
    def load_image_rgb(self, img_path, target_size=None):
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            h, w = img_array.shape[:2]
            
            if self.crop_center:
                print(f"    Cropping center of image {img_path} to size {self.downscale_size}x{self.downscale_size}")
                crop_size = self.downscale_size
                
                if h >= crop_size and w >= crop_size:
                    center_x, center_y = w // 2, h // 2
                    
                    left = center_x - crop_size // 2
                    right = center_x + crop_size // 2
                    top = center_y - crop_size // 2
                    bottom = center_y + crop_size // 2
                    
                    img_array = img_array[top:bottom, left:right]
            else:
                if w > self.downscale_size:
                    print(f"    Downscaling image {img_path} from {w} to {self.downscale_size} pixels wide")
                    scale_factor = self.downscale_size / w
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            return img_array
            
        except Exception as e:
            raise IOError(f"Error loading RGB image {img_path}: {e}")
    
    def find_image_file(self, filename, search_paths=None):
        if search_paths is None:
            search_paths = ['.', 'img', '../img', '../../img']
        
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename
        
        for search_path in search_paths:
            search_dir = Path(search_path)
            if search_dir.exists():
                candidate = search_dir / filename
                if candidate.exists():
                    return str(candidate)
                
                for file_path in search_dir.rglob(filename):
                    if file_path.is_file():
                        return str(file_path)
        
        return None

    def create_output_folder(self, output_path):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return str(Path(output_path).resolve())