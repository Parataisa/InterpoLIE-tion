import os
import cv2
import numpy as np
import re
from pathlib import Path
from PIL import Image

class FileHandler:
    def __init__(self, downscale_size=512, downscale=True, crop_center=False):
        self.downscale_size = downscale_size
        self.downscale = downscale
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
        self.crop_center = crop_center
    
    def natural_sort_key(self, filepath):
        filename = filepath.name
        parts = re.split(r'(\d+)', filename)
        result = []
        for part in parts:
            if part.isdigit():
                result.append(int(part))
            else:
                result.append(part.lower())  
        return result
    
    def scan_folder(self, folder_path):
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        images = []
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                images.append(file_path)
        
        images.sort(key=self.natural_sort_key)
        return images
    
    def apply_image_processing(self, img, apply_downscale=None, apply_crop=None):
        should_downscale = apply_downscale if apply_downscale is not None else self.downscale
        should_crop = apply_crop if apply_crop is not None else self.crop_center
        
        h, w = img.shape[:2]
        
        if should_crop:
            crop_size = self.downscale_size
            if h >= crop_size and w >= crop_size:
                center_x, center_y = w // 2, h // 2
                
                left = center_x - crop_size // 2
                right = center_x + crop_size // 2
                top = center_y - crop_size // 2
                bottom = center_y + crop_size // 2
                
                img = img[top:bottom, left:right]
        
        elif should_downscale:
            h, w = img.shape[:2]
            if w > self.downscale_size:
                scale_factor = self.downscale_size / w
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return img
    
    def load_image(self, img_path, apply_downscale=None, apply_crop=None):
        try:
            img = np.array(Image.open(img_path).convert('L'))
            img = self.apply_image_processing(img, apply_downscale, apply_crop)
            return img.astype(np.float32)
            
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")
    
    def load_image_rgb(self, img_path, target_size=None, apply_downscale=None, apply_crop=None):
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            img_array = self.apply_image_processing(img_array, apply_downscale, apply_crop)
            
            if target_size is not None:
                target_h, target_w = target_size
                img_array = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            return img_array
            
        except Exception as e:
            raise IOError(f"Error loading RGB image {img_path}: {e}")
    
    def find_image_file(self, filename, search_paths=None):
        if search_paths is None:
            search_paths = ['.', 'img', '../img', '../../img']
        
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename
        base_filename = os.path.basename(filename)
        
        for search_path in search_paths:
            search_dir = Path(search_path)
            if search_dir.exists():
                candidate = search_dir / base_filename
                if candidate.exists():
                    return str(candidate)
                
                for file_path in search_dir.rglob(base_filename):
                    if file_path.is_file():
                        return str(file_path)
                
                name_without_ext = Path(base_filename).stem
                for ext in self.supported_formats:
                    candidate_with_ext = search_dir / f"{name_without_ext}{ext}"
                    if candidate_with_ext.exists():
                        return str(candidate_with_ext)
        
        return None

    def create_output_folder(self, output_path):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return str(Path(output_path).resolve())