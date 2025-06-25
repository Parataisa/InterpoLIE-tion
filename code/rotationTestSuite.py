import numpy as np
import cv2
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import glob
import math

from pathlib import Path
from visualizations import create_rotation_visualization
from fileHandler import FileHandler
from analysisReport import AnalysisReports
from kirchner import KirchnerDetector
from tqdm import tqdm

class RotationTestSuite:
    def __init__(self, rotation_angles=None, interpolation_methods=None, crop_center=False, max_gradient=None):
        self.rotation_angles = rotation_angles or [5, 10, 15, 30, 45, 60, 90, 180, 270]
        self.crop_center = crop_center
        self.file_handler = FileHandler(crop_center=crop_center)
        self.max_gradient = max_gradient
            
        self.interpolation_methods = interpolation_methods or {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }

    def process_with_detailed_metrics(self, img_path, detector):
        try:
            start_time = time.time()
            
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Could not load image: {img_path}")
                
            img = img.astype(np.float32)
            
            result = detector.detect(img, skip_internal_downscale=True)
            detailed_metrics = detector.extract_detection_metrics(result['spectrum'])
            
            processing_time = time.time() - start_time
            
            max_gradient = detailed_metrics.get('max_gradient')
            if max_gradient is None or (isinstance(max_gradient, float) and np.isnan(max_gradient)):
                max_gradient = result.get('max_gradient', 0.0)
            
            return {
                'file_name': Path(img_path).name,
                'detected': result['detected'],
                'processing_time': processing_time,
                'max_gradient': max_gradient, 
                'gradient_threshold': detailed_metrics.get('gradient_threshold', 0.008),
                'spectrum_mean': detailed_metrics.get('spectrum_mean', 0.0),
                'spectrum_std': detailed_metrics.get('spectrum_std', 0.0),
                'spectrum_max': detailed_metrics.get('spectrum_max', 0.0),
                'p_map': result['p_map'],
                'spectrum': result['spectrum'],
                'prediction_error': result['prediction_error'],
                'detailed_metrics': detailed_metrics
            }
        except Exception as e:
            return {
                'file_name': Path(img_path).name, 
                'detected': False, 
                'error': str(e),
                'max_gradient': 0.0,
                'processing_time': 0.0,
                'gradient_threshold': 0.008,
                'spectrum_mean': 0.0,
                'spectrum_std': 0.0,
                'spectrum_max': 0.0
            }

    def rotate_image(self, img, angle, interpolation):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(img, rotation_matrix, (new_w, new_h), 
                                flags=interpolation, borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=0)
        
        if self.crop_center and (new_w > w or new_h > h):
            start_x = max(0, (new_w - w) // 2)
            start_y = max(0, (new_h - h) // 2)
            end_x = min(new_w, start_x + w)
            end_y = min(new_h, start_y + h)
            rotated = rotated[start_y:end_y, start_x:end_x]
        
        return rotated

    def create_rotated_images(self, input_folder, output_folder, source_downscale_size=512, source_downscale=True):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        self.file_handler.create_output_folder(output_path)
        images = self.file_handler.scan_folder(input_folder)
        
        self.file_handler.downscale_size = source_downscale_size
        self.file_handler.downscale = source_downscale
        
        print(f"Creating rotated versions of {len(images)} images...")
        print(f"Initial processing: {'Enabled' if source_downscale else 'Disabled'} (target: {source_downscale_size}px)")
        print(f"Crop center: {'Enabled' if self.crop_center else 'Disabled'}")
        print(f"Rotation angles: {self.rotation_angles}")
        
        created_images = []
        
        for img_idx, img_path in tqdm(enumerate(images), total=len(images), desc="Processing images", unit="img"):
            try:
                img = self.file_handler.load_image(img_path, apply_downscale=source_downscale)
                
                original_name = img_path.stem
                original_ext = img_path.suffix
                
                image_folder = output_path / original_name
                self.file_handler.create_output_folder(image_folder)
                
                original_copy = image_folder / f"{original_name}_original{original_ext}"
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                cv2.imwrite(str(original_copy), img_uint8)
                created_images.append({
                    'file_path': str(original_copy),
                    'original_name': original_name,
                    'rotation_angle': 0.0,
                    'interpolation': 'original',
                    'category': 'original'
                })

                for angle in self.rotation_angles:
                    for interp_name, interp_method in self.interpolation_methods.items():
                        try:
                            rotated_img = self.rotate_image(img, angle, interp_method)
                            
                            if isinstance(angle, (int, np.integer)) or angle.is_integer():
                                angle_str = f"{int(angle):03d}"
                            else:
                                angle_str = f"{angle:.1f}".replace(".", "p")
                            
                            rotated_name = f"{original_name}_rot{angle_str}_{interp_name}{original_ext}"
                            rotated_path = image_folder / rotated_name
                            rotated_img_uint8 = np.clip(rotated_img, 0, 255).astype(np.uint8)
                            cv2.imwrite(str(rotated_path), rotated_img_uint8)
                            
                            created_images.append({
                                'file_path': str(rotated_path),
                                'original_name': original_name,
                                'rotation_angle': angle,
                                'interpolation': interp_name,
                                'category': 'rotated'
                            })
                        except Exception as rotation_error:
                            tqdm.write(f"Warning: Failed {angle}° {interp_name} for {original_name}: {rotation_error}")
                            
            except Exception as e:
                tqdm.write(f"Error processing {img_path}: {e}")
        
        print(f"\nCreated {len(created_images)} test images")
        return created_images, str(output_path)

    def run_rotation_test(self, input_folder, output_folder=None, sensitivity='medium', detector_class=None, create_visualizations=True, downscale_size=512, downscale=True):
        if output_folder is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_folder = f'rotation_test_{timestamp}'
        
        output_path = Path(output_folder)
        self.file_handler.create_output_folder(output_path)
        self.file_handler.downscale_size = downscale_size
        self.file_handler.downscale = downscale
        
        print("=== STEP 1: Creating rotated test images ===")
        rotated_images_folder = output_path / 'rotated_images'
        created_images, rotated_folder = self.create_rotated_images(input_folder, rotated_images_folder, 
                                                                  downscale_size, downscale)
        
        print("\n=== STEP 2: Running Kirchner detection ===")
        detector_class = detector_class or KirchnerDetector
        detector = detector_class(sensitivity=sensitivity, downscale_size=downscale_size, downscale=False, max_gradient=self.max_gradient)

        images = self.file_handler.scan_folder(rotated_folder)
        print(f"Found {len(images)} images to process")
        
        results = []
        
        for i, img_path in tqdm(enumerate(images), total=len(images), desc="Running detection", unit="img"):
            try:
                result = self.process_with_detailed_metrics(img_path, detector)
                results.append(result)
                
                max_grad = result.get('max_gradient')
                if max_grad is None or (isinstance(max_grad, float) and np.isnan(max_grad)):
                    tqdm.write(f"⚠️  Gradient issue: {img_path.name} -> {max_grad}")
                    
            except Exception as e:
                tqdm.write(f"Error processing {img_path.name}: {e}")
                results.append({
                    'file_name': img_path.name, 
                    'error': str(e), 
                    'detected': False,
                    'max_gradient': 0.0,
                    'processing_time': 0.0,
                    'gradient_threshold': 0.008,
                    'spectrum_mean': 0.0,
                    'spectrum_std': 0.0,
                    'spectrum_max': 0.0
                })
        
        detection_count = sum(1 for r in results if r.get('detected', False))
        print(f"\n✅ Detection phase completed: {detection_count}/{len(images)} flagged as resampled")
        
        print("\n=== STEP 3: Analyzing results ===")
        analysis_results = AnalysisReports.analyze_rotation_results(created_images, results, output_path)
        
        print("\n=== STEP 4: Creating analysis report ===")
        AnalysisReports.create_rotation_report(analysis_results, output_path)
        
        if create_visualizations:
            print("\n=== STEP 5: Creating visualizations ===")
            self.create_visualizations(results, created_images, output_path, downscale_size)
        
        print(f"\n🎉 Rotation test completed! Results in: {output_path}")
        print(f"📊 Summary: {detection_count}/{len(images)} images flagged as resampled")
        
        return analysis_results

    def create_visualizations(self, results, created_images, output_path, downscale_size):
        vis_folder = output_path / 'visualizations'
        self.file_handler.create_output_folder(vis_folder)
        
        file_path_lookup = {Path(img['file_path']).name: img['file_path'] for img in created_images}
        valid_results = [result for result in results if 'error' not in result and result.get('p_map') is not None]
        
        results_by_image = {}
        for result in valid_results:
            filename = result['file_name']
            if '_rot' in filename:
                original_name = filename.split('_rot')[0]
            elif '_original' in filename:
                original_name = filename.split('_original')[0]
            else:
                original_name = filename.split('.')[0]
            
            if original_name not in results_by_image:
                results_by_image[original_name] = []
            results_by_image[original_name].append(result)
        
        for original_name in results_by_image.keys():
            image_vis_folder = vis_folder / original_name
            self.file_handler.create_output_folder(image_vis_folder)
        
        visualization_count = 0
        errors_count = 0
        
        progress_items = []
        for original_name, image_results in results_by_image.items():
            for result in image_results:
                progress_items.append((original_name, result))
        
        for original_name, result in tqdm(progress_items, desc="Creating visualizations", unit="img"):
            try:
                image_vis_folder = vis_folder / original_name
                filename = result['file_name']
                
                if '_rot' in filename:
                    rot_part = filename.split('_rot')[1]
                    parts = rot_part.split('_')
                    if 'p' in parts[0]: 
                        rotation_angle = float(parts[0].replace('p', '.'))
                    else: 
                        rotation_angle = int(parts[0])
                    interpolation_method = parts[1].split('.')[0] if len(parts) >= 2 else 'unknown'
                else:
                    rotation_angle = 0
                    interpolation_method = 'original'
                
                file_path = file_path_lookup.get(filename)
                
                create_rotation_visualization(
                    result['file_name'],
                    result.get('p_map'),
                    result.get('spectrum'),
                    result.get('prediction_error'),
                    result['detected'],
                    rotation_angle,
                    interpolation_method,
                    result.get('detailed_metrics', {}),
                    image_vis_folder,
                    file_path,
                    crop_center=self.crop_center,
                    downscale_size=downscale_size
                )
                visualization_count += 1
                
            except Exception as e:
                errors_count += 1
                tqdm.write(f"⚠️ Viz error {filename}: {e}")
        
        print(f"\n✅ Created {visualization_count} visualizations ({errors_count} errors)")
        return visualization_count, errors_count

def run_rotation_test(input_folder, rotation_angles=None, interpolation_methods=None, sensitivity='medium', output_folder=None, detector_class=None, create_visualizations=True, downscale_size=512, downscale=True, crop_center=False, max_gradient=None):
    test_suite = RotationTestSuite(rotation_angles=rotation_angles, interpolation_methods=interpolation_methods, crop_center=crop_center, max_gradient=max_gradient)
    return test_suite.run_rotation_test(input_folder, output_folder, sensitivity, detector_class, create_visualizations, downscale_size, downscale)