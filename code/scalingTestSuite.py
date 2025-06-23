import numpy as np
import cv2
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import glob

from pathlib import Path
from visualizations import create_scaling_visualization
from fileHandler import FileHandler
from analysisReport import AnalysisReports
from kirchner import KirchnerDetector
from tqdm import tqdm

class ScalingTestSuite:
    def __init__(self, scaling_factors=None, interpolation_methods=None, crop_center=False):
        self.scaling_factors = scaling_factors or [0.5, 0.8, 1.2, 1.6, 2.0]
        self.crop_center = crop_center
        self.file_handler = FileHandler(crop_center=crop_center)
            
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
            
            return {
                'file_name': Path(img_path).name,
                'detected': result['detected'],
                'processing_time': processing_time,
                'p_map': result['p_map'],
                'spectrum': result['spectrum'],
                'prediction_error': result['prediction_error'],
                'detailed_metrics': detailed_metrics
            }
        except Exception as e:
            return {'file_name': Path(img_path).name, 'detected': None, 'error': str(e)}

    def create_scaled_images(self, input_folder, output_folder, source_downscale_size=512, source_downscale=True):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        self.file_handler.create_output_folder(output_path)
        images = self.file_handler.scan_folder(input_folder)
        
        self.file_handler.downscale_size = source_downscale_size
        self.file_handler.downscale = source_downscale
        
        print(f"Creating scaled versions of {len(images)} images...")
        print(f"Initial processing: {'Enabled' if source_downscale else 'Disabled'} (target: {source_downscale_size}px)")
        print(f"Crop center: {'Enabled' if self.crop_center else 'Disabled'}")
        
        created_images = []
        
        for img_idx, img_path in tqdm(enumerate(images), total=len(images), desc="Processing images", unit="img"):
            try:
                img = self.file_handler.load_image(img_path, apply_downscale=source_downscale)
                original_name = img_path.stem
                original_ext = img_path.suffix
                image_folder = output_path / original_name
                self.file_handler.create_output_folder(image_folder)
                
                orig_h, orig_w = img.shape[:2]
                original_copy = image_folder / f"{original_name}_original{original_ext}"
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                cv2.imwrite(str(original_copy), img_uint8)
                created_images.append({
                    'file_path': str(original_copy),
                    'original_name': original_name,
                    'scaling_factor': 1.0,
                    'interpolation': 'original',
                    'category': 'original'
                })
                
                for scale_factor in self.scaling_factors:
                    for interp_name, interp_method in self.interpolation_methods.items():
                        try:
                            new_h, new_w = int(orig_h * scale_factor), int(orig_w * scale_factor)
                            
                            if scale_factor < 3.0:
                                scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp_method)
                                final_img = scaled_img
                            else:
                                scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp_method)
                                center_y, center_x = new_h // 2, new_w // 2
                                top = center_y - orig_h // 2
                                bottom = top + orig_h
                                left = center_x - orig_w // 2
                                right = left + orig_w
                                
                                # Boundary checks
                                if top < 0:
                                    bottom -= top
                                    top = 0
                                if left < 0:
                                    right -= left
                                    left = 0
                                if bottom > new_h:
                                    diff = bottom - new_h
                                    bottom = new_h
                                    top = max(0, top - diff)
                                if right > new_w:
                                    diff = right - new_w
                                    right = new_w
                                    left = max(0, left - diff)
                                
                                final_img = scaled_img[top:bottom, left:right]
                                
                                if final_img.shape[0] != orig_h or final_img.shape[1] != orig_w:
                                    final_img = cv2.resize(final_img, (orig_w, orig_h), 
                                                        interpolation=cv2.INTER_LINEAR)
                            
                            scaled_name = f"{original_name}_scale{scale_factor:.1f}_{interp_name}{original_ext}"
                            scaled_path = image_folder / scaled_name
                            
                            scaled_img_uint8 = np.clip(final_img, 0, 255).astype(np.uint8)
                            cv2.imwrite(str(scaled_path), scaled_img_uint8)
                            
                            actual_h, actual_w = final_img.shape[:2]
                            created_images.append({
                                'file_path': str(scaled_path),
                                'original_name': original_name,
                                'scaling_factor': scale_factor,
                                'interpolation': interp_name,
                                'category': 'downscaled' if scale_factor < 1.0 else 'upscaled',
                                'dimensions': (actual_h, actual_w)
                            })
                        except Exception as scale_error:
                            tqdm.write(f"Warning: Failed {scale_factor:.1f}x {interp_name} for {original_name}: {scale_error}")
                            
            except Exception as e:
                tqdm.write(f"Error processing {img_path}: {e}")
        
        print(f"\nCreated {len(created_images)} test images")
        return created_images, str(output_path)
    def run_scaling_test(self, input_folder, output_folder=None, sensitivity='medium', detector_class=None, create_visualizations=True, downscale_size=512, downscale=True):
        if output_folder is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_folder = f'scaling_test_{timestamp}'
        
        output_path = Path(output_folder)
        self.file_handler.create_output_folder(output_path)
        self.file_handler.downscale_size = downscale_size
        self.file_handler.downscale = downscale
        
        print("=== STEP 1: Creating scaled test images ===")
        scaled_images_folder = output_path / 'scaled_images'
        created_images, scaled_folder = self.create_scaled_images(input_folder, scaled_images_folder, 
                                                                downscale_size, downscale)
        
        print("\n=== STEP 2: Running Kirchner detection ===")
        detector_class = detector_class or KirchnerDetector
        detector = detector_class(sensitivity=sensitivity, downscale_size=downscale_size, downscale=False)

        images = self.file_handler.scan_folder(scaled_folder)
        print(f"Found {len(images)} images to process")
        
        results = []
        
        for i, img_path in tqdm(enumerate(images), total=len(images), desc="Running detection", unit="img"):
            try:
                result = self.process_with_detailed_metrics(img_path, detector)
                results.append(result)
            except Exception as e:
                tqdm.write(f"Error processing {img_path.name}: {e}")
                results.append({'file_name': img_path.name, 'error': str(e), 'detected': False})
        
        detection_count = sum(1 for r in results if r.get('detected', False))
        print(f"\n✅ Detection phase completed: {detection_count}/{len(images)} flagged as resampled")
        
        print("\n=== STEP 3: Analyzing results ===")
        analysis_results = AnalysisReports.analyze_scaling_results(created_images, results, output_path)
        
        print("\n=== STEP 4: Creating analysis report ===")
        AnalysisReports.create_scaling_report(analysis_results, output_path)
        
        if create_visualizations:
            print("\n=== STEP 5: Creating visualizations ===")
            self.create_visualizations(results, created_images, output_path, downscale_size)
        
        print(f"\n🎉 Scaling test completed! Results in: {output_path}")
        print(f"📊 Summary: {detection_count}/{len(images)} images flagged as resampled")
        
        return analysis_results

    def create_visualizations(self, results, created_images, output_path, downscale_size):
        vis_folder = output_path / 'visualizations'
        self.file_handler.create_output_folder(vis_folder)
        
        file_path_lookup = {Path(img['file_path']).name: img['file_path'] for img in created_images}
        valid_results = [result for result in results if 'error' not in result and result['p_map'] is not None]
        
        results_by_image = {}
        for result in valid_results:
            filename = result['file_name']
            if '_scale' in filename:
                original_name = filename.split('_scale')[0]
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
        total_items = sum(len(image_results) for image_results in results_by_image.items())
        
        progress_items = []
        for original_name, image_results in results_by_image.items():
            for result in image_results:
                progress_items.append((original_name, result))
        
        for original_name, result in tqdm(progress_items, desc="Creating visualizations", unit="img"):
            try:
                image_vis_folder = vis_folder / original_name
                filename = result['file_name']
                
                if '_scale' in filename:
                    scale_part = filename.split('_scale')[1]
                    parts = scale_part.split('_')
                    scaling_factor = float(parts[0]) if len(parts) >= 2 else 1.0
                    interpolation_method = parts[1].split('.')[0] if len(parts) >= 2 else 'unknown'
                else:
                    scaling_factor = 1.0
                    interpolation_method = 'original'
                
                file_path = file_path_lookup.get(filename)
                
                create_scaling_visualization(
                    result['file_name'],
                    result['p_map'],
                    result['spectrum'],
                    result['prediction_error'],
                    result['detected'],
                    scaling_factor,
                    interpolation_method,
                    result['detailed_metrics'],
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

def run_scaling_test(input_folder, scaling_factors=None, interpolation_methods=None, sensitivity='medium', output_folder=None, detector_class=None, create_visualizations=True, downscale_size=512, downscale=True, crop_center=False):
    test_suite = ScalingTestSuite(scaling_factors=scaling_factors, interpolation_methods=interpolation_methods, crop_center=crop_center)
    return test_suite.run_scaling_test(input_folder, output_folder, sensitivity, detector_class, create_visualizations, downscale_size, downscale)