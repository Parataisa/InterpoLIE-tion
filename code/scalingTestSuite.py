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

class ScalingTestSuite:
    def __init__(self, scaling_factors=None, crop_center=False):
        self.scaling_factors = scaling_factors or [0.5, 0.8, 1.2, 1.6, 2.0]
        self.crop_center = crop_center
        self.file_handler = FileHandler(crop_center=crop_center)
            
        self.interpolation_methods = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }

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
        
        for img_idx, img_path in enumerate(images):
            print(f"Processing image {img_idx + 1}/{len(images)}: {img_path.name}")
            try:
                print(f"    Loading and processing: {img_path.name}")
                img = self.file_handler.load_image(img_path, apply_downscale=source_downscale)
                print(f"    Processed image size: {img.shape} (crop_center: {self.crop_center})")
                
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
                    'scaling_factor': 1.0,
                    'interpolation': 'original',
                    'category': 'original'
                })

                h, w = img.shape[:2]
                print(f"    Creating scaled versions from {w}x{h} processed image...")
                for scale_factor in self.scaling_factors:
                    for interp_name, interp_method in self.interpolation_methods.items():
                        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                        try:
                            scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp_method)
                            
                            scaled_name = f"{original_name}_scale{scale_factor:.1f}_{interp_name}{original_ext}"
                            scaled_path = image_folder / scaled_name
                            scaled_img_uint8 = np.clip(scaled_img, 0, 255).astype(np.uint8)
                            cv2.imwrite(str(scaled_path), scaled_img_uint8)
                            
                            created_images.append({
                                'file_path': str(scaled_path),
                                'original_name': original_name,
                                'scaling_factor': scale_factor,
                                'interpolation': interp_name,
                                'category': 'downscaled' if scale_factor < 1.0 else 'upscaled'
                            })
                            
                        except Exception as scale_error:
                            print(f"      Warning: Failed to create {scale_factor:.1f}x {interp_name} version: {scale_error}")
                            continue
                        
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue
        
        print(f"Created {len(created_images)} test images")
        return created_images, str(output_path)

    def process_with_detailed_metrics(self, img_path, detector):
        try:
            start_time = time.time()
            
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Could not load image: {img_path}")
                
            img = img.astype(np.float32)
            print(f"      Processing scaled image, size: {img.shape}")
            
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
            print(f"ERROR processing {img_path}: {e}")
            return {
                'file_name': Path(img_path).name,
                'detected': None,
                'error': str(e)
            }

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
        detector_class = KirchnerDetector
        detector = detector_class(sensitivity=sensitivity, downscale_size=downscale_size, downscale=False)

        images = self.file_handler.scan_folder(scaled_folder)
        print(f"Found {len(images)} images to process")
        
        results = []
        for i, img_path in enumerate(images):
            print(f"Processing {i+1}/{len(images)}: {img_path.name}")
            result = self.process_with_detailed_metrics(img_path, detector)
            results.append(result)
        
        if create_visualizations:
            print("\n=== STEP 3: Creating visualizations ===")
            self._create_visualizations(results, created_images, output_path, downscale_size)
        
        print("\n=== STEP 4: Analyzing results ===")
        analysis_results = AnalysisReports.analyze_scaling_results(created_images, results, output_path)
        
        print("\n=== STEP 5: Creating analysis report ===")
        AnalysisReports.create_scaling_report(analysis_results, output_path)
        
        print(f"\nScaling test completed! Results in: {output_path}")
        if 'overall_detection_rate' in analysis_results:
            print(f"Overall detection rate: {analysis_results['overall_detection_rate']:.3f}")
        return analysis_results

    def _create_visualizations(self, results, created_images, output_path, downscale_size):
        vis_folder = output_path / 'visualizations'
        self.file_handler.create_output_folder(vis_folder)
        
        results_by_image = {}
        for result in results:
            if 'error' not in result and result['p_map'] is not None:
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
        
        print(f"Creating visualizations for {len(results_by_image)} original images")
        
        visualization_count = 0
        for original_name, image_results in results_by_image.items():
            image_vis_folder = vis_folder / original_name
            self.file_handler.create_output_folder(image_vis_folder)
            
            for result in image_results:
                try:
                    filename = result['file_name']
                    scaling_factor = 1.0
                    interpolation_method = 'original'
                    
                    if '_scale' in filename:
                        parts = filename.split('_scale')[1].split('_')
                        if len(parts) >= 2:
                            scaling_factor = float(parts[0])
                            interpolation_method = parts[1].split('.')[0]
                    elif '_original' in filename:
                        scaling_factor = 1.0
                        interpolation_method = 'original'
                    
                    file_path = None
                    for created_image in created_images:
                        if Path(created_image['file_path']).name == filename:
                            file_path = created_image['file_path']
                            break
                    
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
                    print(f"    Warning: Could not create visualization for {result['file_name']}: {e}")
        
        print(f"Created {visualization_count} visualizations")

def run_scaling_test(input_folder, scaling_factors=None, sensitivity='medium', output_folder=None, detector_class=None, create_visualizations=True, downscale_size=512, downscale=True, crop_center=False):
    test_suite = ScalingTestSuite(scaling_factors=scaling_factors, crop_center=crop_center)
    return test_suite.run_scaling_test(input_folder, output_folder, sensitivity, detector_class, create_visualizations, downscale_size, downscale)