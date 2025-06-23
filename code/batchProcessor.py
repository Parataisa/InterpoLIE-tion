import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from pathlib import Path
from visualizations import create_batch_visualization
from fileHandler import FileHandler
from analysisReport import AnalysisReports
from kirchner import KirchnerDetector

matplotlib.use('Agg')

class BatchProcessor:
    def __init__(self, input_folder, output_folder, sensitivity='medium', 
                 downscale_size=512, downscale=True, crop_center=False):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.sensitivity = sensitivity
        
        self.file_handler = FileHandler(downscale_size, downscale, crop_center=crop_center)
        self.file_handler.create_output_folder(self.output_folder)

    def process_single(self, img_path):
        try:
            print(f"  Processing: {img_path.name}")
            start_time = time.time()
            
            image = self.file_handler.load_image(img_path)
            print(f"    Image loaded, size: {image.shape}")
            
            
            detector = KirchnerDetector(sensitivity=self.sensitivity, 
                                        downscale_size=self.file_handler.downscale_size, 
                                        downscale=False) 
            
            result = detector.detect(image, skip_internal_downscale=True)
            detailed_metrics = detector.extract_detection_metrics(result['spectrum'])
            
            processing_time = time.time() - start_time
            print(f"    Result: {'DETECTED' if result['detected'] else 'CLEAN'} ({processing_time:.2f}s)")

            return {
                'file_name': img_path.name,
                'file_path': str(img_path),
                'detected': result['detected'],
                'processing_time': processing_time,
                'sensitivity': self.sensitivity,
                'p_map': result['p_map'],
                'spectrum': result['spectrum'],
                'prediction_error': result['prediction_error'],
                'detailed_metrics': detailed_metrics,
                'max_gradient': detailed_metrics.get('max_gradient', 0),
                'gradient_threshold': detailed_metrics.get('gradient_threshold', 0.008),
                'spectrum_mean': detailed_metrics.get('spectrum_mean', 0),
                'spectrum_std': detailed_metrics.get('spectrum_std', 0),
                'spectrum_max': detailed_metrics.get('spectrum_max', 0),
            }
                
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            return {
                'file_name': img_path.name,
                'detected': None,
                'error': str(e)
            }

    def process_batch(self, save_visualizations=True):
        images = self.file_handler.scan_folder(self.input_folder)
        
        print(f"Found {len(images)} images to process")
        print(f"Downscaling: {'Enabled' if self.file_handler.downscale else 'Disabled'} (target: {self.file_handler.downscale_size}px)")

        if not images:
            return pd.DataFrame()

        results = []
        start_time = time.time()

        for i, img_path in enumerate(images):
            print(f"Processing {i+1}/{len(images)}: {img_path.name}")
            result = self.process_single(img_path)
            results.append(result)

        df = pd.DataFrame(results)
        csv_path = self.output_folder / 'results.csv'
        df.to_csv(csv_path, index=False)

        if save_visualizations:
            self.create_visualizations(results)
            
        AnalysisReports.create_batch_analysis_report(results, self.output_folder)

        total_time = time.time() - start_time
        self.print_summary(results, total_time, csv_path)
        return df

    def create_visualizations(self, results):
        vis_folder = self.output_folder / 'visualizations'
        self.file_handler.create_output_folder(vis_folder)

        print(f"Creating visualizations for {len([r for r in results if 'error' not in r])} valid results...")

        for result in results:
            if 'error' not in result and 'p_map' in result and result['p_map'] is not None:
                try:
                    print(f"  Creating visualization for: {result['file_name']}")
                    
                    if 'file_path' not in result or not result['file_path']:
                        result['file_path'] = str(self.input_folder / result['file_name'])
                    
                    create_batch_visualization(result, vis_folder, 
                                             self.file_handler.crop_center, 
                                             self.file_handler.downscale_size)
                    print(f"    ✓ Visualization created successfully")
                except Exception as e:
                    print(f"    ✗ Could not create visualization for {result['file_name']}: {e}")

    def print_summary(self, results, total_time, csv_path):
        detected_count = sum(1 for r in results if r.get('detected'))
        error_count = sum(1 for r in results if 'error' in r)

        print(f"\nSUMMARY:")
        print(f"Total: {len(results)}")
        print(f"Detected: {detected_count}")
        print(f"Errors: {error_count}")
        print(f"Time: {total_time:.1f}s")
        print(f"Results: {csv_path}")

def quick_scan(input_folder, output_folder=None, sensitivity='medium', downscale_size=512, downscale=True, crop_center=False):
    if output_folder is None:
        output_folder = "results"

    processor = BatchProcessor(input_folder, output_folder, sensitivity, downscale_size, downscale, crop_center=crop_center)
    return processor.process_batch()