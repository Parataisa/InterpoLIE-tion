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
from tqdm import tqdm

matplotlib.use('Agg')

class BatchProcessor:
    def __init__(self, input_folder, output_folder, sensitivity='medium', 
                 downscale_size=512, downscale=True, crop_center=False):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.sensitivity = sensitivity
        
        self.file_handler = FileHandler(downscale_size, downscale, crop_center=crop_center)
        self.file_handler.create_output_folder(self.output_folder)

    def process_single(self, img_path, silent=False):
        try:
            if not silent:
                print(f"  Processing: {img_path.name}")
            start_time = time.time()
            
            image = self.file_handler.load_image(img_path)
            if not silent:
                print(f"    Image loaded, size: {image.shape}")
            
            detector = KirchnerDetector(sensitivity=self.sensitivity, 
                                        downscale_size=self.file_handler.downscale_size, 
                                        downscale=False) 
            
            result = detector.detect(image, skip_internal_downscale=True)
            detailed_metrics = detector.extract_detection_metrics(result['spectrum'])
            
            processing_time = time.time() - start_time
            if not silent:
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
            if not silent:
                print(f"ERROR processing {img_path}: {e}")
            return {'file_name': img_path.name, 'detected': None, 'error': str(e)}

    def process_batch(self, save_visualizations=True):
        images = self.file_handler.scan_folder(self.input_folder)
        
        print(f"Found {len(images)} images to process")
        print(f"Downscaling: {'Enabled' if self.file_handler.downscale else 'Disabled'} (target: {self.file_handler.downscale_size}px)")

        if not images:
            return pd.DataFrame()

        results = []
        start_time = time.time()

        for i, img_path in tqdm(enumerate(images), total=len(images), desc="Processing images", unit="img"):
            try:
                result = self.process_single(img_path, silent=True)
                results.append(result)
            except Exception as e:
                tqdm.write(f"❌ Error: {img_path.name}: {e}")
                results.append({'file_name': img_path.name, 'error': str(e), 'detected': False})

        print("\n💾 Saving results...")
        df = pd.DataFrame(results)
        csv_path = self.output_folder / 'results.csv'
        df.to_csv(csv_path, index=False)

        print("📊 Creating analysis report...")
        AnalysisReports.create_batch_analysis_report(results, self.output_folder)
        
        if save_visualizations:
            print("🎨 Creating visualizations...")
            try:
                self.create_visualizations(results)
                print("✅ Visualizations created")
            except Exception as e:
                print(f"⚠️  Could not create visualizations: {e}")

        total_time = time.time() - start_time
        detection_count = sum(1 for r in results if r.get('detected', False))
        
        print(f"\n✅ Batch processing completed!")
        print(f"📊 {detection_count}/{len(images)} images flagged as resampled")
        print(f"⏱️  Total time: {total_time:.2f} seconds ({len(images)/total_time:.2f} img/sec)")
        print(f"💾 Results: {csv_path}")
        
        return df

    def create_visualizations(self, results):
        vis_folder = self.output_folder / 'visualizations'
        self.file_handler.create_output_folder(vis_folder)

        valid_results = [result for result in results if 'error' not in result and 'p_map' in result and result['p_map'] is not None]
        
        if not valid_results:
            print("No valid results found for visualization")
            return

        batch_success = 0
        batch_errors = 0
        for result in tqdm(valid_results, desc="Creating visualizations", unit="img"):
            try:
                filename = result['file_name']
                if 'file_path' not in result or not result['file_path']:
                    result['file_path'] = str(self.input_folder / result['file_name'])
                
                create_batch_visualization(result, vis_folder, 
                                        self.file_handler.crop_center, 
                                        self.file_handler.downscale_size)
                batch_success += 1
            except Exception as e:
                batch_errors += 1
                tqdm.write(f"❌ Viz error {result['file_name']}: {e}")
        
        print(f"\n✅ Visualizations: {batch_success} successful, {batch_errors} errors")

    def print_summary(self, results, total_time, csv_path):
        detected_count = sum(1 for r in results if r.get('detected'))
        error_count = sum(1 for r in results if 'error' in r)

        print(f"\nSUMMARY:")
        print(f"Total: {len(results)}")
        print(f"Detected: {detected_count}")
        print(f"Errors: {error_count}")
        print(f"Time: {total_time:.1f}s")
        print(f"Results: {csv_path}")

def quick_scan(input_folder, output_folder=None, sensitivity='medium', downscale_size=512, downscale=True, crop_center=False, save_visualizations=True):
    if output_folder is None:
        output_folder = "results"

    processor = BatchProcessor(input_folder, output_folder, sensitivity, downscale_size, downscale, crop_center=crop_center)
    return processor.process_batch(save_visualizations=save_visualizations)