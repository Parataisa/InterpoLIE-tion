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
                 downscale_size=512, downscale=True, crop_center=False, save_intermediate_steps=False):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.sensitivity = sensitivity
        self.save_intermediate_steps = save_intermediate_steps
        
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
            
            result = detector.detect(image, skip_internal_downscale=True, save_intermediate_steps=self.save_intermediate_steps)
            detailed_metrics = detector.extract_detection_metrics(result['spectrum'])
            
            processing_time = time.time() - start_time
            if not silent:
                status = 'DETECTED' if result['detected'] else 'CLEAN'
                max_grad = detailed_metrics.get('max_gradient', 'N/A')
                threshold = detailed_metrics.get('gradient_threshold', 'N/A')
                print(f"    Result: {status} (gradient: {max_grad:.8f}, threshold: {threshold:.8f}, time: {processing_time:.2f}s)")

            max_gradient = detailed_metrics.get('max_gradient')
            if max_gradient is None or (isinstance(max_gradient, float) and np.isnan(max_gradient)):
                max_gradient = result.get('max_gradient', 0.0)
                if not silent:
                    print(f"    Warning: Using fallback gradient value: {max_gradient}")

            return {
                'file_name': img_path.name,
                'file_path': str(img_path),
                'detected': result['detected'],
                'processing_time': processing_time,
                'sensitivity': self.sensitivity,
                'p_map': result['p_map'],
                'spectrum': result['spectrum'],
                'prediction_error': result['prediction_error'],
                'gradient_map': result.get('gradient_map'),
                'detailed_metrics': detailed_metrics,
                'max_gradient': max_gradient,
                'gradient_threshold': detailed_metrics.get('gradient_threshold', 0.008),
                'spectrum_mean': detailed_metrics.get('spectrum_mean', 0),
                'spectrum_std': detailed_metrics.get('spectrum_std', 0),
                'spectrum_max': detailed_metrics.get('spectrum_max', 0),
            }
        except Exception as e:
            if not silent:
                print(f"ERROR processing {img_path}: {e}")
            return {
                'file_name': img_path.name, 
                'detected': False, 
                'error': str(e),
                'max_gradient': 0.0,
                'processing_time': 0.0,
                'gradient_threshold': 0.008,
                'gradient_map': None
            }

    def process_batch(self, save_visualizations=True, use_batch_max_gradient=False):
        images = self.file_handler.scan_folder(self.input_folder)
        
        print(f"Found {len(images)} images to process")
        print(f"Downscaling: {'Enabled' if self.file_handler.downscale else 'Disabled'} (target: {self.file_handler.downscale_size}px)")
        print(f"🎯 Using default Kirchner detector thresholds for sensitivity: {self.sensitivity}")

        if not images:
            return pd.DataFrame()

        results = []
        start_time = time.time()

        for i, img_path in tqdm(enumerate(images), total=len(images), desc="Processing images", unit="img"):
            try:
                result = self.process_single(img_path, silent=True)
                results.append(result)
                
                max_grad = result.get('max_gradient')
                if max_grad is None or (isinstance(max_grad, float) and np.isnan(max_grad)):
                    tqdm.write(f"⚠️  Gradient issue: {img_path.name} -> {max_grad}")
                    
            except Exception as e:
                tqdm.write(f"❌ Error: {img_path.name}: {e}")
                results.append({
                    'file_name': img_path.name, 
                    'error': str(e), 
                    'detected': False,
                    'max_gradient': 0.0,
                    'processing_time': 0.0,
                    'gradient_threshold': 0.008,
                    'gradient_map': None
                })

        print(f"\n🔍 Validating gradient data...")
        valid_gradients = 0
        invalid_gradients = 0
        
        for result in results:
            max_grad = result.get('max_gradient')
            if max_grad is not None and not (isinstance(max_grad, float) and np.isnan(max_grad)):
                valid_gradients += 1
            else:
                invalid_gradients += 1
                if 'detailed_metrics' in result and result['detailed_metrics']:
                    backup_grad = result['detailed_metrics'].get('max_gradient')
                    if backup_grad is not None and not (isinstance(backup_grad, float) and np.isnan(backup_grad)):
                        result['max_gradient'] = backup_grad
                        valid_gradients += 1
                        invalid_gradients -= 1
                        print(f"   Fixed gradient for {result.get('file_name', 'unknown')}")

        print(f"   Valid gradients: {valid_gradients}/{len(results)}")
        if invalid_gradients > 0:
            print(f"   ⚠️  Invalid gradients: {invalid_gradients}")

        if use_batch_max_gradient:
            valid_gradients_list = [r.get('max_gradient', 0) for r in results if r.get('max_gradient') is not None and not (isinstance(r.get('max_gradient'), float) and np.isnan(r.get('max_gradient')))]
            if len(valid_gradients_list) >= 2:
                sorted_gradients = sorted(valid_gradients_list, reverse=True)
                max_gradient_threshold = sorted_gradients[0]  # For Second highest value Change to 1
                print(f"\n🔄 Re-evaluating results with SECOND HIGHEST gradient threshold: {max_gradient_threshold:.8f}")
                print(f"   Highest gradient found: {sorted_gradients[0]:.8f} (excluded from threshold)")
                
                updated_detections = 0
                for result in results:
                    if 'error' not in result and result.get('max_gradient') is not None:
                        old_detected = result.get('detected', False)
                        new_detected = result['max_gradient'] > max_gradient_threshold
                        result['detected'] = new_detected
                        result['gradient_threshold'] = max_gradient_threshold
                        
                        if 'detailed_metrics' in result and result['detailed_metrics']:
                            result['detailed_metrics']['gradient_threshold'] = max_gradient_threshold
                        
                        if old_detected != new_detected:
                            updated_detections += 1
                
                print(f"   Updated {updated_detections} detection results")
            elif len(valid_gradients_list) == 1:
                max_gradient_threshold = valid_gradients_list[0]
                print(f"\n⚠️  Only one valid gradient found, using it as threshold: {max_gradient_threshold:.8f}")
            else:
                print(f"⚠️  No valid gradients found for re-evaluation")

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
        
        valid_gradients_list = [r.get('max_gradient', 0) for r in results if r.get('max_gradient') is not None and not (isinstance(r.get('max_gradient'), float) and np.isnan(r.get('max_gradient')))]
        if valid_gradients_list:
            max_gradient = max(valid_gradients_list)
            mean_gradient = sum(valid_gradients_list) / len(valid_gradients_list)
            threshold_used = results[0].get('gradient_threshold', 0.008) if results else 0.008
            above_threshold = sum(1 for g in valid_gradients_list if g > threshold_used)
            
            print(f"\n🎯 Final Gradient Analysis Summary:")
            if use_batch_max_gradient:
                print(f"   Threshold used: {threshold_used:.8f} (max gradient)")
            else:
                print(f"   Threshold used: {threshold_used:.8f} (default)")
            print(f"   Max gradient: {max_gradient:.8f}")
            print(f"   Mean gradient: {mean_gradient:.8f}")
            print(f"   Values above threshold: {above_threshold}/{len(valid_gradients_list)} ({above_threshold/len(valid_gradients_list)*100:.1f}%)")
        
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

def quick_scan(input_folder, output_folder=None, sensitivity='medium', downscale_size=512, downscale=True, crop_center=False, save_visualizations=True, save_intermediate_steps=False, use_batch_max_gradient=False):
    if output_folder is None:
        output_folder = "results"

    processor = BatchProcessor(input_folder, output_folder, sensitivity, downscale_size, downscale, crop_center=crop_center, save_intermediate_steps=save_intermediate_steps)
    return processor.process_batch(save_visualizations=save_visualizations, use_batch_max_gradient=use_batch_max_gradient)