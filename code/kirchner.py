import numpy as np
import cv2
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Pure Kirchner fast resampling detector implementation
Based on: "Fast and reliable resampling detection by spectral analysis 
of fixed linear predictor residue" (2008)
"""

class KirchnerDetector:
    def __init__(self, sensitivity='medium'):
        # Fixed predictor coefficients (Kirchner's optimal choice)
        self.predictor = np.array([
            [-0.25,  0.50, -0.25],
            [ 0.50,  0.00,  0.50],
            [-0.25,  0.50, -0.25]
        ])
        
        # Kirchner's contrast enhancement parameters
        self.lambda_param = 1.0
        self.tau = 2.0  
        self.sigma = 1.0
        
        # Detection thresholds based on sensitivity level
        thresholds = {
            'low': {'gradient': 0.001, 'peak_ratio': 0.003, 'max_peak': 0.10},
            'medium': {'gradient': 0.02, 'peak_ratio': 0.005, 'max_peak': 0.15},
            'high': {'gradient': 0.04, 'peak_ratio': 0.008, 'max_peak': 0.20}
        }
        
        t = thresholds.get(sensitivity, thresholds['high'])
        self.gradient_threshold = t['gradient']
        self.peak_ratio_threshold = t['peak_ratio'] 
        self.max_peak_threshold = t['max_peak']
    
    def detect(self, img_path):
        # Step 1: Load and preprocess image
        img = self._load_image(img_path)
        
        # Step 2: Apply fixed linear predictor
        prediction_error = cv2.filter2D(img, -1, self.predictor, 
                                      borderType=cv2.BORDER_REFLECT)
        
        # Step 3: Generate contrast-enhanced p-map
        p_map = self._generate_kirchner_pmap(prediction_error)
        
        # Step 4: Compute DFT and apply contrast function
        enhanced_spectrum = self._compute_enhanced_spectrum(p_map)
        
        # Step 5: Detect peaks in spectrum
        is_resampled = self._detect_peaks(enhanced_spectrum)
        
        return {
            'detected': is_resampled,
            'p_map': p_map,
            'spectrum': enhanced_spectrum,
            'prediction_error': prediction_error
        }
    
    def _load_image(self, img_path):
        img = np.array(Image.open(img_path))
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        
        if max(img.shape) > 2048:
            scale = 2048 / max(img.shape)
            h, w = int(img.shape[0] * scale), int(img.shape[1] * scale)
            img = cv2.resize(img, (w, h))
        
        img = img.astype(np.float64)
        if img.max() > 1:
            img /= 255.0
            
        return img
    
    def _generate_kirchner_pmap(self, prediction_error):
        """
        Generate p-map using Kirchner's contrast function:
        p = λ * exp(-|e|^tau / sigma)
        """
        abs_error = np.abs(prediction_error)
        p_map = self.lambda_param * np.exp(-(abs_error ** self.tau) / self.sigma)
        
        return (p_map - p_map.min()) / (p_map.max() - p_map.min() + 1e-8)
    
    def _compute_enhanced_spectrum(self, p_map):
        """
        Compute DFT and apply Kirchner's contrast enhancement
        """
        rows, cols = p_map.shape
        
        p_map_centered = p_map - np.mean(p_map)
        window = np.outer(np.hanning(rows), np.hanning(cols))
        windowed = p_map_centered * window
        
        # Compute DFT
        spectrum = fft2(windowed)
        spectrum = fftshift(spectrum)
        magnitude_spectrum = np.abs(spectrum)
        enhanced = self._apply_contrast_function(magnitude_spectrum)
        
        return enhanced
    
    def _apply_contrast_function(self, spectrum):
        rows, cols = spectrum.shape
        center_r, center_c = rows // 2, cols // 2
        
        y, x = np.ogrid[:rows, :cols]
        r = np.sqrt((x - center_c)**2 + (y - center_r)**2)
        r_norm = r / (min(rows, cols) // 2)
        
        radial_filter = np.where(r_norm <= 0.5, r_norm**2, 1.0)
        filtered = spectrum * radial_filter
        
        gamma = 0.5
        normalized = filtered / (np.max(filtered) + 1e-8)
        gamma_corrected = normalized ** gamma
        
        enhanced = gamma_corrected * np.max(filtered)
        
        return enhanced
    
    def _detect_peaks(self, spectrum):
        rows, cols = spectrum.shape
        center_r, center_c = rows // 2, cols // 2
        
        exclude_radius = min(rows, cols) // 10
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_c)**2 + (y - center_r)**2)
        mask = distance >= exclude_radius
        
        first_quad = spectrum[center_r:, center_c:]
        quad_mask = mask[center_r:, center_c:]
        
        if not np.any(quad_mask):
            return False
        
        values = first_quad[quad_mask]
        if len(values) == 0:
            return False
        
        sorted_vals = np.sort(values)
        cumulative = np.cumsum(sorted_vals)
        cumulative = cumulative / (cumulative[-1] + 1e-8)
        
        max_gradient = np.max(np.diff(cumulative)) if len(cumulative) > 1 else 0
        mean_val = np.mean(values)
        max_peak = np.max(values)
        threshold = mean_val + 3 * np.std(values)
        peak_ratio = np.sum(values > threshold) / len(values)
        
        gradient_detected = max_gradient > self.gradient_threshold
        peak_detected = (peak_ratio > self.peak_ratio_threshold and 
                        max_peak > self.max_peak_threshold)
        
        return gradient_detected or peak_detected
        
    def visualize_kirchner_analysis(self, img_path, save_path=None):
        result = self.detect(img_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        status = "DETECTED" if result['detected'] else "NOT DETECTED"
        fig.suptitle(f'Kirchner Analysis: {status}', 
                    fontsize=14, fontweight='bold',
                    color='red' if result['detected'] else 'green')
        
        # 1. P-map
        im1 = axes[0, 0].imshow(result['p_map'], cmap='hot', vmin=0, vmax=1)
        axes[0, 0].set_title('Probability Map')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
        # 2. Prediction Error
        error_range = np.percentile(result['prediction_error'], [5, 95])
        im2 = axes[0, 1].imshow(result['prediction_error'], cmap='RdBu_r', 
                              vmin=error_range[0], vmax=error_range[1])
        axes[0, 1].set_title('Prediction Error')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        
        # 3. Enhanced Spectrum
        spectrum_log = np.log(result['spectrum'] + 1e-6)
        rows, cols = result['spectrum'].shape
        freq_x = np.linspace(-0.5, 0.5, cols)
        freq_y = np.linspace(-0.5, 0.5, rows)
        
        im3 = axes[1, 0].imshow(spectrum_log, cmap='viridis',
                              extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]])
        axes[1, 0].set_title('Log Spectrum')
        axes[1, 0].axhline(0, color='white', alpha=0.5)
        axes[1, 0].axvline(0, color='white', alpha=0.5)
        plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        
        # 4. Error Distribution
        error_flat = result['prediction_error'].flatten()
        axes[1, 1].hist(error_flat, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(error_flat), color='red', linestyle='--')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return result


def save_visualization(filename, p_map, spectrum, prediction_error, detected, output_folder):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{filename} - {"DETECTED" if detected else "NOT DETECTED"}', 
                 fontsize=14, fontweight='bold',
                 color='red' if detected else 'green')
    
    im1 = axes[0, 0].imshow(p_map, cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title('Probability Map')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    error_range = np.percentile(prediction_error, [5, 95])
    im2 = axes[0, 1].imshow(prediction_error, cmap='RdBu_r', 
                           vmin=error_range[0], vmax=error_range[1])
    axes[0, 1].set_title('Prediction Error')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    spectrum_log = np.log(spectrum + 1e-6)
    rows, cols = spectrum.shape
    freq_x = np.linspace(-0.5, 0.5, cols)
    freq_y = np.linspace(-0.5, 0.5, rows)
    
    im3 = axes[1, 0].imshow(spectrum_log, cmap='viridis',
                           extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]])
    axes[1, 0].set_title('Log Spectrum')
    axes[1, 0].axhline(0, color='white', alpha=0.5)
    axes[1, 0].axvline(0, color='white', alpha=0.5)
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    error_flat = prediction_error.flatten()
    axes[1, 1].hist(error_flat, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(error_flat), color='red', linestyle='--')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_folder / f'{filename.split(".")[0]}_analysis.png'
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

class BatchProcessor:
    def __init__(self, input_folder, output_folder, sensitivity='medium', max_workers=12):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.detector = KirchnerDetector(sensitivity)
        self.max_workers = max_workers
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def scan_images(self):
        images = []
        for file_path in self.input_folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                images.append(file_path)
        return images
    
    def process_single(self, img_path):
        try:
            start_time = time.time()
            result = self.detector.detect(str(img_path))
            processing_time = time.time() - start_time
            
            return {
                'file_name': img_path.name,
                'detected': result['detected'],
                'processing_time': processing_time,
                'p_map': result['p_map'],
                'spectrum': result['spectrum'], 
                'prediction_error': result['prediction_error']
            }
        except Exception as e:
            return {
                'file_name': img_path.name,
                'detected': None,
                'error': str(e)
            }
    
    def process_batch(self, save_visualizations=True):
        images = self.scan_images()
        print(f"Found {len(images)} images")
        
        if not images:
            return pd.DataFrame()
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_single, img): img for img in images}
            
            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)
                
                status = 'DETECTED' if result.get('detected') else 'NOT DETECTED'
                if 'error' in result:
                    status = 'ERROR'
                
                progress = ((i + 1) / len(images)) * 100
                print(f"{progress:5.1f}% - {result['file_name']}: {status}")
        
        df = pd.DataFrame(results)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_folder / f'results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        if save_visualizations:
            vis_folder = self.output_folder / 'visualizations'
            vis_folder.mkdir(exist_ok=True)
            
            for result in results:
                if 'p_map' in result and result['p_map'] is not None:
                    try:
                        save_visualization(
                            result['file_name'],
                            result['p_map'],
                            result['spectrum'],
                            result['prediction_error'],
                            result['detected'],
                            vis_folder
                        )
                    except Exception:
                        pass
        
        total_time = time.time() - start_time
        detected_count = sum(1 for r in results if r.get('detected'))
        error_count = sum(1 for r in results if 'error' in r)
        
        print(f"\nSUMMARY:")
        print(f"Total: {len(results)}")
        print(f"Detected: {detected_count}")
        print(f"Errors: {error_count}")
        print(f"Time: {total_time:.1f}s")
        print(f"Results: {csv_path}")
        
        return df

def quick_scan(input_folder, output_folder=None, sensitivity='medium'):
    if output_folder is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_folder = f"results_{timestamp}"
    
    processor = BatchProcessor(input_folder, output_folder, sensitivity)
    return processor.process_batch()

def detect_single_image(image_path, sensitivity='medium', save_plot=False):
    detector = KirchnerDetector(sensitivity)
    result = detector.detect(image_path)
    
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Resampling detected: {'YES' if result['detected'] else 'NO'}")
    
    if save_plot:
        output_folder = Path(os.path.dirname(image_path) or '.')
        save_visualization(
            os.path.basename(image_path),
            result['p_map'],
            result['spectrum'],
            result['prediction_error'],
            result['detected'],
            output_folder
        )
        print(f"Plot saved to: {output_folder}")
    
    return result['detected']

def test_kirchner_sensitivity(img_path):
    """Test Kirchner detector with different sensitivity levels"""
    print(f"Testing Kirchner detector on: {img_path}")
    print("=" * 50)
    
    base_path = Path('.')
    img_name = Path(img_path).stem
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    test_folder = base_path / f"test_results_{img_name}_{timestamp}"
    test_folder.mkdir(parents=True, exist_ok=True)
    
    detection_metrics = {}
    
    for sensitivity in ['low', 'medium', 'high']:
        detector = KirchnerDetector(sensitivity=sensitivity)
        
        result = detector.detect(img_path)
        
        rows, cols = result['spectrum'].shape
        center_r, center_c = rows // 2, cols // 2
        
        exclude_radius = min(rows, cols) // 10
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_c)**2 + (y - center_r)**2)
        mask = distance >= exclude_radius
        first_quad = result['spectrum'][center_r:, center_c:]
        quad_mask = mask[center_r:, center_c:]
        
        if np.any(quad_mask):
            values = first_quad[quad_mask]
            if len(values) > 0:
                sorted_vals = np.sort(values)
                cumulative = np.cumsum(sorted_vals)
                cumulative = cumulative / (cumulative[-1] + 1e-8)
                
                max_gradient = np.max(np.diff(cumulative)) if len(cumulative) > 1 else 0
                mean_val = np.mean(values)
                max_peak = np.max(values)
                threshold = mean_val + 3 * np.std(values)
                peak_ratio = np.sum(values > threshold) / len(values)
                
                detection_metrics[sensitivity] = {
                    'max_gradient': max_gradient,
                    'gradient_threshold': detector.gradient_threshold,
                    'peak_ratio': peak_ratio,
                    'peak_ratio_threshold': detector.peak_ratio_threshold,
                    'max_peak': max_peak,
                    'max_peak_threshold': detector.max_peak_threshold
                }
                
                save_visualization(
                    f"{img_name}_{sensitivity}",
                    result['p_map'],
                    result['spectrum'],
                    result['prediction_error'],
                    result['detected'],
                    test_folder
                )
        
        status = "DETECTED" if result['detected'] else "NOT DETECTED"
        print(f"{sensitivity.upper():>6}: {status}")
    
    print("\nDetection Metrics:")
    print("-" * 50)
    
    metrics_data = []
    
    for sensitivity, metrics in detection_metrics.items():
        print(f"\n{sensitivity.upper()} SENSITIVITY:")
        
        gradient_detected = metrics['max_gradient'] > metrics['gradient_threshold']
        print(f"  Max Gradient: {metrics['max_gradient']:.6f} (Threshold: {metrics['gradient_threshold']:.6f}) -> {'DETECTED' if gradient_detected else 'not detected'}")
        
        peak_ratio_detected = metrics['peak_ratio'] > metrics['peak_ratio_threshold']
        print(f"  Peak Ratio: {metrics['peak_ratio']:.6f} (Threshold: {metrics['peak_ratio_threshold']:.6f}) -> {'DETECTED' if peak_ratio_detected else 'not detected'}")
        
        max_peak_detected = metrics['max_peak'] > metrics['max_peak_threshold']
        print(f"  Max Peak: {metrics['max_peak']:.6f} (Threshold: {metrics['max_peak_threshold']:.6f}) -> {'DETECTED' if max_peak_detected else 'not detected'}")
        
        metrics_data.append({
            'sensitivity': sensitivity,
            'max_gradient': metrics['max_gradient'],
            'gradient_threshold': metrics['gradient_threshold'],
            'gradient_detected': gradient_detected,
            'peak_ratio': metrics['peak_ratio'],
            'peak_ratio_threshold': metrics['peak_ratio_threshold'],
            'peak_ratio_detected': peak_ratio_detected,
            'max_peak': metrics['max_peak'],
            'max_peak_threshold': metrics['max_peak_threshold'],
            'max_peak_detected': max_peak_detected
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(test_folder / f"{img_name}_metrics.csv", index=False)
    
    print("=" * 50)
    print(f"Test results saved to: {test_folder}")
    
    return test_folder

def run_demo():
    if os.path.exists('img'):
        print("Running demo...")
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_folder = f'results_{timestamp}'
        
        try:
            results = quick_scan('img', output_folder, sensitivity='high')
            print(f"Demo completed! Results in: {output_folder}")
            if not results.empty:
                detected = results['detected'].sum()
                print(f"Detected resampling in {detected}/{len(results)} images")
            return output_folder
        except Exception as e:
            print(f"Demo failed: {e}")
            return None
    else:
        print("No 'img' folder found for demo")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) > 2:
            test_kirchner_sensitivity(sys.argv[2])
        else:
            print("Usage: python kirchner.py test path/to/image.jpg")
    else:
        run_demo()