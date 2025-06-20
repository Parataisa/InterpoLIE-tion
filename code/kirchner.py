import numpy as np
import cv2
import os
import sys
import time
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import glob

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter, convolve
from PIL import Image
from matplotlib.colors import LogNorm

matplotlib.use('Agg')

"""
Kirchner fast resampling detector implementation
Based on: "Fast and reliable resampling detection by spectral analysis 
of fixed linear predictor residue" (2008)

Key Mathematical Foundations:
- Resampling: s(omega * chi) = sum h(omega * chi - chi_k) * s(chi_k) [Eq. 3]
- Prediction Error: e(omega * chi) = s(omega * chi) - sigma alpha_k * s(omega * chi + omega * k) [Eq. 5]
- Variance Periodicity: Var[e(x)] = Var[e(x + 1)] [Theorem 1]
- P-map: p = lambda_param * exp(-|e|^tau / sigma) [Eq. 21]
"""

class KirchnerDetector:
    def __init__(self, sensitivity='medium', lambda_param=1.0, tau=2.0, sigma=1.0, threshold_factor=10.0):
        """
        Initialize Kirchner detector with fixed linear predictor.
        
        Args:
            sensitivity: Detection sensitivity ('low', 'medium', 'high')
            lambda_param: P-map amplitude scaling factor
            tau: Error sensitivity parameter (>= 1) 
            sigma: Variance scaling parameter (> 0)
            threshold_factor: Peak detection threshold multiplier
        """
        # Kirchner's fixed 3x3 linear predictor filter
        self.predictor_filter = np.array([
            [-0.25, 0.50, -0.25],
            [0.50,  0.00,  0.50],
            [-0.25, 0.50, -0.25]
        ])
        
        # P-map generation parameters
        self.lambda_param = lambda_param
        self.tau = tau
        self.sigma = sigma
        self.threshold_factor = threshold_factor

        # Detection thresholds based on sensitivity level for fallback analysis
        thresholds = {
            'low':       {'gradient': 0.001, 'peak_ratio': 0.012, 'max_peak': 0.10},
            'medium':    {'gradient': 0.02,  'peak_ratio': 0.016, 'max_peak': 0.15},
            'high':      {'gradient': 0.04,  'peak_ratio': 0.020, 'max_peak': 0.20}
        }

        t = thresholds.get(sensitivity, thresholds['medium'])
        self.gradient_threshold = t['gradient']
        self.peak_ratio_threshold = t['peak_ratio']
        self.max_peak_threshold = t['max_peak']
        self.sensitivity = sensitivity

    def detect(self, img_path):
        try:
            print(f"      Loading image: {Path(img_path).name}")
            image = self._load_image(img_path)
            print(f"      Image shape: {image.shape}")
            
            print(f"      Running 6-step detection...")
            results = self.detect_resampling(image)
            print(f"      Detection complete: {results['decision']}")
            
            return {
                'detected': results['decision'],
                'p_map': results['p_map'],
                'spectrum': results['spectrum'],
                'prediction_error': results['error']
            }
        except Exception as e:
            print(f"      ERROR in detect(): {e}")
            raise

    def detect_resampling(self, image):
        try:
            # Step 1: Input preparation
            print(f"        Step 1: Input preparation")
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.astype(np.float64)
            
            # Step 2: Apply fixed linear predictor
            print(f"        Step 2: Applying predictor filter")
            predicted = convolve(image, self.predictor_filter, mode='reflect')
            
            # Step 3: Calculate prediction error
            print(f"        Step 3: Computing prediction error")
            error = image - predicted
            
            # Step 4: Generate contrast-enhanced P-map
            print(f"        Step 4: Generating P-map")
            p_map = self._generate_p_map(error)
            
            # Step 5: Spectral analysis via DFT
            print(f"        Step 5: Computing spectrum")
            spectrum = self._compute_spectrum(p_map)
            
            # Step 6: Peak detection and decision
            print(f"        Step 6: Peak detection")
            peaks = self._detect_peaks(spectrum)
            decision = self._make_decision(peaks)
            print(f"        Final decision: {decision}")
            
            results = {
                'p_map': p_map,
                'spectrum': spectrum,
                'peaks': peaks,
                'decision': decision,
                'error': error,
                'max_peak_strength': np.max(peaks['strengths']) if peaks['strengths'] else 0
            }
            
            return results
        except Exception as e:
            print(f"        ERROR in detect_resampling: {e}")
            raise

    def _load_image(self, img_path):
        try:
            if isinstance(img_path, str):
                img = np.array(Image.open(img_path))
            else:
                img = np.array(Image.open(str(img_path)))
                
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
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

    def _generate_p_map(self, error):
        """Step 4: Generate contrast-enhanced probability map"""
        # Kirchner's contrast function: p = lambda_param * exp(-|e|^tau / sigma)
        abs_error = np.abs(error)
        p_map = self.lambda_param * np.exp(-(abs_error ** self.tau) / self.sigma)
        
        # Normalize for consistency with other parts of pipeline
        return p_map

    def _compute_spectrum(self, p_map):
        """Step 5: Compute frequency spectrum using DFT"""
        # Apply 2D FFT and shift zero frequency to center
        fft_result = fft2(p_map)
        spectrum = np.abs(fftshift(fft_result))
        
        # Normalize spectrum
        spectrum = spectrum / (np.max(spectrum) + 1e-8)
        return spectrum

    def _detect_peaks(self, spectrum):
        """Step 6a: Detect characteristic peaks in spectrum (optimized version)"""
        print(f"        Detecting peaks in spectrum shape: {spectrum.shape}")
        
        peaks = {
            'positions': [],
            'strengths': [],
            'frequencies': []
        }
        
        # Get spectrum center and create frequency grid
        center = np.array(spectrum.shape) // 2
        h, w = spectrum.shape
        
        step_size = max(1, min(h, w) // 200)  
        search_range_h = min(h-10, center[0] + h//4)
        search_range_w = min(w-10, center[1] + w//4)
        
        print(f"        Search area: {search_range_h}x{search_range_w}, step_size: {step_size}")
        
        peak_count = 0
        max_peaks = 50  
        
        # Search for peaks (excluding DC component) with adaptive step size
        for i in range(5, search_range_h, step_size):
            for j in range(5, search_range_w, step_size):
                if peak_count >= max_peaks:
                    break
                    
                # Skip DC component area
                if abs(i - center[0]) < 3 and abs(j - center[1]) < 3:
                    continue
                
                # Check if current point is a local maximum
                local_region = spectrum[max(0, i-2):min(h, i+3), max(0, j-2):min(w, j+3)]
                if local_region.size == 0:
                    continue
                    
                local_max = np.max(local_region)
                local_mean = np.mean(local_region)
                
                # Peak detection criterion: peak > threshold_factor * local_average
                if (spectrum[i, j] == local_max and 
                    local_mean > 0 and  # Avoid division by zero
                    spectrum[i, j] > self.threshold_factor * local_mean and
                    spectrum[i, j] > 0.1):  # Minimum absolute threshold
                    
                    peaks['positions'].append((i, j))
                    peaks['strengths'].append(spectrum[i, j])
                    
                    # Calculate normalized frequency
                    freq_x = (j - center[1]) / w
                    freq_y = (i - center[0]) / h
                    peaks['frequencies'].append((freq_x, freq_y))
                    
                    peak_count += 1
            
            if peak_count >= max_peaks:
                break
        
        print(f"        Found {len(peaks['strengths'])} peaks")
        return peaks

    def _make_decision(self, peaks):
        """Step 6b: Make final resampling decision"""
        if not peaks['strengths']:
            return False
        
        # Decision based on strongest peak
        max_strength = np.max(peaks['strengths'])
        
        # Threshold for resampling detection (empirically determined)
        decision_threshold = 0.15
        
        return max_strength > decision_threshold

    def extract_detection_metrics(self, spectrum):
        # Try new peak detection method first
        peaks = self._detect_peaks(spectrum)
        
        # Fallback to gradient analysis for compatibility
        rows, cols = spectrum.shape
        center_r, center_c = rows // 2, cols // 2

        exclude_radius = min(rows, cols) // 10
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_c)**2 + (y - center_r)**2)
        mask = distance >= exclude_radius
        first_quad = spectrum[center_r:, center_c:]
        quad_mask = mask[center_r:, center_c:]

        if not np.any(quad_mask):
            return None
            
        values = first_quad[quad_mask]
        if len(values) == 0:
            return None

        sorted_vals = np.sort(values)
        cumulative = np.cumsum(sorted_vals)
        cumulative = cumulative / (cumulative[-1] + 1e-8)

        max_gradient = np.max(np.diff(cumulative)) if len(cumulative) > 1 else 0
        mean_val = np.mean(values)
        max_peak = np.max(values)
        threshold = mean_val + 3 * np.std(values)
        peak_ratio = np.sum(values > threshold) / len(values)

        # Combine new peak method with gradient analysis
        peak_count = len(peaks['strengths']) if isinstance(peaks, dict) else 0
        max_peak_strength = np.max(peaks['strengths']) if peak_count > 0 else max_peak

        return {
            'max_gradient': max_gradient,
            'gradient_threshold': self.gradient_threshold,
            'gradient_detected': max_gradient > self.gradient_threshold,
            'peak_ratio': peak_ratio,
            'peak_ratio_threshold': self.peak_ratio_threshold,
            'peak_ratio_detected': peak_ratio > self.peak_ratio_threshold,
            'max_peak': max_peak_strength,
            'max_peak_threshold': self.max_peak_threshold,
            'max_peak_detected': max_peak_strength > self.max_peak_threshold,
            'peak_count': peak_count,
            'kirchner_peaks': peak_count > 0
        }


def save_visualization(filename, p_map, spectrum, prediction_error, detected, output_folder):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{filename} - {"DETECTED" if detected else "NOT DETECTED"}',
                 fontsize=14, fontweight='bold',
                 color='red' if detected else 'green')

    # 1. P-map 
    im1 = axes[0, 0].imshow(p_map, cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title('Probability Map (P-Map)')
    axes[0, 0].set_xlabel('Pixel Column')
    axes[0, 0].set_ylabel('Pixel Row')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    # 2. Prediction Error
    error_range = np.percentile(prediction_error, [5, 95])
    im2 = axes[0, 1].imshow(prediction_error, cmap='RdBu_r',
                            vmin=error_range[0], vmax=error_range[1])
    axes[0, 1].set_title('Prediction Error')
    axes[0, 1].set_xlabel('Pixel Column')
    axes[0, 1].set_ylabel('Pixel Row')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

    # 3. Spectrum (with log scale for better visibility)
    rows, cols = spectrum.shape
    freq_x = np.linspace(-0.5, 0.5, cols)
    freq_y = np.linspace(-0.5, 0.5, rows)

    spectrum_min = spectrum[spectrum > 0].min() if np.any(spectrum > 0) else 1e-6

    im3 = axes[1, 0].imshow(spectrum, cmap='inferno',
                             norm=LogNorm(vmin=spectrum_min, vmax=spectrum.max()),
                             extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]],
                             origin='lower')
    axes[1, 0].set_title('Frequency Spectrum (Log Scale)')
    axes[1, 0].set_xlabel('Normalized Frequency f_x')
    axes[1, 0].set_ylabel('Normalized Frequency f_y')
    axes[1, 0].axhline(0, color='white', alpha=0.5, linewidth=0.5)
    axes[1, 0].axvline(0, color='white', alpha=0.5, linewidth=0.5)
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    # 4. Error Distribution with statistics
    error_flat = prediction_error.flatten()
    n_bins = min(50, int(np.sqrt(len(error_flat))))
    axes[1, 1].hist(error_flat, bins=n_bins, alpha=0.7, edgecolor='black', density=True)
    axes[1, 1].axvline(np.mean(error_flat), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].axvline(np.median(error_flat), color='orange', linestyle='--', linewidth=2, label='Median')
    axes[1, 1].set_title('Prediction Error Distribution')
    axes[1, 1].set_xlabel('Error Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'μ={np.mean(error_flat):.4f}\nσ={np.std(error_flat):.4f}'
    axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    output_path = output_folder / f'{filename.split(".")[0]}_analysis.png'
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


class BatchProcessor:
    def __init__(self, input_folder, output_folder, sensitivity='medium', max_workers=4, test_all_sensitivities=False):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.detector = KirchnerDetector(sensitivity)
        self.max_workers = max_workers  # Reduced from 12 to 4 for stability
        self.test_all_sensitivities = test_all_sensitivities
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
            print(f"  Processing: {img_path.name}")
            start_time = time.time()
            
            if self.test_all_sensitivities:
                results = {}
                detailed_metrics = {}
                
                for sensitivity in ['low', 'medium', 'high']:
                    print(f"    Testing {sensitivity} sensitivity...")
                    detector = KirchnerDetector(sensitivity=sensitivity)
                    result = detector.detect(str(img_path))
                    
                    metrics = detector.extract_detection_metrics(result['spectrum'])
                    results[sensitivity] = {
                        'detected': result['detected'],
                        'p_map': result['p_map'],
                        'spectrum': result['spectrum'],
                        'prediction_error': result['prediction_error'],
                        'metrics': metrics
                    }
                    detailed_metrics[sensitivity] = metrics
                
                processing_time = time.time() - start_time
                print(f"    Completed in {processing_time:.2f}s")
                
                return {
                    'file_name': img_path.name,
                    'processing_time': processing_time,
                    'multi_sensitivity_results': results,
                    'detailed_metrics': detailed_metrics,
                    'detected_low': results['low']['detected'],
                    'detected_medium': results['medium']['detected'],
                    'detected_high': results['high']['detected']
                }
            else:
                print(f"    Running detection...")
                result = self.detector.detect(str(img_path))
                processing_time = time.time() - start_time
                print(f"    Detection result: {result['detected']} in {processing_time:.2f}s")

                return {
                    'file_name': img_path.name,
                    'detected': result['detected'],
                    'processing_time': processing_time,
                    'p_map': result['p_map'],
                    'spectrum': result['spectrum'],
                    'prediction_error': result['prediction_error']
                }
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'file_name': img_path.name,
                'detected': None,
                'error': str(e)
            }

    def process_batch(self, save_visualizations=True):
        images = self.scan_images()
        print(f"Found {len(images)} images")
        
        if self.test_all_sensitivities:
            print("Testing all sensitivity levels: LOW, MEDIUM, HIGH")

        if not images:
            return pd.DataFrame()

        results = []
        start_time = time.time()

        if len(images) <= 20:
            print("Running in sequential mode for better debugging...")
            for i, img_path in enumerate(images):
                try:
                    print(f"Processing {i+1}/{len(images)}: {img_path.name}")
                    result = self.process_single(img_path)
                    results.append(result)
                    
                    if self.test_all_sensitivities and 'multi_sensitivity_results' in result:
                        low_status = 'DETECTED' if result['detected_low'] else 'NOT DETECTED'
                        med_status = 'DETECTED' if result['detected_medium'] else 'NOT DETECTED'
                        high_status = 'DETECTED' if result['detected_high'] else 'NOT DETECTED'
                        
                        print(f"    Results: LOW: {low_status}, MEDIUM: {med_status}, HIGH: {high_status}")
                    else:
                        status = 'DETECTED' if result.get('detected') else 'NOT DETECTED'
                        if 'error' in result:
                            status = 'ERROR'
                        print(f"    Result: {status}")
                        
                except Exception as e:
                    print(f"ERROR processing {img_path}: {e}")
                    results.append({
                        'file_name': img_path.name,
                        'detected': None,
                        'error': str(e)
                    })
        else:
            print(f"Running in parallel mode with {self.max_workers} workers...")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.process_single, img): img for img in images}

                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=60)  
                        results.append(result)

                        if self.test_all_sensitivities and 'multi_sensitivity_results' in result:
                            low_status = 'DETECTED' if result['detected_low'] else 'NOT DETECTED'
                            med_status = 'DETECTED' if result['detected_medium'] else 'NOT DETECTED'
                            high_status = 'DETECTED' if result['detected_high'] else 'NOT DETECTED'
                            
                            progress = ((i + 1) / len(images)) * 100
                            print(f"{progress:5.1f}% - {result['file_name']}:")
                            print(f"    LOW: {low_status}, MEDIUM: {med_status}, HIGH: {high_status}")
                        else:
                            status = 'DETECTED' if result.get('detected') else 'NOT DETECTED'
                            if 'error' in result:
                                status = 'ERROR'

                            progress = ((i + 1) / len(images)) * 100
                            print(f"{progress:5.1f}% - {result['file_name']}: {status}")
                    except Exception as e:
                        print(f"TIMEOUT/ERROR processing image {i+1}: {e}")
                        results.append({
                            'file_name': f'unknown_{i}',
                            'detected': None,
                            'error': str(e)
                        })

        df = self._create_results_dataframe(results)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_folder / f'results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)

        if save_visualizations:
            vis_folder = self.output_folder / 'visualizations'
            vis_folder.mkdir(exist_ok=True)

            for result in results:
                if 'error' not in result:
                    try:
                        if self.test_all_sensitivities and 'multi_sensitivity_results' in result:
                            self._create_multi_sensitivity_visualization(result, vis_folder)
                        elif 'p_map' in result and result['p_map'] is not None:
                            save_visualization(
                                result['file_name'],
                                result['p_map'],
                                result['spectrum'],
                                result['prediction_error'],
                                result['detected'],
                                vis_folder
                            )
                    except Exception as e:
                        print(f"Warning: Could not create visualization for {result['file_name']}: {e}")

        total_time = time.time() - start_time
        if self.test_all_sensitivities:
            detected_low = sum(1 for r in results if r.get('detected_low'))
            detected_medium = sum(1 for r in results if r.get('detected_medium'))
            detected_high = sum(1 for r in results if r.get('detected_high'))
            error_count = sum(1 for r in results if 'error' in r)

            print(f"\nMULTI-SENSITIVITY SUMMARY:")
            print(f"Total: {len(results)}")
            print(f"Detected (LOW): {detected_low}")
            print(f"Detected (MEDIUM): {detected_medium}")
            print(f"Detected (HIGH): {detected_high}")
            print(f"Errors: {error_count}")
        else:
            detected_count = sum(1 for r in results if r.get('detected'))
            error_count = sum(1 for r in results if 'error' in r)

            print(f"\nSUMMARY:")
            print(f"Total: {len(results)}")
            print(f"Detected: {detected_count}")
            print(f"Errors: {error_count}")
        
        print(f"Time: {total_time:.1f}s")
        print(f"Results: {csv_path}")

        return df

    def _create_results_dataframe(self, results):
        if not results:
            return pd.DataFrame()
            
        if self.test_all_sensitivities:
            df_data = []
            for result in results:
                if 'error' in result:
                    df_data.append({
                        'file_name': result['file_name'],
                        'detected_low': None,
                        'detected_medium': None,
                        'detected_high': None,
                        'processing_time': None,
                        'error': result['error']
                    })
                else:
                    row = {
                        'file_name': result['file_name'],
                        'detected_low': result['detected_low'],
                        'detected_medium': result['detected_medium'],
                        'detected_high': result['detected_high'],
                        'processing_time': result['processing_time']
                    }
                    if 'detailed_metrics' in result:
                        for sensitivity, metrics in result['detailed_metrics'].items():
                            if metrics:
                                prefix = f"{sensitivity}_"
                                row.update({
                                    f"{prefix}max_gradient": metrics['max_gradient'],
                                    f"{prefix}gradient_threshold": metrics['gradient_threshold'],
                                    f"{prefix}gradient_detected": metrics['gradient_detected'],
                                    f"{prefix}peak_ratio": metrics['peak_ratio'],
                                    f"{prefix}peak_ratio_threshold": metrics['peak_ratio_threshold'],
                                    f"{prefix}peak_ratio_detected": metrics['peak_ratio_detected'],
                                    f"{prefix}max_peak": metrics['max_peak'],
                                    f"{prefix}max_peak_threshold": metrics['max_peak_threshold'],
                                    f"{prefix}max_peak_detected": metrics['max_peak_detected']
                                })
                    
                    df_data.append(row)
            
            return pd.DataFrame(df_data)
        else:
            return pd.DataFrame(results)

    def _create_multi_sensitivity_visualization(self, result, vis_folder):
        filename = result['file_name']
        multi_results = result['multi_sensitivity_results']
        detailed_metrics = result['detailed_metrics']
        
        p_map = multi_results['medium']['p_map']
        spectrum = multi_results['medium']['spectrum']
        prediction_error = multi_results['medium']['prediction_error']
        
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.6], hspace=0.4, wspace=0.3)
        
        detection_summary = []
        for sens in ['low', 'medium', 'high']:
            status = "✓" if multi_results[sens]['detected'] else "✗"
            detection_summary.append(f"{sens.upper()}: {status}")
        
        fig.suptitle(f'Multi-Sensitivity Threshold Analysis: {filename}\n{" | ".join(detection_summary)}', 
                     fontsize=14, fontweight='bold')

        # P-map
        ax_pmap = fig.add_subplot(gs[0, 0])
        im1 = ax_pmap.imshow(p_map, cmap='gray', vmin=0, vmax=1)
        ax_pmap.set_title('Probability Map\n(Same for all sensitivities)')
        ax_pmap.set_xlabel('Pixel Column')
        ax_pmap.set_ylabel('Pixel Row')
        plt.colorbar(im1, ax=ax_pmap, shrink=0.8)

        # Enhanced Spectrum  
        ax_spectrum = fig.add_subplot(gs[0, 1])
        rows, cols = spectrum.shape
        freq_x = np.linspace(-0.5, 0.5, cols)
        freq_y = np.linspace(-0.5, 0.5, rows)
        
        spectrum_min = spectrum[spectrum > 0].min() if np.any(spectrum > 0) else 1e-6
        
        im2 = ax_spectrum.imshow(spectrum, cmap='inferno',
                      norm=LogNorm(vmin=spectrum_min, vmax=spectrum.max()),
                      extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]],
                      origin='lower')
        ax_spectrum.set_title('Enhanced Spectrum\n(Same for all sensitivities)')
        ax_spectrum.set_xlabel('Normalized Frequency (f_1)')
        ax_spectrum.set_ylabel('Normalized Frequency (f_2)')
        ax_spectrum.axhline(0, color='white', alpha=0.5, linewidth=0.5)
        ax_spectrum.axvline(0, color='white', alpha=0.5, linewidth=0.5)
        plt.colorbar(im2, ax=ax_spectrum, shrink=0.8)

        # Bottom row: Detailed Metrics Table
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')
        
        if all(detailed_metrics[s] is not None for s in ['low', 'medium', 'high']):
            table_data = []
            headers = ['Level', 'Max Grad', 'Grad Thresh', 'Grad', 
                      'Peak Ratio', 'Ratio Thresh', 'Ratio', 
                      'Max Peak', 'Peak Thresh', 'Peak', 'Result']
            
            for sensitivity in ['low', 'medium', 'high']:
                metrics = detailed_metrics[sensitivity]
                detected = multi_results[sensitivity]['detected']
                
                row = [
                    sensitivity.upper(),
                    f"{metrics['max_gradient']:.5f}",
                    f"{metrics['gradient_threshold']:.5f}",
                    "✓" if metrics['gradient_detected'] else "✗",
                    f"{metrics['peak_ratio']:.5f}",
                    f"{metrics['peak_ratio_threshold']:.5f}",
                    "✓" if metrics['peak_ratio_detected'] else "✗",
                    f"{metrics['max_peak']:.5f}",
                    f"{metrics['max_peak_threshold']:.5f}",
                    "✓" if metrics['max_peak_detected'] else "✗",
                    "DETECTED" if detected else "NOT DETECTED"
                ]
                table_data.append(row)
            
            table = ax_table.table(cellText=table_data, colLabels=headers,
                                  cellLoc='center', loc='center',
                                  bbox=[0, 0.0, 1, 1.0])
            table.auto_set_font_size(True)
            table.set_fontsize(8)
            table.scale(1.2, 2.5)
            
            cellDict = table.get_celld()
            for i in range(len(headers)):
                for j in range(len(table_data) + 1):  
                    cellDict[(j, i)].set_width(0.09)  
                    cellDict[(j, i)].set_height(0.15)
            
            for i, row in enumerate(table_data):
                if "DETECTED" in row[-1]:
                    table[(i+1, len(headers)-1)].set_facecolor('#ffdddd')
                else:
                    table[(i+1, len(headers)-1)].set_facecolor('#ddffdd')
                for j, cell in enumerate(row):
                    if cell == "✓":
                        table[(i+1, j)].set_facecolor('#ddffdd')
                    elif cell == "✗":
                        table[(i+1, j)].set_facecolor('#ffdddd')
            
            for j in range(len(headers)):
                table[(0, j)].set_facecolor('#e6e6e6')
                table[(0, j)].set_text_props(weight='bold')

        output_path = vis_folder / f'{filename.split(".")[0]}_multi_sensitivity_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def quick_scan(input_folder, output_folder=None, sensitivity='medium', test_all_sensitivities=False):
    if output_folder is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        suffix = "_multi_sensitivity" if test_all_sensitivities else ""
        output_folder = f"results_{timestamp}{suffix}"

    processor = BatchProcessor(input_folder, output_folder, sensitivity, test_all_sensitivities=test_all_sensitivities)
    return processor.process_batch()


def quick_scan_all_sensitivities(input_folder, output_folder=None):
    return quick_scan(input_folder, output_folder, test_all_sensitivities=True)


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


class ScalingTestSuite:
    def __init__(self, scaling_factors=None, interpolation_methods=None):
        if scaling_factors is None:
            self.scaling_factors = [
                0.5, 0.6, 0.7, 0.8, 0.9,            # Downscaling
                1.1, 1.2, 1.3, 1.4, 1.5,            # Moderate upscaling
                1.6, 1.7, 1.8, 1.9, 2.0,            # Strong upscaling
                2.5, 3.0                            # Extreme upscaling
            ]
        else:
            self.scaling_factors = scaling_factors
            
        if interpolation_methods is None:
            self.interpolation_methods = {
                'nearest': cv2.INTER_NEAREST,
                'linear': cv2.INTER_LINEAR,
                'cubic': cv2.INTER_CUBIC,
                'lanczos': cv2.INTER_LANCZOS4
            }
        else:
            self.interpolation_methods = interpolation_methods

    def create_scaled_images(self, input_folder, output_folder):
        """Create scaled versions of all images for testing."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        original_folder = output_path / 'original'
        original_folder.mkdir(exist_ok=True)
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
        images = [f for f in input_path.rglob('*') 
                 if f.is_file() and f.suffix.lower() in supported_formats]
        
        print(f"Creating scaled versions of {len(images)} images...")
        print(f"Scaling factors: {self.scaling_factors}")
        print(f"Interpolation methods: {list(self.interpolation_methods.keys())}")
        
        created_images = []
        
        for img_idx, img_path in enumerate(images):
            print(f"Processing image {img_idx + 1}/{len(images)}: {img_path.name}")
            try:
                # Load original image
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"  Could not load {img_path.name}, skipping...")
                    continue
                
                # Check original size and limit if too large
                h, w = img.shape[:2]
                max_original_size = 2048
                if max(h, w) > max_original_size:
                    scale_down = max_original_size / max(h, w)
                    new_h, new_w = int(h * scale_down), int(w * scale_down)
                    img = cv2.resize(img, (new_w, new_w), interpolation=cv2.INTER_AREA)
                    print(f"  Resized original from {h}x{w} to {new_h}x{new_w}")
                    h, w = new_h, new_w
                    
                original_name = img_path.stem
                original_ext = img_path.suffix
                
                # Copy original to test folder
                original_copy = original_folder / f"{original_name}_original{original_ext}"
                cv2.imwrite(str(original_copy), img)
                created_images.append({
                    'file_path': str(original_copy),
                    'original_name': original_name,
                    'scaling_factor': 1.0,
                    'interpolation': 'original',
                    'category': 'original'
                })
                
                for scale_factor in self.scaling_factors:
                    print(f"  Creating scale {scale_factor:.1f} versions...")
                    for interp_name, interp_method in self.interpolation_methods.items():
                        
                        scale_folder = output_path / f"scale_{scale_factor:.1f}_{interp_name}"
                        scale_folder.mkdir(exist_ok=True)
                        
                        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                        
                        if new_h < 32 or new_w < 32:
                            print(f"    Skipping {interp_name}: too small ({new_w}x{new_h})")
                            continue
                        if new_h > 4096 or new_w > 4096:
                            print(f"    Skipping {interp_name}: too large ({new_w}x{new_h})")
                            continue
                            
                        try:
                            print(f"    Scaling to {new_w}x{new_h} with {interp_name}...")
                            scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp_method)
                            
                            scaled_name = f"{original_name}_scale{scale_factor:.1f}_{interp_name}{original_ext}"
                            scaled_path = scale_folder / scaled_name
                            cv2.imwrite(str(scaled_path), scaled_img)
                            
                            created_images.append({
                                'file_path': str(scaled_path),
                                'original_name': original_name,
                                'scaling_factor': scale_factor,
                                'interpolation': interp_name,
                                'category': 'downscaled' if scale_factor < 1.0 else 'upscaled'
                            })
                            print(f"    ✓ Created {scaled_name}")
                            
                        except Exception as scale_error:
                            print(f"    ✗ Error scaling with {interp_name}: {scale_error}")
                            continue
                        
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue
        
        print(f"Created {len(created_images)} test images")
        
        config_df = pd.DataFrame(created_images)
        config_path = output_path / 'test_configuration.csv'
        config_df.to_csv(config_path, index=False)
        print(f"Test configuration saved to: {config_path}")
        
        return created_images, str(output_path)

    def run_scaling_test(self, input_folder, output_folder=None, sensitivity='medium'):
        if output_folder is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_folder = f'scaling_test_{timestamp}'
        
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Create scaled test images
        print("=== STEP 1: Creating scaled test images ===")
        scaled_images_folder = output_path / 'scaled_images'
        created_images, scaled_folder = self.create_scaled_images(input_folder, scaled_images_folder)
        
        # Step 2: Run detection on all scaled images
        print("\n=== STEP 2: Running Kirchner detection ===")
        results_folder = output_path / 'detection_results'
        processor = BatchProcessor(scaled_folder, results_folder, sensitivity=sensitivity)
        detection_results = processor.process_batch(save_visualizations=True)
        
        # Step 3: Analyze results by scaling factor
        print("\n=== STEP 3: Analyzing results by scaling factor ===")
        analysis_results = self._analyze_scaling_results(created_images, detection_results, output_path)
        
        # Step 4: Create comprehensive report
        print("\n=== STEP 4: Creating analysis report ===")
        self._create_scaling_report(analysis_results, output_path)
        
        print(f"\nScaling test completed! Results in: {output_path}")
        return analysis_results

    def _analyze_scaling_results(self, created_images, detection_results, output_path):
        config_df = pd.DataFrame(created_images)
        
        # Extract filename from file_path for matching
        config_df['file_name'] = config_df['file_path'].apply(lambda x: os.path.basename(x))
        
        # Merge with detection results
        if not detection_results.empty:
            merged_df = config_df.merge(detection_results, on='file_name', how='left')
        else:
            merged_df = config_df.copy()
            merged_df['detected'] = False
        
        # Fill missing detection results
        merged_df['detected'] = merged_df['detected'].fillna(False)
        
        # Calculate detection rates by scaling factor
        scaling_analysis = merged_df.groupby(['scaling_factor', 'interpolation']).agg({
            'detected': ['count', 'sum', 'mean'],
            'processing_time': 'mean'
        }).round(4)
        
        scaling_analysis.columns = ['total_images', 'detected_count', 'detection_rate', 'avg_processing_time']
        scaling_analysis = scaling_analysis.reset_index()
        
        # Calculate detection rates by interpolation method
        interpolation_analysis = merged_df.groupby('interpolation').agg({
            'detected': ['count', 'sum', 'mean'],
            'processing_time': 'mean'
        }).round(4)
        
        interpolation_analysis.columns = ['total_images', 'detected_count', 'detection_rate', 'avg_processing_time']
        interpolation_analysis = interpolation_analysis.reset_index()
        
        # Save detailed results
        detailed_results_path = output_path / 'detailed_scaling_results.csv'
        merged_df.to_csv(detailed_results_path, index=False)
        
        scaling_results_path = output_path / 'scaling_factor_analysis.csv'
        scaling_analysis.to_csv(scaling_results_path, index=False)
        
        interpolation_results_path = output_path / 'interpolation_method_analysis.csv'
        interpolation_analysis.to_csv(interpolation_results_path, index=False)
        
        return {
            'detailed_results': merged_df,
            'scaling_analysis': scaling_analysis,
            'interpolation_analysis': interpolation_analysis,
            'total_images': len(merged_df),
            'overall_detection_rate': merged_df['detected'].mean()
        }

    def _create_scaling_report(self, analysis_results, output_path):
        detailed_df = analysis_results['detailed_results']
        scaling_df = analysis_results['scaling_analysis']
        interpolation_df = analysis_results['interpolation_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Kirchner Detector: Scaling Factor Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Detection rate by scaling factor
        ax1 = axes[0, 0]
        for interp_method in scaling_df['interpolation'].unique():
            method_data = scaling_df[scaling_df['interpolation'] == interp_method]
            ax1.plot(method_data['scaling_factor'], method_data['detection_rate'], 
                    'o-', label=interp_method, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Scaling Factor')
        ax1.set_ylabel('Detection Rate')
        ax1.set_title('Detection Rate vs Scaling Factor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Original')
        
        # Plot 2: Detection rate by interpolation method
        ax2 = axes[0, 1]
        bars = ax2.bar(interpolation_df['interpolation'], interpolation_df['detection_rate'])
        ax2.set_ylabel('Overall Detection Rate')
        ax2.set_title('Detection Rate by Interpolation Method')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 3: Heatmap of detection rates
        ax3 = axes[1, 0]
        pivot_data = scaling_df.pivot(index='interpolation', columns='scaling_factor', values='detection_rate')
        im = ax3.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax3.set_xticks(range(len(pivot_data.columns)))
        ax3.set_xticklabels([f'{x:.1f}' for x in pivot_data.columns], rotation=45)
        ax3.set_yticks(range(len(pivot_data.index)))
        ax3.set_yticklabels(pivot_data.index)
        ax3.set_xlabel('Scaling Factor')
        ax3.set_ylabel('Interpolation Method')
        ax3.set_title('Detection Rate Heatmap')
        
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Detection Rate')
        
        # Plot 4: Processing time analysis
        ax4 = axes[1, 1]
        scaling_df_clean = scaling_df.dropna(subset=['avg_processing_time'])
        if not scaling_df_clean.empty:
            for interp_method in scaling_df_clean['interpolation'].unique():
                method_data = scaling_df_clean[scaling_df_clean['interpolation'] == interp_method]
                ax4.plot(method_data['scaling_factor'], method_data['avg_processing_time'], 
                        's-', label=interp_method, alpha=0.7)
            
            ax4.set_xlabel('Scaling Factor')
            ax4.set_ylabel('Average Processing Time (s)')
            ax4.set_title('Processing Time vs Scaling Factor')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        plot_path = output_path / 'scaling_analysis_report.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        self._create_text_report(analysis_results, output_path)
        
        print(f"Analysis report saved to: {plot_path}")

    def _create_text_report(self, analysis_results, output_path):
        detailed_df = analysis_results['detailed_results']
        scaling_df = analysis_results['scaling_analysis']
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("KIRCHNER DETECTOR: SCALING FACTOR ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total images tested: {analysis_results['total_images']}")
        report_lines.append(f"Overall detection rate: {analysis_results['overall_detection_rate']:.3f}")
        report_lines.append("")
        
        # Summary by scaling factor
        report_lines.append("DETECTION RATES BY SCALING FACTOR:")
        report_lines.append("-" * 40)
        for scale in sorted(scaling_df['scaling_factor'].unique()):
            scale_data = scaling_df[scaling_df['scaling_factor'] == scale]
            avg_rate = scale_data['detection_rate'].mean()
            report_lines.append(f"Scale {scale:4.1f}: {avg_rate:.3f} detection rate")
        
        report_lines.append("")
        
        # Summary by interpolation method
        report_lines.append("DETECTION RATES BY INTERPOLATION METHOD:")
        report_lines.append("-" * 45)
        interpolation_df = analysis_results['interpolation_analysis']
        for method in interpolation_df['interpolation']:
            rate = interpolation_df[interpolation_df['interpolation'] == method]['detection_rate'].iloc[0]
            report_lines.append(f"{method:8s}: {rate:.3f} detection rate")
        
        report_lines.append("")
        
        # Best and worst performing combinations
        report_lines.append("BEST PERFORMING COMBINATIONS:")
        report_lines.append("-" * 30)
        best_combinations = scaling_df.nlargest(5, 'detection_rate')[['scaling_factor', 'interpolation', 'detection_rate']]
        for _, row in best_combinations.iterrows():
            report_lines.append(f"Scale {row['scaling_factor']:4.1f} + {row['interpolation']:8s}: {row['detection_rate']:.3f}")
        
        report_lines.append("")
        report_lines.append("WORST PERFORMING COMBINATIONS:")
        report_lines.append("-" * 31)
        worst_combinations = scaling_df.nsmallest(5, 'detection_rate')[['scaling_factor', 'interpolation', 'detection_rate']]
        for _, row in worst_combinations.iterrows():
            report_lines.append(f"Scale {row['scaling_factor']:4.1f} + {row['interpolation']:8s}: {row['detection_rate']:.3f}")
        
        report_path = output_path / 'scaling_analysis_summary.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to: {report_path}")


def run_scaling_test(input_folder, scaling_factors=None, sensitivity='medium', output_folder=None):
    test_suite = ScalingTestSuite(scaling_factors=scaling_factors)
    return test_suite.run_scaling_test(input_folder, output_folder, sensitivity)


def test_single_image_debug():
    img_files = glob.glob('img/*.jpg') + glob.glob('img/*.png') + glob.glob('img/*.jpeg')
    if not img_files:
        print("No images found in img/ folder")
        return
    
    test_img = img_files[0]
    print(f"Testing single image: {test_img}")
    
    try:
        detector = KirchnerDetector(sensitivity='high')
        result = detector.detect(test_img)
        print(f"SUCCESS: Detection result = {result['detected']}")
        return result
    except Exception as e:
        print(f"ERROR testing single image: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_demo():
    if os.path.exists('img'):
        print("Running demo...")
        
        print("\n=== Debug Test: Single Image ===")
        single_result = test_single_image_debug()
        if single_result is None:
            print("Single image test failed, stopping demo")
            return None
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        print("\n=== Single Sensitivity Demo (HIGH) ===")
        output_folder_single = f'results_single_{timestamp}'
        try:
            results_single = quick_scan('img', output_folder_single, sensitivity='high')
            print(f"Single sensitivity demo completed! Results in: {output_folder_single}")
            if not results_single.empty:
                detected = results_single['detected'].sum()
                print(f"Detected resampling in {detected}/{len(results_single)} images")
        except Exception as e:
            print(f"Single sensitivity demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n=== Multi-Sensitivity Demo (ALL LEVELS) ===")
        output_folder_multi = f'results_multi_{timestamp}'
        try:
            results_multi = quick_scan_all_sensitivities('img', output_folder_multi)
            print(f"Multi-sensitivity demo completed! Results in: {output_folder_multi}")
            if not results_multi.empty:
                detected_low = results_multi.get('detected_low', pd.Series()).sum()
                detected_medium = results_multi.get('detected_medium', pd.Series()).sum()
                detected_high = results_multi.get('detected_high', pd.Series()).sum()
                total = len(results_multi)
                print(f"Detection Summary:")
                print(f"  LOW sensitivity: {detected_low}/{total} images")
                print(f"  MEDIUM sensitivity: {detected_medium}/{total} images")
                print(f"  HIGH sensitivity: {detected_high}/{total} images")
        except Exception as e:
            print(f"Multi-sensitivity demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n=== Scaling Test Demo ===")
        scaling_output = f'scaling_test_{timestamp}'
        try:
            demo_scaling_factors = [0.7, 0.8, 0.9, 1.2, 1.5, 2.0]
            results_scaling = run_scaling_test('img', 
                                             scaling_factors=demo_scaling_factors,
                                             sensitivity='medium',
                                             output_folder=scaling_output)
            print(f"Scaling test demo completed! Results in: {scaling_output}")
            print(f"Overall detection rate: {results_scaling['overall_detection_rate']:.3f}")
        except Exception as e:
            print(f"Scaling test demo failed: {e}")
            import traceback
            traceback.print_exc()
            
        return output_folder_multi
    else:
        print("No 'img' folder found for demo")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "scaling-test":
            if len(sys.argv) < 3:
                print("Usage: python kirchner.py scaling-test <input_folder> [output_folder]")
                print("       python kirchner.py scaling-test <input_folder> --factors 0.8,1.2,1.5")
                sys.exit(1)
                
            input_folder = sys.argv[2]
            output_folder = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
            
            scaling_factors = None
            if '--factors' in sys.argv:
                idx = sys.argv.index('--factors')
                if idx + 1 < len(sys.argv):
                    factors_str = sys.argv[idx + 1]
                    scaling_factors = [float(f.strip()) for f in factors_str.split(',')]
                    
            # Parse sensitivity
            sensitivity = 'medium'
            if '--sensitivity' in sys.argv:
                idx = sys.argv.index('--sensitivity')
                if idx + 1 < len(sys.argv):
                    sensitivity = sys.argv[idx + 1]
            
            print("Running SCALING TEST...")
            print(f"Input folder: {input_folder}")
            print(f"Scaling factors: {scaling_factors if scaling_factors else 'default range'}")
            print(f"Sensitivity: {sensitivity}")
            
            results = run_scaling_test(input_folder, scaling_factors, sensitivity, output_folder)
            print("Scaling test completed!")
            
        elif command == "batch":
            if len(sys.argv) < 3:
                print("Usage: python kirchner.py batch <input_folder> [output_folder] [options]")
                sys.exit(1)
                
            input_folder = sys.argv[2]
            output_folder = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
            test_all = "--all-sensitivities" in sys.argv
            
            if test_all:
                print("Running batch processing with ALL sensitivity levels...")
                results = quick_scan_all_sensitivities(input_folder, output_folder)
            else:
                sensitivity = 'medium'  
                if '--sensitivity' in sys.argv:
                    idx = sys.argv.index('--sensitivity')
                    if idx + 1 < len(sys.argv):
                        sensitivity = sys.argv[idx + 1]
                
                print(f"Running batch processing with {sensitivity.upper()} sensitivity...")
                results = quick_scan(input_folder, output_folder, sensitivity)
            
            print("Batch processing completed!")
            
        elif command == "single":
            # Single image mode
            if len(sys.argv) < 3:
                print("Usage: python kirchner.py single <image_path> [--sensitivity medium] [--save-plot]")
                sys.exit(1)
                
            image_path = sys.argv[2]
            sensitivity = 'medium'
            save_plot = '--save-plot' in sys.argv
            
            if '--sensitivity' in sys.argv:
                idx = sys.argv.index('--sensitivity')
                if idx + 1 < len(sys.argv):
                    sensitivity = sys.argv[idx + 1]
            
            print(f"Analyzing single image: {image_path}")
            detected = detect_single_image(image_path, sensitivity, save_plot)
            
        else:
            print("Unknown command. Available commands:")
            print("  demo                          - Run demonstration")
            print("  batch <folder>                - Batch process images")
            print("  scaling-test <folder>         - Run scaling factor test")
            print("  single <image>                - Analyze single image")
            print("")
            print("Examples:")
            print("  python kirchner.py demo")
            print("  python kirchner.py batch img/")
            print("  python kirchner.py batch img/ --all-sensitivities")
            print("  python kirchner.py scaling-test img/ --factors 0.8,1.2,1.5")
            print("  python kirchner.py single image.jpg --save-plot")
    else:
        run_demo()