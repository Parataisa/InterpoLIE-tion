import numpy as np
import cv2
import os
import sys
import time
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
from PIL import Image
from matplotlib.colors import LogNorm

matplotlib.use('Agg')

"""
Enhanced Kirchner fast resampling detector implementation
Based on: "Fast and reliable resampling detection by spectral analysis 
of fixed linear predictor residue" (2008)

Key Mathematical Foundations:
- Resampling: s(omega chi) = sum h(omega chi - χ)s(χ) [Eq. 3]
- Prediction Error: e(omega chi) = s(omega chi) - sigma alpha s(omega chi + omegak) [Eq. 5]
- Variance Periodicity: Var[e(x)] = Var[e(x + 1)] [Theorem 1]
- P-map: p = λ exp(-|e|^tau / sigma) [Eq. 21]
"""

class KirchnerDetector:
    def __init__(self, sensitivity='medium'):
        """
        Initialize Kirchner detector with fixed linear predictor.
        
        Fixed predictor coefficients (Kirchner's optimal choice):
        Based on second-order derivative approximation for optimal 
        periodic artifact detection in resampled signals.
        """
        # Fixed predictor coefficients - approximates second-order derivative
        # Theoretical basis: Maximizes detection of linear dependencies
        # introduced by interpolation during resampling
        self.predictor = np.array([
            [-0.25,  0.50, -0.25],
            [ 0.50,  0.00,  0.50],
            [-0.25,  0.50, -0.25]
        ])

        # Kirchner's contrast enhancement parameters
        # From Eq. 21: p = λ exp(-|e|^tau / sigma)
        self.lambda_param = 1.0  # amplitude scaling
        self.tau = 2.0           # error sensitivity (≥ 1)
        self.sigma = 1.0         # variance scaling (> 0)

        # Detection thresholds based on sensitivity level
        thresholds = {
            'low':       {'gradient': 0.001, 'peak_ratio': 0.012, 'max_peak': 0.10},
            'medium':    {'gradient': 0.02,  'peak_ratio': 0.016, 'max_peak': 0.15},
            'high':      {'gradient': 0.04,  'peak_ratio': 0.020, 'max_peak': 0.20}
        }

        t = thresholds.get(sensitivity, thresholds['medium'])
        self.gradient_threshold = t['gradient']
        self.peak_ratio_threshold = t['peak_ratio']
        self.max_peak_threshold = t['max_peak']

    def detect(self, img_path):
        """
        Main detection pipeline implementing Kirchner's algorithm.
        
        Process:
        1. Load and preprocess image
        2. Apply fixed linear predictor: e(omega chi) = s(omega chi) - sum alpha s(omega chi + omegak)
        3. Generate contrast-enhanced p-map: p = lambda exp(-|e|^tau / sigma)
        4. Compute DFT and apply contrast function
        5. Detect characteristic peaks indicating resampling
        """
        # Step 1: Load and preprocess image
        img = self._load_image(img_path)

        # Step 2: Apply fixed linear predictor (Eq. 5)
        # e(omega chi) = s(omega chi) - sum  alphas(omega chi + omegak) where alpha₀ := 0
        prediction_error = cv2.filter2D(img, -1, self.predictor,
                                         borderType=cv2.BORDER_REFLECT)

        # Step 3: Generate contrast-enhanced p-map (Eq. 21)
        p_map = self._generate_kirchner_pmap(prediction_error)

        # Step 4: Compute DFT and apply contrast function
        enhanced_spectrum = self._compute_enhanced_spectrum(p_map)

        # Step 5: Detect peaks using gradient analysis
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
        Generate p-map using Kirchner's contrast function.
        
        Mathematical formula (Eq. 21):
        p = lambda * exp(- |error|^tau / sigma)
        
        Where:
        - lambda: amplitude scaling factor
        - tau: error sensitivity parameter (>= 1)
        - sigma: variance scaling parameter (> 0)
        
        Physical interpretation:
        - Large prediction errors → small p-map values (low probability)
        - Small prediction errors → large p-map values (high probability)
        - Periodic variations in error create periodic p-map patterns
        """
        abs_error = np.abs(prediction_error)
        
        # Apply Kirchner's contrast function: p = lambda * exp(- |e|^tau / sigma)
        p_map = self.lambda_param * np.exp(-(abs_error ** self.tau) / self.sigma)

        return (p_map - p_map.min()) / (p_map.max() - p_map.min() + 1e-8)

    def _compute_enhanced_spectrum(self, p_map):
        """
        Process:
        1. Center p-map (remove DC component)
        2. Apply Hanning window for spectral leakage reduction
        3. Compute 2D DFT with fftshift for centered spectrum
        4. Apply radial contrast function for peak enhancement
        
        Theoretical basis:
        - Periodic artifacts in p-map create distinct peaks in frequency domain
        - Characteristic peak positions follow: |fₒ| = 0.5 - |delta(omega) - 0.5| (Eq. 16)
        """
        rows, cols = p_map.shape

        # Remove DC component and apply windowing
        p_map_centered = p_map - np.mean(p_map)
        window = np.outer(np.hanning(rows), np.hanning(cols))
        windowed = p_map_centered * window

        # Compute DFT with centered spectrum
        spectrum = fft2(windowed)
        spectrum = fftshift(spectrum)
        magnitude_spectrum = np.abs(spectrum)
        
        # Apply contrast function for peak enhancement
        enhanced = self._apply_contrast_function(magnitude_spectrum)

        return enhanced

    def _apply_contrast_function(self, spectrum):
        """
        Enhancement strategy:
        1. Radial weighting to attenuate low frequencies
        2. Gamma correction for peak emphasis
        
        Radial filter design:
        - r_norm = r / (min(rows, cols) // 2): normalized radius
        - Filter: r_norm^2 for r ≤ 0.5, 1.0 for r > 0.5
        - Suppresses DC and low-frequency noise
        """
        rows, cols = spectrum.shape
        center_r, center_c = rows // 2, cols // 2

        # Create radial coordinate system
        y, x = np.ogrid[:rows, :cols]
        r = np.sqrt((x - center_c)**2 + (y - center_r)**2)
        r_norm = r / (min(rows, cols) // 2)

        # Apply radial weighting function
        # Attenuates low frequencies while preserving high frequencies
        radial_filter = np.where(r_norm <= 0.5, r_norm**2, 1.0)
        filtered = spectrum * radial_filter

        # Gamma correction for peak enhancement
        gamma = 0.5
        normalized = filtered / (np.max(filtered) + 1e-8)
        gamma_corrected = normalized ** gamma

        return gamma_corrected

    def _detect_peaks(self, spectrum):
        """
        Detection strategy:
        1. Focus on first quadrant (avoid DC and symmetry)
        2. Exclude center region (DC component interference)
        3. Analyze cumulative distribution of spectral values
        4. Detect sharp transitions indicating distinct peaks
        
        Mathematical basis:
        - Resampled signals exhibit periodic artifacts with period = original sampling rate
        - Aliasing creates characteristic peak patterns in frequency domain
        - Sharp gradients in cumulative periodogram indicate distinct peaks
        """
        rows, cols = spectrum.shape
        center_r, center_c = rows // 2, cols // 2

        # Exclude center region to avoid DC component interference
        exclude_radius = min(rows, cols) // 10
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_c)**2 + (y - center_r)**2)
        mask = distance >= exclude_radius

        # Focus on first quadrant for peak analysis
        first_quad = spectrum[center_r:, center_c:]
        quad_mask = mask[center_r:, center_c:]

        if not np.any(quad_mask):
            return False

        values = first_quad[quad_mask]
        if len(values) == 0:
            return False

        # Compute cumulative distribution for gradient analysis
        # Sharp transitions indicate characteristic peaks
        sorted_vals = np.sort(values)
        cumulative = np.cumsum(sorted_vals)
        cumulative = cumulative / (cumulative[-1] + 1e-8)

        # Peak detection using multiple criteria
        max_gradient = np.max(np.diff(cumulative)) if len(cumulative) > 1 else 0
        mean_val = np.mean(values)
        max_peak = np.max(values)
        threshold = mean_val + 3 * np.std(values)
        peak_ratio = np.sum(values > threshold) / len(values)

        # Apply detection thresholds
        gradient_detected = max_gradient > self.gradient_threshold
        peak_detected = (peak_ratio > self.peak_ratio_threshold and
                         max_peak > self.max_peak_threshold)

        return gradient_detected or peak_detected

def save_visualization(filename, p_map, spectrum, prediction_error, detected, output_folder):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{filename} - {"DETECTED" if detected else "NOT DETECTED"}',
                 fontsize=14, fontweight='bold',
                 color='red' if detected else 'green')

    # 1. P-map 
    im1 = axes[0, 0].imshow(p_map, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Probability Map')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    # 2. Prediction Error
    error_range = np.percentile(prediction_error, [5, 95])
    im2 = axes[0, 1].imshow(prediction_error, cmap='RdBu_r',
                            vmin=error_range[0], vmax=error_range[1])
    axes[0, 1].set_title('Prediction Error')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

    # 3. Enhanced Spectrum
    rows, cols = spectrum.shape
    freq_x = np.linspace(-0.5, 0.5, cols)
    freq_y = np.linspace(-0.5, 0.5, rows)

    spectrum_min = spectrum[spectrum > 0].min() if np.any(spectrum > 0) else 1e-6

    im3 = axes[1, 0].imshow(spectrum, cmap='inferno',
                             norm=LogNorm(vmin=spectrum_min, vmax=spectrum.max()),
                             extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]],
                             origin='lower')
    axes[1, 0].set_title('Log Spectrum')
    axes[1, 0].axhline(0, color='white', alpha=0.5)
    axes[1, 0].axvline(0, color='white', alpha=0.5)
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    # 4. Error Distribution
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
    def __init__(self, input_folder, output_folder, sensitivity='medium', max_workers=12, test_all_sensitivities=False):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.detector = KirchnerDetector(sensitivity)
        self.max_workers = max_workers
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
            start_time = time.time()
            
            if self.test_all_sensitivities:
                results = {}
                detailed_metrics = {}
                
                for sensitivity in ['low', 'medium', 'high']:
                    detector = KirchnerDetector(sensitivity=sensitivity)
                    result = detector.detect(str(img_path))
                    
                    metrics = self._extract_detection_metrics(result['spectrum'], detector)
                    results[sensitivity] = {
                        'detected': result['detected'],
                        'p_map': result['p_map'],
                        'spectrum': result['spectrum'],
                        'prediction_error': result['prediction_error'],
                        'metrics': metrics
                    }
                    detailed_metrics[sensitivity] = metrics
                
                processing_time = time.time() - start_time
                
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

    def _extract_detection_metrics(self, spectrum, detector):
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

        return {
            'max_gradient': max_gradient,
            'gradient_threshold': detector.gradient_threshold,
            'gradient_detected': max_gradient > detector.gradient_threshold,
            'peak_ratio': peak_ratio,
            'peak_ratio_threshold': detector.peak_ratio_threshold,
            'peak_ratio_detected': peak_ratio > detector.peak_ratio_threshold,
            'max_peak': max_peak,
            'max_peak_threshold': detector.max_peak_threshold,
            'max_peak_detected': max_peak > detector.max_peak_threshold
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

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_single, img): img for img in images}

            for i, future in enumerate(futures):
                result = future.result()
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
        ax_spectrum.set_xlabel('Normalized Frequency (f₁)')
        ax_spectrum.set_ylabel('Normalized Frequency (f₂)')
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
    """Single image detection with optional visualization."""
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

def run_demo():
    """Run demonstration with sample images, including multi-sensitivity testing."""
    if os.path.exists('img'):
        print("Running demo...")
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
            
        return output_folder_multi
    else:
        print("No 'img' folder found for demo")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_folder = sys.argv[2]
        output_folder = sys.argv[3] if len(sys.argv) > 3 else None
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
    else:
        run_demo()