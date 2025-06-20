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

class ScalingTestSuite:
    def __init__(self, scaling_factors=None, interpolation_methods=None):
        # self.scaling_factors = [
        #     0.5, 0.6, 0.7, 0.8, 0.9,            # Downscaling
        #     1.1, 1.2, 1.3, 1.4, 1.5,            # Moderate upscaling
        #     1.6, 1.7, 1.8, 1.9, 2.0,            # Strong upscaling
        #     2.5, 3.0                            # Extreme upscaling
        # ]
        self.scaling_factors = [
            0.5, 0.8, 1.2, 1.5,
        ]
            
        self.interpolation_methods = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }

    def create_scaled_images(self, input_folder, output_folder):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
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
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"  Could not load {img_path.name}, skipping...")
                    continue
                
                h, w = img.shape[:2]
                max_original_size = 2048
                if max(h, w) > max_original_size:
                    scale_down = max_original_size / max(h, w)
                    new_h, new_w = int(h * scale_down), int(w * scale_down)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    print(f"  Resized original from {h}x{w} to {new_h}x{new_w}")
                    h, w = new_h, new_w
                    
                original_name = img_path.stem
                original_ext = img_path.suffix
                
                image_folder = output_path / original_name
                image_folder.mkdir(exist_ok=True)
                
                original_copy = image_folder / f"{original_name}_original{original_ext}"
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
                            scaled_path = image_folder / scaled_name
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

    def process_with_detailed_metrics(self, img_path, detector):
        try:
            print(f"  Processing with metrics: {Path(img_path).name}")
            start_time = time.time()
            
            result = detector.detect(str(img_path))
            detailed_metrics = detector.extract_detection_metrics(result['spectrum'])
            
            if detailed_metrics and detailed_metrics['peak_count'] == 0:
                print(f"    DEBUG: No peaks detected. Spectrum max: {np.max(result['spectrum']):.6f}")
                print(f"    DEBUG: Current peak threshold: {detector.peak_threshold:.6f}")
                
                original_threshold = detector.peak_threshold
                detector.peak_threshold = detector.peak_threshold * 0.5  # Lower threshold
                debug_peaks = detector.detect_characteristic_peaks(result['spectrum'])
                detector.peak_threshold = original_threshold  # Restore original
                
                if debug_peaks:
                    print(f"    DEBUG: With lower threshold, found {len(debug_peaks)} peaks")
                    print(f"    DEBUG: Strongest peak: {debug_peaks[0]['strength']:.6f}")
                else:
                    print(f"    DEBUG: Even with lower threshold, no peaks found")
            
            processing_time = time.time() - start_time
            print(f"    Completed in {processing_time:.2f}s")
            
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
            import traceback
            traceback.print_exc()
            return {
                'file_name': Path(img_path).name,
                'detected': None,
                'error': str(e)
            }

    def run_scaling_test(self, input_folder, output_folder=None, sensitivity='medium', detector_class=None):
        """Run complete scaling test suite."""
        if output_folder is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_folder = f'scaling_test_{timestamp}'
        
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Create scaled test images
        print("=== STEP 1: Creating scaled test images ===")
        scaled_images_folder = output_path / 'scaled_images'
        created_images, scaled_folder = self.create_scaled_images(input_folder, scaled_images_folder)
        
        # Step 2: Run detection with detailed metrics
        print("\n=== STEP 2: Running Kirchner detection with detailed metrics ===")
        
        if detector_class is None:
            from kirchner import KirchnerDetector
            detector_class = KirchnerDetector
            
        detector = detector_class(sensitivity=sensitivity)
        print(f"Using detector with sensitivity: {sensitivity}")
        print(f"Peak threshold: {detector.peak_threshold}")
        print(f"Min peaks required: {detector.min_peaks}")
        
        # Find all created images
        images = []
        for file_path in Path(scaled_folder).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}:
                images.append(file_path)
        
        print(f"Found {len(images)} images to process")
        
        # Process images with detailed metrics
        results = []
        for i, img_path in enumerate(images):
            print(f"Processing {i+1}/{len(images)}: {img_path.name}")
            result = self.process_with_detailed_metrics(img_path, detector)
            results.append(result)
            
            status = 'DETECTED' if result.get('detected') else 'NOT DETECTED'
            if 'error' in result:
                status = 'ERROR'
            print(f"    Result: {status}")
        
        # Step 3: Create detailed visualizations
        print("\n=== STEP 3: Creating detailed visualizations ===")
        vis_folder = output_path / 'visualizations'
        vis_folder.mkdir(exist_ok=True)
        
        visualization_count = 0
        for result in results:
            if 'error' not in result and result['p_map'] is not None:
                try:
                    filename = result['file_name']
                    scaling_factor = 1.0
                    interpolation_method = 'original'
                    
                    # Parse scaling info from filename
                    if '_scale' in filename:
                        parts = filename.split('_scale')[1].split('_')
                        if len(parts) >= 2:
                            scaling_factor = float(parts[0])
                            interpolation_method = parts[1].split('.')[0]
                    
                    save_scaling_visualization(
                        result['file_name'],
                        result['p_map'],
                        result['spectrum'],
                        result['prediction_error'],
                        result['detected'],
                        scaling_factor,
                        interpolation_method,
                        result['detailed_metrics'],
                        vis_folder
                    )
                    visualization_count += 1
                except Exception as e:
                    print(f"Warning: Could not create visualization for {result['file_name']}: {e}")
        
        print(f"Created {visualization_count} visualizations")
        
        # Step 4: Analyze results
        print("\n=== STEP 4: Analyzing results ===")
        analysis_results = self.analyze_scaling_results(created_images, results, output_path)
        
        # Step 5: Create comprehensive report
        print("\n=== STEP 5: Creating analysis report ===")
        self.create_scaling_report(analysis_results, output_path)
        
        print(f"\nScaling test completed! Results in: {output_path}")
        print(f"Overall detection rate: {analysis_results['overall_detection_rate']:.3f}")
        return analysis_results

    def analyze_scaling_results(self, created_images, detection_results, output_path):
        config_df = pd.DataFrame(created_images)
        
        results_data = []
        for result in detection_results:
            row = {
                'file_name': result['file_name'],
                'detected': result.get('detected', False),
                'processing_time': result.get('processing_time', None)
            }
            
            if 'detailed_metrics' in result and result['detailed_metrics']:
                metrics = result['detailed_metrics']
                row.update({
                    'max_gradient': metrics.get('max_gradient', None),
                    'gradient_detected': metrics.get('gradient_detected', None),
                    'peak_ratio': metrics.get('peak_ratio', None),
                    'peak_ratio_detected': metrics.get('peak_ratio_detected', None),
                    'max_peak': metrics.get('max_peak', None),
                    'max_peak_detected': metrics.get('max_peak_detected', None),
                    'peak_count': metrics.get('peak_count', None),
                    'kirchner_peaks': metrics.get('kirchner_peaks', None)
                })
            
            results_data.append(row)
        
        detection_df = pd.DataFrame(results_data)
        
        # Merge configuration with results
        config_df['file_name'] = config_df['file_path'].apply(lambda x: os.path.basename(x))
        if not detection_df.empty:
            merged_df = config_df.merge(detection_df, on='file_name', how='left')
        else:
            merged_df = config_df.copy()
            merged_df['detected'] = False
        
        merged_df['detected'] = merged_df['detected'].fillna(False)
        
        # Create scaling factor analysis
        scaling_analysis = merged_df.groupby(['scaling_factor', 'interpolation']).agg({
            'detected': ['count', 'sum', 'mean'],
            'processing_time': 'mean'
        }).round(4)
        
        scaling_analysis.columns = ['total_images', 'detected_count', 'detection_rate', 'avg_processing_time']
        scaling_analysis = scaling_analysis.reset_index()
        
        detailed_results_path = output_path / 'scaling_results.csv'
        merged_df.to_csv(detailed_results_path, index=False)
        print(f"Detailed results saved to: {detailed_results_path}")
        
        scaling_results_path = output_path / 'scaling_factor_analysis.csv'
        scaling_analysis.to_csv(scaling_results_path, index=False)
        print(f"Scaling analysis saved to: {scaling_results_path}")
        
        return {
            'detailed_results': merged_df,
            'scaling_analysis': scaling_analysis,
            'total_images': len(merged_df),
            'overall_detection_rate': merged_df['detected'].mean() if len(merged_df) > 0 else 0.0
        }

    def create_scaling_report(self, analysis_results, output_path):
        detailed_df = analysis_results['detailed_results']
        scaling_df = analysis_results['scaling_analysis']
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Kirchner Detector: Scaling Factor Analysis with Detailed Metrics', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Detection rate by scaling factor
        ax1 = axes[0, 0]
        if len(scaling_df) > 0:
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
        else:
            ax1.text(0.5, 0.5, 'No scaling data available', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Plot 2: Detection criteria analysis
        ax2 = axes[0, 1]
        if 'max_gradient' in detailed_df.columns:
            criteria_cols = ['gradient_detected', 'peak_ratio_detected', 'max_peak_detected']
            criteria_names = ['Max Gradient', 'Peak Ratio', 'Max Peak']
            
            detection_rates = []
            for col in criteria_cols:
                if col in detailed_df.columns:
                    rate = detailed_df[col].fillna(False).mean()
                    detection_rates.append(rate)
                else:
                    detection_rates.append(0)
            
            bars = ax2.bar(criteria_names, detection_rates, color=['skyblue', 'lightcoral', 'lightgreen'])
            ax2.set_ylabel('Detection Rate')
            ax2.set_title('Detection Rate by Criteria')
            ax2.set_ylim(0, 1)
            
            for bar, rate in zip(bars, detection_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No detailed metrics available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Heatmap of detection rates
        ax3 = axes[1, 0]
        if len(scaling_df) > 0 and len(scaling_df['interpolation'].unique()) > 1:
            try:
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
            except Exception as e:
                ax3.text(0.5, 0.5, f'Heatmap creation failed:\n{str(e)}', 
                        ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for heatmap', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Peak count analysis
        ax4 = axes[1, 1]
        if 'peak_count' in detailed_df.columns:
            peak_counts = detailed_df['peak_count'].dropna()
            
            if len(peak_counts) > 0:
                zero_peaks = (peak_counts == 0).sum()
                total_images = len(peak_counts)
                
                if peak_counts.max() > 0:
                    non_zero_peaks = peak_counts[peak_counts > 0]
                    if len(non_zero_peaks) > 0:
                        ax4.hist(non_zero_peaks, bins=min(20, len(non_zero_peaks)), alpha=0.7, edgecolor='black')
                        ax4.axvline(non_zero_peaks.mean(), color='red', linestyle='--', 
                                   linewidth=2, label=f'Mean: {non_zero_peaks.mean():.1f}')
                        ax4.set_xlabel('Peak Count (Non-Zero Only)')
                        ax4.set_ylabel('Frequency')
                        ax4.set_title(f'Peak Count Distribution\n{zero_peaks}/{total_images} images had 0 peaks')
                        ax4.legend()
                        ax4.grid(True, alpha=0.3)
                    else:
                        ax4.text(0.5, 0.5, f'All {total_images} images had 0 peaks\nCheck peak detection thresholds', 
                                ha='center', va='center', transform=ax4.transAxes, 
                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                else:
                    ax4.text(0.5, 0.5, f'All {total_images} images had 0 peaks\nPeak detection may need tuning', 
                            ha='center', va='center', transform=ax4.transAxes,
                            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
            else:
                ax4.text(0.5, 0.5, 'No peak count data available', 
                        ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'No peak count data available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        # Adjust layout
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.92, wspace=0.3, hspace=0.4)
        
        # Save plot
        plot_path = output_path / 'scaling_analysis_report.png'
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white', dpi=150)
        plt.close()
        
        print(f"Analysis report saved to: {plot_path}")


def save_scaling_visualization(filename, p_map, spectrum, prediction_error, detected, 
                         scaling_factor, interpolation_method, detailed_metrics, output_folder):
    fig = plt.figure(figsize=(16, 12)) 
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.4, wspace=0.3)
    
    title_color = 'red' if detected else 'green'
    status = "DETECTED" if detected else "NOT DETECTED"
    fig.suptitle(f'{filename} - {status}\nScale: {scaling_factor:.1f}x, Method: {interpolation_method}',
                fontsize=16, fontweight='bold', color=title_color, y=0.95) 

    # 1. P-map (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(p_map, cmap='hot', vmin=0, vmax=1)
    ax1.set_title('Probability Map (P-Map)', fontsize=12)
    ax1.set_xlabel('Pixel Column')
    ax1.set_ylabel('Pixel Row')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 2. Prediction Error (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    error_range = np.percentile(prediction_error, [5, 95])
    im2 = ax2.imshow(prediction_error, cmap='RdBu_r',
                    vmin=error_range[0], vmax=error_range[1])
    ax2.set_title('Prediction Error', fontsize=12)
    ax2.set_xlabel('Pixel Column')
    ax2.set_ylabel('Pixel Row')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # 3. Spectrum (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    rows, cols = spectrum.shape
    freq_x = np.linspace(-0.5, 0.5, cols)
    freq_y = np.linspace(-0.5, 0.5, rows)
    
    spectrum_min = spectrum[spectrum > 0].min() if np.any(spectrum > 0) else 1e-6
    im3 = ax3.imshow(spectrum, cmap='inferno',
                    norm=LogNorm(vmin=spectrum_min, vmax=spectrum.max()),
                    extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]],
                    origin='lower')
    ax3.set_title('Frequency Spectrum (Log Scale)', fontsize=12)
    ax3.set_xlabel('Normalized Frequency f_x')
    ax3.set_ylabel('Normalized Frequency f_y')
    ax3.axhline(0, color='white', alpha=0.5, linewidth=0.5)
    ax3.axvline(0, color='white', alpha=0.5, linewidth=0.5)
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # 4. Error Distribution (middle-right)
    ax4 = fig.add_subplot(gs[1, 1])
    error_flat = prediction_error.flatten()
    n_bins = min(50, int(np.sqrt(len(error_flat))))
    ax4.hist(error_flat, bins=n_bins, alpha=0.7, edgecolor='black', density=True)
    ax4.axvline(np.mean(error_flat), color='red', linestyle='--', linewidth=2, label='Mean')
    ax4.axvline(np.median(error_flat), color='orange', linestyle='--', linewidth=2, label='Median')
    ax4.set_title('Prediction Error Distribution', fontsize=12)
    ax4.set_xlabel('Error Value')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    stats_text = f'mean={np.mean(error_flat):.4f}\nstd={np.std(error_flat):.4f}'
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 5. Enhanced Detailed Metrics Table (bottom, full width)
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    headers = ['Detection Criterion', 'Measured Value', 'Threshold', 'Result', 'Peak Analysis']
    
    peak_count = detailed_metrics.get('peak_count', 0)
    peak_info = f"{peak_count} peaks found"
    if peak_count == 0:
        peak_info += "\n(May need threshold tuning)"
        
    table_data = [
        ['Maximum Gradient', 
         f"{detailed_metrics.get('max_gradient', 0):.5f}", 
         f">{detailed_metrics.get('gradient_threshold', 0):.5f}",
         "✓ PASS" if detailed_metrics.get('gradient_detected', False) else "✗ FAIL",
         peak_info],
        
        ['Peak Ratio', 
         f"{detailed_metrics.get('peak_ratio', 0):.3f}", 
         f"≥{detailed_metrics.get('peak_ratio_threshold', 0):.3f}",
         "✓ PASS" if detailed_metrics.get('peak_ratio_detected', False) else "✗ FAIL",
         "✓ Found" if detailed_metrics.get('kirchner_peaks', False) else "✗ None"],
        
        ['Maximum Peak Strength', 
         f"{detailed_metrics.get('max_peak', 0):.5f}", 
         f">{detailed_metrics.get('max_peak_threshold', 0):.5f}",
         "✓ PASS" if detailed_metrics.get('max_peak_detected', False) else "✗ FAIL",
         f"Spectrum max: {detailed_metrics.get('spectrum_max', 0):.3f}"]
    ]
    
    table = ax_table.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0.05, 0.1, 0.9, 0.8])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Increased font size
    table.scale(1.0, 2.0)   # Better vertical scaling
    
    cellDict = table.get_celld()
    n_rows, n_cols = len(table_data) + 1, len(headers)
    
    for i in range(n_rows):
        for j in range(n_cols):
            cell = cellDict.get((i, j))
            if cell:
                cell.set_height(0.15)  # Consistent row height
                cell.set_linewidth(1)
                cell.set_edgecolor('gray')
                
                # Header styling
                if i == 0:
                    cell.set_facecolor('#e8e8e8')
                    cell.set_text_props(weight='bold', size=10)
                else:
                    if j == 3:  # Result column
                        text = table_data[i-1][j]
                        if "✓ PASS" in text:
                            cell.set_facecolor('#d4edda')  # Light green
                        elif "✗ FAIL" in text:
                            cell.set_facecolor('#f8d7da')  # Light red
                    elif j == 4 and "tuning" in str(table_data[i-1][j]):  # Peak analysis with warning
                        cell.set_facecolor('#fff3cd')  # Warning yellow
                    
                    cell.set_text_props(size=9)
    
    ax_table.text(0.5, 0.98, 'Detection Criteria Analysis', 
                 ha='center', va='top', transform=ax_table.transAxes,
                 fontsize=14, fontweight='bold')
    
    overall_status = "RESAMPLING DETECTED" if detected else "NO RESAMPLING DETECTED"
    status_color = 'red' if detected else 'green'
    ax_table.text(0.5, 0.02, f'Overall Result: {overall_status}', 
                 ha='center', va='bottom', transform=ax_table.transAxes,
                 fontsize=12, fontweight='bold', color=status_color,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=status_color))

    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.88, wspace=0.25, hspace=0.35)
    
    base_name = filename.split(".")[0]
    output_path = output_folder / f'{base_name}_scale{scaling_factor:.1f}_{interpolation_method}_analysis.png'
    fig.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return str(output_path)


def run_scaling_test(input_folder, scaling_factors=None, sensitivity='medium', output_folder=None, detector_class=None):
    test_suite = ScalingTestSuite(scaling_factors=scaling_factors)
    return test_suite.run_scaling_test(input_folder, output_folder, sensitivity, detector_class)