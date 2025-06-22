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
        self.scaling_factors = scaling_factors or [0.5, 0.8, 1.2, 1.5]
            
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
                    h, w = new_h, new_w
                    
                original_name = img_path.stem
                original_ext = img_path.suffix
                
                image_folder = output_path / original_name
                image_folder.mkdir(exist_ok=True)
                
                # Save original
                original_copy = image_folder / f"{original_name}_original{original_ext}"
                cv2.imwrite(str(original_copy), img)
                created_images.append({
                    'file_path': str(original_copy),
                    'original_name': original_name,
                    'scaling_factor': 1.0,
                    'interpolation': 'original',
                    'category': 'original'
                })
                
                # Create scaled versions
                for scale_factor in self.scaling_factors:
                    for interp_name, interp_method in self.interpolation_methods.items():
                        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                        if new_h < 32 or new_w < 32 or new_h > 4096 or new_w > 4096:
                            continue
                            
                        try:
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
                            
                        except Exception as scale_error:
                            continue
                        
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue
        
        print(f"Created {len(created_images)} test images")
        return created_images, str(output_path)

    def process_with_detailed_metrics(self, img_path, detector):
        try:
            start_time = time.time()
            
            result = detector.detect(str(img_path))
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
        
        # Step 2: Run detection
        print("\n=== STEP 2: Running Kirchner detection ===")
        
        if detector_class is None:
            from kirchner import KirchnerDetector
            detector_class = KirchnerDetector
            
        detector = detector_class(sensitivity=sensitivity)
        print(f"Using detector with sensitivity: {sensitivity}")
        
        images = []
        for file_path in Path(scaled_folder).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}:
                images.append(file_path)
        
        print(f"Found {len(images)} images to process")
        
        results = []
        for i, img_path in enumerate(images):
            print(f"Processing {i+1}/{len(images)}: {img_path.name}")
            result = self.process_with_detailed_metrics(img_path, detector)
            results.append(result)
        
        # Step 3: Create visualizations
        print("\n=== STEP 3: Creating visualizations ===")
        vis_folder = output_path / 'visualizations'
        vis_folder.mkdir(exist_ok=True)
        
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
            image_vis_folder.mkdir(exist_ok=True)
            
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
                    
                    save_scaling_visualization(
                        result['file_name'],
                        result['p_map'],
                        result['spectrum'],
                        result['prediction_error'],
                        result['detected'],
                        scaling_factor,
                        interpolation_method,
                        result['detailed_metrics'],
                        image_vis_folder
                    )
                    visualization_count += 1
                except Exception as e:
                    print(f"    Warning: Could not create visualization for {result['file_name']}: {e}")
        
        print(f"Created {visualization_count} visualizations")
        
        # Step 4: Analyze results
        print("\n=== STEP 4: Analyzing results ===")
        analysis_results = self.analyze_scaling_results(created_images, results, output_path)
        
        # Step 5: Create report
        print("\n=== STEP 5: Creating analysis report ===")
        self.create_scaling_report(analysis_results, output_path)
        
        print(f"\nScaling test completed! Results in: {output_path}")
        if 'overall_detection_rate' in analysis_results:
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
                    'gradient_threshold': metrics.get('gradient_threshold', None),
                    'peak_count': metrics.get('peak_count', None),
                    'spectrum_mean': metrics.get('spectrum_mean', None),
                    'spectrum_std': metrics.get('spectrum_std', None),
                    'spectrum_max': metrics.get('spectrum_max', None)
                })
            
            results_data.append(row)
        
        detection_df = pd.DataFrame(results_data)
        
        config_df['file_name'] = config_df['file_path'].apply(lambda x: os.path.basename(x))
        if not detection_df.empty:
            merged_df = config_df.merge(detection_df, on='file_name', how='left')
        else:
            merged_df = config_df.copy()
            merged_df['detected'] = False
        
        merged_df['detected'] = merged_df['detected'].infer_objects(copy=False).fillna(False)
        
        scaling_analysis = merged_df.groupby(['scaling_factor', 'interpolation']).agg({
            'detected': ['count', 'sum', 'mean'],
            'processing_time': 'mean',
            'max_gradient': 'mean',
        }).round(6)
        
        scaling_analysis.columns = ['total_images', 'detected_count', 'detection_rate', 
                                   'avg_processing_time', 'avg_max_gradient']
        scaling_analysis = scaling_analysis.reset_index()
        
        detailed_results_path = output_path / 'scaling_results_detailed.csv'
        merged_df.to_csv(detailed_results_path, index=False)
        
        scaling_results_path = output_path / 'scaling_factor_analysis.csv'
        scaling_analysis.to_csv(scaling_results_path, index=False)
        
        return {
            'detailed_results': merged_df,
            'scaling_analysis': scaling_analysis,
            'total_images': len(merged_df),
            'overall_detection_rate': merged_df['detected'].mean() if len(merged_df) > 0 else 0.0
        }

    def create_scaling_report(self, analysis_results, output_path):
        detailed_df = analysis_results['detailed_results']
        scaling_df = analysis_results['scaling_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Kirchner Detector: Scaling Factor Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Detection rate by scaling factor
        ax1 = axes[0, 0]
        if len(scaling_df) > 0:
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            markers = ['o', 's', '^', 'D', 'v']
            
            for i, interp_method in enumerate(scaling_df['interpolation'].unique()):
                method_data = scaling_df[scaling_df['interpolation'] == interp_method]
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                ax1.plot(method_data['scaling_factor'], method_data['detection_rate'], 
                        marker=marker, linestyle='-', label=interp_method, 
                        linewidth=2, markersize=8, color=color)
            
            ax1.set_xlabel('Scaling Factor')
            ax1.set_ylabel('Detection Rate')
            ax1.set_title('Detection Rate vs Scaling Factor')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Overall summary
        ax2 = axes[0, 1]
        if len(detailed_df) > 0:
            original_detected = detailed_df[detailed_df['category'] == 'original']['detected'].sum()
            upscaled_detected = detailed_df[detailed_df['category'] == 'upscaled']['detected'].sum()
            downscaled_detected = detailed_df[detailed_df['category'] == 'downscaled']['detected'].sum()
            
            original_total = len(detailed_df[detailed_df['category'] == 'original'])
            upscaled_total = len(detailed_df[detailed_df['category'] == 'upscaled'])
            downscaled_total = len(detailed_df[detailed_df['category'] == 'downscaled'])
            
            categories = ['Original', 'Upscaled', 'Downscaled']
            detection_rates = [
                original_detected / max(original_total, 1),
                upscaled_detected / max(upscaled_total, 1),
                downscaled_detected / max(downscaled_total, 1)
            ]
            
            bars = ax2.bar(categories, detection_rates, color=['gray', 'green', 'red'], alpha=0.7)
            ax2.set_ylabel('Detection Rate')
            ax2.set_title('Detection Rate by Image Category')
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, detection_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.2f}', ha='center', va='bottom')
        
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
                
                # Add text annotations for values
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        value = pivot_data.values[i, j]
                        # Choose text color based on background brightness
                        text_color = 'white' if value < 0.5 else 'black'
                        text = f'{value:.2f}'
                        ax3.text(j, i, text, ha="center", va="center", 
                                color=text_color, fontsize=10, fontweight='bold')
                
                cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
                cbar.set_label('Detection Rate')
            except Exception as e:
                ax3.text(0.5, 0.5, f'Heatmap unavailable', 
                        ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Processing time analysis
        ax4 = axes[1, 1]
        if 'processing_time' in detailed_df.columns:
            processing_times = detailed_df['processing_time'].dropna()
            
            if len(processing_times) > 0:
                ax4.hist(processing_times, bins=20, alpha=0.7, edgecolor='black')
                ax4.axvline(processing_times.mean(), color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: {processing_times.mean():.2f}s')
                ax4.set_xlabel('Processing Time (seconds)')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Processing Time Distribution')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_path / 'scaling_analysis_report.png'
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Analysis report saved to: {plot_path}")


def save_scaling_visualization(filename, p_map, spectrum, prediction_error, detected, 
                         scaling_factor, interpolation_method, detailed_metrics, output_folder):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.3)
    
    title_color = 'red' if detected else 'green'
    status = "DETECTED" if detected else "CLEAN"
    fig.suptitle(f'{filename} - {status}\nScale: {scaling_factor:.1f}x, Method: {interpolation_method}',
                fontsize=16, fontweight='bold', color=title_color, y=0.95) 

    # P-map
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(p_map, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('P-Map (Equation 21)', fontsize=12)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Prediction Error
    ax2 = fig.add_subplot(gs[0, 1])
    error_range = np.percentile(prediction_error, [5, 95])
    im2 = ax2.imshow(prediction_error, cmap='RdBu_r',
                    vmin=error_range[0], vmax=error_range[1])
    ax2.set_title('Prediction Error', fontsize=12)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Spectrum
    ax3 = fig.add_subplot(gs[1, 0])
    rows, cols = spectrum.shape
    freq_x = np.linspace(-0.5, 0.5, cols)
    freq_y = np.linspace(-0.5, 0.5, rows)
    
    spectrum_min = spectrum[spectrum > 0].min() if np.any(spectrum > 0) else 1e-6
    im3 = ax3.imshow(spectrum, cmap='inferno',
                    norm=LogNorm(vmin=spectrum_min, vmax=spectrum.max()),
                    extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]],
                    origin='lower')
    ax3.set_title('Frequency Spectrum', fontsize=12)
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # Error Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    error_flat = prediction_error.flatten()
    n_bins = min(50, int(np.sqrt(len(error_flat))))
    ax4.hist(error_flat, bins=n_bins, alpha=0.7, edgecolor='black', density=True)
    ax4.axvline(np.mean(error_flat), color='red', linestyle='--', linewidth=2, label='Mean')
    ax4.set_title('Error Distribution', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Detection Summary Table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # Create simplified, clearer table
    peak_count = detailed_metrics.get('peak_count', 0)
    max_gradient = detailed_metrics.get('max_gradient', 0)
    peak_detected = detailed_metrics.get('peak_method_detected', False)
    gradient_detected = detailed_metrics.get('gradient_method_detected', False)
    
    table_data = [
        ['Peak Analysis', f'{peak_count} peaks found', 'PASS' if peak_detected else 'NOT DETECTED'],
        ['Gradient Analysis', f'Max gradient: {max_gradient:.4f}', 'PASS' if gradient_detected else 'NOT DETECTED'],
        ['Final Decision', 'Peak OR Gradient method', 'DETECTED' if detected else 'NOT DETECTED']
    ]
    
    headers = ['Detection Method', 'Result', 'Status']
    
    table = ax_table.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0.1, 0.3, 0.8, 0.6])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)
    
    # Style table
    cellDict = table.get_celld()
    n_rows, n_cols = len(table_data) + 1, len(headers)
    
    for i in range(n_rows):
        for j in range(n_cols):
            cell = cellDict.get((i, j))
            if cell:
                if i == 0:  # Header
                    cell.set_facecolor('#e8e8e8')
                    cell.set_text_props(weight='bold')
                else:
                    if j == 2:  # Status column
                        text = table_data[i-1][j]
                        if text in ['PASS', 'DETECTED']:
                            cell.set_facecolor('#d4edda')
                        elif text in ['NOT DETECTED']:
                            cell.set_facecolor('#e7f3ff')

    base_name = filename.split(".")[0]
    output_path = output_folder / f'{base_name}_scale{scaling_factor:.1f}_{interpolation_method}_analysis.png'
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return str(output_path)


def run_scaling_test(input_folder, scaling_factors=None, sensitivity='medium', output_folder=None, detector_class=None):
    test_suite = ScalingTestSuite(scaling_factors=scaling_factors)
    return test_suite.run_scaling_test(input_folder, output_folder, sensitivity, detector_class)