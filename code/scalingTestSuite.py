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
from visualizations import create_scaling_visualization
from fileHandler import FileHandler

class ScalingTestSuite:
    def __init__(self, scaling_factors=None, interpolation_methods=None, crop_center=False):
        self.scaling_factors = scaling_factors or [0.5, 0.8, 1.2, 1.6, 2.0]
        self.file_handler = FileHandler(crop_center=crop_center)
        self.crop_center = crop_center
            
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
        
        print(f"Creating scaled versions of {len(images)} images...")
        print(f"Initial downscaling: {'Enabled' if source_downscale else 'Disabled'} (target: {source_downscale_size}px)")
        
        created_images = []
        
        for img_idx, img_path in enumerate(images):
            print(f"Processing image {img_idx + 1}/{len(images)}: {img_path.name}")
            try:
                img = self.file_handler.load_image(img_path, apply_downscale=source_downscale)
                
                original_name = img_path.stem
                original_ext = img_path.suffix
                
                image_folder = output_path / original_name
                self.file_handler.create_output_folder(image_folder)
                
                original_copy = image_folder / f"{original_name}_original{original_ext}"
                cv2.imwrite(str(original_copy), img)
                created_images.append({
                    'file_path': str(original_copy),
                    'original_name': original_name,
                    'scaling_factor': 1.0,
                    'interpolation': 'original',
                    'category': 'original'
                })

                h, w = img.shape[:2]
                for scale_factor in self.scaling_factors:
                    for interp_name, interp_method in self.interpolation_methods.items():
                        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
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
        
        # Step 1: Create scaled test images with initial downscaling
        print("=== STEP 1: Creating scaled test images ===")
        scaled_images_folder = output_path / 'scaled_images'
        created_images, scaled_folder = self.create_scaled_images(input_folder, scaled_images_folder, 
                                                                  downscale_size, downscale)
        
        # Step 2: Run detection
        print("\n=== STEP 2: Running Kirchner detection ===")
        
        if detector_class is None:
            from kirchner import KirchnerDetector
            detector_class = KirchnerDetector
            
        detector = detector_class(sensitivity=sensitivity, downscale_size=downscale_size, downscale=False)
        print(f"Using detector with sensitivity: {sensitivity}")
        print(f"Detector downscaling: Disabled (images pre-processed)")
        
        images = self.file_handler.scan_folder(scaled_folder)
        
        print(f"Found {len(images)} images to process")
        
        results = []
        for i, img_path in enumerate(images):
            print(f"Processing {i+1}/{len(images)}: {img_path.name}")
            result = self.process_with_detailed_metrics(img_path, detector)
            results.append(result)
        if create_visualizations:
            # Step 3: Create visualizations
            print("\n=== STEP 3: Creating visualizations ===")
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
                        
                        # Find the actual image file path
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
                            crop_center=self.crop_center
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
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Kirchner Detector: Scaling Factor Analysis', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Plot 1: Detection Rate vs Scaling Factor 
        ax1 = axes[0, 0]
        if len(scaling_df) > 0:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            markers = ['o', 's', '^', 'D', 'v', 'P']
            linestyles = ['-', '--', '-.', ':', '-', '--']
            
            for i, interp_method in enumerate(scaling_df['interpolation'].unique()):
                method_data = scaling_df[scaling_df['interpolation'] == interp_method]
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                linestyle = linestyles[i % len(linestyles)]
                
                ax1.plot(method_data['scaling_factor'], method_data['detection_rate'], 
                        marker=marker, linestyle=linestyle, label=interp_method, 
                        linewidth=2.5, markersize=8, color=color, markeredgecolor='white',
                        markeredgewidth=1, alpha=0.9)
            
            ax1.set_xlabel('Scaling Factor', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Detection Rate', fontsize=12, fontweight='bold')
            ax1.set_title('Detection Rate vs Scaling Factor', fontsize=14, fontweight='bold')
            ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
            ax1.grid(True, alpha=0.4, linestyle='--')
            ax1.axvline(x=1.0, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
            ax1.set_ylim(-0.05, 1.05)
            
            ax1.axvspan(0.5, 1.0, alpha=0.1, color='red', label='_downscaled')
            ax1.axvspan(1.0, max(scaling_df['scaling_factor']), alpha=0.1, color='blue', label='_upscaled')
        
        # Plot 2: Detection Rate by Image Category
        ax2 = axes[0, 1]
        if len(detailed_df) > 0:
            categories_data = {}
            for category in ['original', 'upscaled', 'downscaled']:
                cat_data = detailed_df[detailed_df['category'] == category]
                if len(cat_data) > 0:
                    detected = cat_data['detected'].sum()
                    total = len(cat_data)
                    categories_data[category] = {'detected': detected, 'total': total, 'rate': detected/total}
            
            if categories_data:
                categories = list(categories_data.keys())
                detection_rates = [categories_data[cat]['rate'] for cat in categories]
                totals = [categories_data[cat]['total'] for cat in categories]
                
                colors_cat = {'original': '#808080', 'upscaled': '#2ca02c', 'downscaled': '#d62728'}
                bar_colors = [colors_cat.get(cat, '#1f77b4') for cat in categories]
                
                bars = ax2.bar(categories, detection_rates, color=bar_colors, alpha=0.8, 
                              edgecolor='white', linewidth=2)
                ax2.set_ylabel('Detection Rate', fontsize=12, fontweight='bold')
                ax2.set_title('Detection Rate by Image Category', fontsize=14, fontweight='bold')
                ax2.set_ylim(0, 1.1)
                ax2.grid(True, alpha=0.4, axis='y')
                
                for bar, rate, total in zip(bars, detection_rates, totals):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{rate:.2f}\n({total} imgs)', ha='center', va='bottom', 
                            fontsize=11, fontweight='bold')
        
        # Plot 3: Detection Rate Heatmap 
        ax3 = axes[1, 0]
        if len(scaling_df) > 0 and len(scaling_df['interpolation'].unique()) > 1:
            try:
                pivot_data = scaling_df.pivot(index='interpolation', columns='scaling_factor', values='detection_rate')
                
                from matplotlib.colors import LinearSegmentedColormap
                colors_heatmap = ['#8B0000', '#FF4500', '#FFD700', '#90EE90', '#006400']
                n_bins = 100
                cmap = LinearSegmentedColormap.from_list('custom', colors_heatmap, N=n_bins)
                
                im = ax3.imshow(pivot_data.values, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                               interpolation='nearest')
                
                ax3.set_xticks(range(len(pivot_data.columns)))
                ax3.set_xticklabels([f'{x:.1f}' for x in pivot_data.columns], rotation=45, fontsize=10)
                ax3.set_yticks(range(len(pivot_data.index)))
                ax3.set_yticklabels(pivot_data.index, fontsize=10)
                ax3.set_xlabel('Scaling Factor', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Interpolation Method', fontsize=12, fontweight='bold')
                ax3.set_title('Detection Rate Heatmap', fontsize=14, fontweight='bold')
                
                # text annotations
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        value = pivot_data.values[i, j]
                        if np.isnan(value):
                            text = 'nan'
                            text_color = 'gray'
                        else:
                            text = f'{value:.2f}'
                            text_color = 'white' if value < 0.5 else 'black'
                        
                        ax3.text(j, i, text, ha="center", va="center", 
                                color=text_color, fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.3))
                
                cbar = plt.colorbar(im, ax=ax3, shrink=0.8, aspect=20)
                cbar.set_label('Detection Rate', fontsize=11, fontweight='bold')
                cbar.ax.tick_params(labelsize=10)
                
            except Exception as e:
                ax3.text(0.5, 0.5, f'Heatmap unavailable\n({str(e)})', 
                        ha='center', va='center', transform=ax3.transAxes,
                        fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # Plot 4: Individual Image Gradients vs Scaling Factor
        ax4 = axes[1, 1]
        if len(detailed_df) > 0:
            valid_data = detailed_df.dropna(subset=['max_gradient', 'scaling_factor'])
            
            if len(valid_data) > 0:
                detected_data = valid_data[valid_data['detected'] == True]
                clean_data = valid_data[valid_data['detected'] == False]
                
                if len(clean_data) > 0:
                    ax4.scatter(clean_data['scaling_factor'], clean_data['max_gradient'], 
                              c='lightgreen', alpha=0.6, s=25, label=f'Clean Images ({len(clean_data)})', 
                              marker='o', edgecolors='darkgreen', linewidths=0.5)
                
                if len(detected_data) > 0:
                    ax4.scatter(detected_data['scaling_factor'], detected_data['max_gradient'], 
                              c='lightcoral', alpha=0.8, s=40, label=f'Detected Images ({len(detected_data)})', 
                              marker='^', edgecolors='darkred', linewidths=0.5)
                
                colors_trend = ['#000080', '#8B0000', '#006400', '#FF8C00', '#4B0082', '#8B4513']
                markers_trend = ['o', 's', '^', 'D', 'v', 'P']
                linestyles_trend = ['-', '--', '-.', ':', '-', '--']
                
                for i, interp_method in enumerate(scaling_df['interpolation'].unique()):
                    method_data = scaling_df[scaling_df['interpolation'] == interp_method]
                    color = colors_trend[i % len(colors_trend)]
                    marker = markers_trend[i % len(markers_trend)]
                    linestyle = linestyles_trend[i % len(linestyles_trend)]
                    
                    ax4.plot(method_data['scaling_factor'], method_data['avg_max_gradient'], 
                            marker=marker, linestyle=linestyle, 
                            label=f'{interp_method} (avg)', linewidth=2.5, markersize=7, 
                            color=color, alpha=0.9, markeredgecolor='white', markeredgewidth=1)

                if 'gradient_threshold' in valid_data.columns:
                    gradient_thresh = valid_data['gradient_threshold'].dropna().unique()
                    if len(gradient_thresh) > 0:
                        ax4.axhline(y=gradient_thresh[0], color='black', linestyle='--', 
                                   linewidth=2.5, alpha=0.8, 
                                   label=f'Threshold: {gradient_thresh[0]:.4f}')

                ax4.set_xlabel('Scaling Factor', fontsize=12, fontweight='bold')
                ax4.set_ylabel('Max ∇C(f)', fontsize=12, fontweight='bold')
                ax4.set_title('Individual Images & Average Gradients vs Scaling Factor', 
                             fontsize=14, fontweight='bold')
                ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, 
                          frameon=True, fancybox=True, shadow=True)
                ax4.grid(True, alpha=0.4, linestyle='--')
                ax4.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
                
                ax4.axvspan(0.5, 1.0, alpha=0.05, color='red')
                ax4.axvspan(1.0, max(valid_data['scaling_factor']), alpha=0.05, color='blue')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)  
        plot_path = output_path / 'scaling_analysis_report.png'
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()

def run_scaling_test(input_folder, scaling_factors=None, sensitivity='medium', output_folder=None, detector_class=None, create_visualizations=True, downscale_size=512, downscale=True):
    test_suite = ScalingTestSuite(scaling_factors=scaling_factors)
    return test_suite.run_scaling_test(input_folder, output_folder, sensitivity, detector_class, create_visualizations, downscale_size, downscale)