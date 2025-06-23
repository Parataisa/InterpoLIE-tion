import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from matplotlib.colors import LogNorm
from visualizations import create_batch_visualization
matplotlib.use('Agg')

class BatchProcessor:
    def __init__(self, input_folder, output_folder, sensitivity='medium', max_workers=24):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.sensitivity = sensitivity
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
            print(f"  Processing: {img_path.name}")
            start_time = time.time()
            
            from kirchner import KirchnerDetector
            
            detector = KirchnerDetector(sensitivity=self.sensitivity)
            result = detector.detect(str(img_path))
            
            detailed_metrics = detector.extract_detection_metrics(result['spectrum'])
            detector_info = detector.get_detector_info()
            
            processing_time = time.time() - start_time
            print(f"    Result: {'DETECTED' if result['detected'] else 'CLEAN'} ({processing_time:.2f}s)")

            base_result = {
                'file_name': img_path.name,
                'file_path': str(img_path),
                'detected': result['detected'],
                'processing_time': processing_time,
                'sensitivity': self.sensitivity,
                'p_map': result['p_map'],
                'spectrum': result['spectrum'],
                'prediction_error': result['prediction_error'],
                'detailed_metrics': detailed_metrics
            }
            
            base_result.update({
                'lambda_param': detector_info['lambda_param'],
                'tau': detector_info['tau'],
                'sigma': detector_info['sigma'],
                'gradient_threshold': detector_info['gradient_threshold'],
                'max_gradient': detailed_metrics.get('max_gradient', 0),
                'spectrum_mean': detailed_metrics.get('spectrum_mean', 0),
                'spectrum_std': detailed_metrics.get('spectrum_std', 0),
                'spectrum_max': detailed_metrics.get('spectrum_max', 0),
            })
            
            return base_result
                
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            return {
                'file_name': img_path.name,
                'detected': None,
                'error': str(e)
            }

    def process_batch(self, save_visualizations=True, create_analysis_report=True):
        images = self.scan_images()
        
        print(f"Found {len(images)} images to process")

        if not images:
            return pd.DataFrame()

        results = []
        start_time = time.time()

        for i, img_path in enumerate(images):
            try:
                print(f"Processing {i+1}/{len(images)}: {img_path.name}")
                result = self.process_single(img_path)
                results.append(result)
                    
            except Exception as e:
                print(f"ERROR processing {img_path}: {e}")
                results.append({
                    'file_name': img_path.name,
                    'detected': None,
                    'error': str(e)
                })

        df = pd.DataFrame(results)
        csv_path = self.save_results_csv(df)

        if save_visualizations:
            self.create_visualizations(results)
            
        self.create_batch_analysis_report(results)

        total_time = time.time() - start_time
        self._print_summary(results, total_time, csv_path)
        return df

    def save_results_csv(self, df):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_folder / f'results_fast_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        return csv_path

    def create_visualizations(self, results):
        vis_folder = self.output_folder / 'visualizations'
        vis_folder.mkdir(exist_ok=True)

        for result in results:
            if 'error' not in result and 'p_map' in result and result['p_map'] is not None:
                try:
                    create_batch_visualization(result, vis_folder)
                except Exception as e:
                    print(f"Warning: Could not create visualization for {result['file_name']}: {e}")
        
        self.create_batch_analysis_report(results)

    def create_batch_analysis_report(self, results):
        valid_results = [r for r in results if 'error' not in r and r.get('detected') is not None]
        
        if not valid_results:
            print("No valid results for analysis report")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Kirchner Detector: Enhanced Batch Analysis Report', 
                    fontsize=18, fontweight='bold', y=0.98) 
        
        gradients = [r.get('max_gradient', 0) for r in valid_results if r.get('max_gradient') is not None]
        
        # Plot 1: Spectrum Analysis - Average Spectrum Comparison
        ax1 = axes[0, 0]
        try:
            detected_spectra = [r['spectrum'] for r in valid_results if r['detected'] and 'spectrum' in r]
            clean_spectra = [r['spectrum'] for r in valid_results if not r['detected'] and 'spectrum' in r]
            
            if detected_spectra and clean_spectra:
                avg_detected = np.mean(detected_spectra, axis=0)
                avg_clean = np.mean(clean_spectra, axis=0)
                
                rows, cols = avg_clean.shape
                freq_x = np.linspace(-0.5, 0.5, cols)
                freq_y = np.linspace(-0.5, 0.5, rows)
                spectrum_diff = np.log10(avg_detected + 1e-10) - np.log10(avg_clean + 1e-10)
                
                im1 = ax1.imshow(spectrum_diff, cmap='RdBu_r', 
                            extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]],
                            origin='lower', vmin=-2, vmax=2)
                ax1.set_title(f'Spectrum Difference: Detected - Clean\n({len(detected_spectra)} vs {len(clean_spectra)} images)', 
                            fontsize=12, fontweight='bold')
                ax1.set_xlabel('Normalized Frequency fx', fontsize=11)
                ax1.set_ylabel('Normalized Frequency fy', fontsize=11)
                plt.colorbar(im1, ax=ax1, shrink=0.8, label='Log10 Difference')
                
            elif clean_spectra:
                avg_clean = np.mean(clean_spectra, axis=0)
                rows, cols = avg_clean.shape
                freq_x = np.linspace(-0.5, 0.5, cols)
                freq_y = np.linspace(-0.5, 0.5, rows)
                avg_clean_log = np.log10(avg_clean + avg_clean[avg_clean > 0].min())
                
                im1 = ax1.imshow(avg_clean_log, cmap='gray', 
                            extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]],
                            origin='lower')
                ax1.set_title(f'Average Clean Spectrum\n({len(clean_spectra)} images)', 
                            fontsize=12, fontweight='bold')
                ax1.set_xlabel('Normalized Frequency fx', fontsize=11)
                ax1.set_ylabel('Normalized Frequency fy', fontsize=11)
                plt.colorbar(im1, ax=ax1, shrink=0.8, label='Log10 Magnitude')
            else:
                ax1.text(0.5, 0.5, 'No spectrum data available', ha='center', va='center',
                        transform=ax1.transAxes, fontsize=14, fontweight='bold')
        except Exception as e:
            ax1.text(0.5, 0.5, f'Spectrum analysis failed:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        
        # Plot 2: P-Map Statistics Analysis  
        ax2 = axes[0, 1]
        try:
            p_map_stats = []
            for r in valid_results:
                if 'p_map' in r and r['p_map'] is not None:
                    p_map = r['p_map']
                    stats = {
                        'detected': r['detected'],
                        'mean': np.mean(p_map),
                        'std': np.std(p_map),
                        'max': np.max(p_map),
                        'variance': np.var(p_map),
                        'above_05': np.sum(p_map > 0.5) / p_map.size, 
                        'above_08': np.sum(p_map > 0.8) / p_map.size  
                    }
                    p_map_stats.append(stats)
            
            if p_map_stats:
                detected_stats = [s for s in p_map_stats if s['detected']]
                clean_stats = [s for s in p_map_stats if not s['detected']]
                
                metrics = ['mean', 'std', 'max', 'variance', 'above_05', 'above_08']
                metric_labels = ['Mean', 'Std Dev', 'Maximum', 'Variance', 'Frac > 0.5', 'Frac > 0.8']
                
                x_pos = np.arange(len(metrics))
                width = 0.35
                
                if clean_stats:
                    clean_means = [np.mean([s[m] for s in clean_stats]) for m in metrics]
                    ax2.bar(x_pos - width/2, clean_means, width, label=f'Clean ({len(clean_stats)})', 
                        color='#2ca02c', alpha=0.8, edgecolor='darkgreen')
                
                if detected_stats:
                    detected_means = [np.mean([s[m] for s in detected_stats]) for m in metrics]
                    ax2.bar(x_pos + width/2, detected_means, width, label=f'Detected ({len(detected_stats)})', 
                        color='#d62728', alpha=0.8, edgecolor='darkred')
                
                ax2.set_xlabel('P-Map Metrics', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Average Value', fontsize=11, fontweight='bold')
                ax2.set_title('P-Map Statistical Comparison', fontsize=12, fontweight='bold')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(metric_labels, rotation=45, ha='right')
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3, axis='y')
            else:
                ax2.text(0.5, 0.5, 'No P-map data available', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=14, fontweight='bold')
        except Exception as e:
            ax2.text(0.5, 0.5, f'P-map analysis failed:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        # Plot 3: Individual Gradient Values vs Image Index 
        ax3 = axes[1, 0]
        if gradients and valid_results:
            image_indices = list(range(len(valid_results)))
            gradient_values = [r.get('max_gradient', 0) for r in valid_results if r.get('max_gradient') is not None]
            detection_status = [r['detected'] for r in valid_results if r.get('max_gradient') is not None]
            
            detected_indices = [i for i, detected in enumerate(detection_status) if detected]
            clean_indices = [i for i, detected in enumerate(detection_status) if not detected]
            detected_grads = [gradient_values[i] for i in detected_indices]
            clean_grads = [gradient_values[i] for i in clean_indices]
            
            if clean_indices:
                ax3.scatter(clean_indices, clean_grads, 
                        c='lightgreen', alpha=0.6, s=60, label=f'Clean Images ({len(clean_indices)})', 
                        marker='o', edgecolors='darkgreen', linewidths=1.5)
            
            if detected_indices:
                ax3.scatter(detected_indices, detected_grads, 
                        c='lightcoral', alpha=0.8, s=80, label=f'Detected Images ({len(detected_indices)})', 
                        marker='^', edgecolors='darkred', linewidths=1.5)
            
            window_size = max(3, len(gradient_values) // 8)
            rolling_gradient = []
            for i in range(len(gradient_values)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(gradient_values), i + window_size // 2 + 1)
                window_grads = gradient_values[start_idx:end_idx]
                rolling_gradient.append(sum(window_grads) / len(window_grads))
            
            ax3.plot(image_indices[:len(rolling_gradient)], rolling_gradient, color='#000080', linestyle='-', 
                    linewidth=2.5, alpha=0.9, label=f'Rolling Average (window={window_size})')
            
            if valid_results and 'gradient_threshold' in valid_results[0]:
                threshold = valid_results[0]['gradient_threshold']
                ax3.axhline(y=threshold, color='black', linestyle='--', 
                        linewidth=2.5, alpha=0.8, 
                        label=f'Threshold: {threshold:.6f}')
                
                max_grad = max(gradient_values) if gradient_values else threshold * 2
                ax3.axhspan(0, threshold, alpha=0.05, color='green')
                ax3.axhspan(threshold, max_grad * 1.1, alpha=0.05, color='red')

            ax3.set_xlabel('Image Index', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Max ∇C(f)', fontsize=12, fontweight='bold')
            ax3.set_title('Individual Image Gradients vs Processing Order', 
                        fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
            ax3.grid(True, alpha=0.4, linestyle='--')
        
        # Plot 4: Detection Confidence & Processing Time Analysis
        ax4 = axes[1, 1]
        try:
            threshold = valid_results[0].get('gradient_threshold', 0) if valid_results else 0
            margins = []
            processing_times = []
            detection_status = []
            
            for r in valid_results:
                if r.get('max_gradient') is not None:
                    margin = r['max_gradient'] - threshold
                    margins.append(margin)
                    processing_times.append(r.get('processing_time', 0))
                    detection_status.append(r['detected'])
            
            if margins and processing_times:
                detected_margins = [margins[i] for i, d in enumerate(detection_status) if d]
                clean_margins = [margins[i] for i, d in enumerate(detection_status) if not d]
                detected_times = [processing_times[i] for i, d in enumerate(detection_status) if d]
                clean_times = [processing_times[i] for i, d in enumerate(detection_status) if not d]
                
                if clean_margins:
                    ax4.scatter(clean_margins, clean_times, c='lightgreen', alpha=0.6, s=60, 
                            label=f'Clean ({len(clean_margins)})', marker='o', 
                            edgecolors='darkgreen', linewidths=1.5)
                
                if detected_margins:
                    ax4.scatter(detected_margins, detected_times, c='lightcoral', alpha=0.8, s=80, 
                            label=f'Detected ({len(detected_margins)})', marker='^', 
                            edgecolors='darkred', linewidths=1.5)
                
                ax4.axvline(x=0, color='black', linestyle='--', linewidth=2.5, alpha=0.8, 
                        label='Detection Threshold')
                
                max_margin = max(margins) if margins else 1
                min_margin = min(margins) if margins else -1
                max_time = max(processing_times) if processing_times else 1
                
                ax4.axvspan(min_margin, 0, alpha=0.05, color='green')  
                ax4.axvspan(0, max_margin, alpha=0.05, color='red')    
                
                ax4.set_xlabel('Detection Margin (Gradient - Threshold)', fontsize=11, fontweight='bold')
                ax4.set_ylabel('Processing Time (seconds)', fontsize=11, fontweight='bold')
                ax4.set_title('Detection Confidence vs Processing Time', fontsize=12, fontweight='bold')
                ax4.legend(fontsize=10)
                ax4.grid(True, alpha=0.3)
                
                if processing_times:
                    avg_time = np.mean(processing_times)
                    summary_text = f'Avg Processing: {avg_time:.3f}s\n'
                    summary_text += f'Total Images: {len(valid_results)}\n'
                    summary_text += f'Detected: {sum(detection_status)}\n'
                    summary_text += f'Threshold: {threshold:.6f}'
                    
                    ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, 
                            fontsize=10, fontweight='bold', va='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
            else:
                ax4.text(0.5, 0.5, 'Insufficient data for\nconfidence analysis', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=14, fontweight='bold')
        except Exception as e:
            ax4.text(0.5, 0.5, f'Confidence analysis failed:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94) 
        plot_path = self.output_folder / 'enhanced_batch_analysis_report.png'
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print(f"Enhanced batch analysis report saved: {plot_path}")
        
        def _print_summary(self, results, total_time, csv_path):
            detected_count = sum(1 for r in results if r.get('detected'))
            error_count = sum(1 for r in results if 'error' in r)

            print(f"\nSUMMARY:")
            print(f"Total: {len(results)}")
            print(f"Detected: {detected_count}")
            print(f"Errors: {error_count}")
            print(f"Time: {total_time:.1f}s")
            print(f"Results: {csv_path}")


def quick_scan(input_folder, output_folder=None, sensitivity='medium'):
    if output_folder is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_folder = f"results_fast_{timestamp}"

    processor = BatchProcessor(input_folder, output_folder, sensitivity)
    return processor.process_batch()