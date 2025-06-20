import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from matplotlib.colors import LogNorm
import matplotlib
matplotlib.use('Agg')


class BatchProcessor:
    def __init__(self, input_folder, output_folder, sensitivity='medium', max_workers=16, test_all_sensitivities=False):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.sensitivity = sensitivity
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
            print(f"  Processing: {img_path.name}")
            start_time = time.time()
            
            if self.test_all_sensitivities:
                return self.process_multi_sensitivity(img_path, start_time)
            else:
                return self.process_single_sensitivity(img_path, start_time)
                
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'file_name': img_path.name,
                'detected': None,
                'error': str(e)
            }

    def process_multi_sensitivity(self, img_path, start_time):
        from kirchner import KirchnerDetector
        
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

    def process_single_sensitivity(self, img_path, start_time):
        from kirchner import KirchnerDetector
        
        print(f"    Running detection...")
        detector = KirchnerDetector(self.sensitivity)
        result = detector.detect(str(img_path))
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

    def process_batch(self, save_visualizations=True):
        images = self.scan_images()
        print(f"Found {len(images)} images")
        
        if self.test_all_sensitivities:
            print("Testing all sensitivity levels: LOW, MEDIUM, HIGH")

        if not images:
            return pd.DataFrame()

        results = []
        start_time = time.time()

        # Process images sequentially for better debugging
        print("Running in sequential mode for better debugging...")
        for i, img_path in enumerate(images):
            try:
                print(f"Processing {i+1}/{len(images)}: {img_path.name}")
                result = self.process_single(img_path)
                results.append(result)
                
                self.print_progress_result(result, i+1, len(images))
                    
            except Exception as e:
                print(f"ERROR processing {img_path}: {e}")
                results.append({
                    'file_name': img_path.name,
                    'detected': None,
                    'error': str(e)
                })

        # Save results and create visualizations
        df = self.create_results_dataframe(results)
        csv_path = self.save_results_csv(df)

        if save_visualizations:
            self.create_visualizations(results)

        total_time = time.time() - start_time
        self._print_summary(results, total_time, csv_path)
        return df

    def print_progress_result(self, result, current, total):
        if self.test_all_sensitivities and 'multi_sensitivity_results' in result:
            low_status = 'DETECTED' if result['detected_low'] else 'NOT DETECTED'
            med_status = 'DETECTED' if result['detected_medium'] else 'NOT DETECTED'
            high_status = 'DETECTED' if result['detected_high'] else 'NOT DETECTED'
            
            progress = (current / total) * 100
            print(f"    Results: LOW: {low_status}, MEDIUM: {med_status}, HIGH: {high_status}")
        else:
            status = 'DETECTED' if result.get('detected') else 'NOT DETECTED'
            if 'error' in result:
                status = 'ERROR'
            print(f"    Result: {status}")

    def create_results_dataframe(self, results):
        if not results:
            return pd.DataFrame()
            
        if self.test_all_sensitivities:
            return self.create_multi_sensitivity_dataframe(results)
        else:
            return pd.DataFrame(results)

    def create_multi_sensitivity_dataframe(self, results):
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
                
                # Add detailed metrics if available
                if 'detailed_metrics' in result:
                    for sensitivity, metrics in result['detailed_metrics'].items():
                        if metrics:
                            self.print_progress_result(row, metrics, sensitivity)
                
                df_data.append(row)
        
        return pd.DataFrame(df_data)

    def print_progress_result(self, row, metrics, sensitivity):
        prefix = f"{sensitivity}_"
        metric_fields = [
            'max_gradient', 'gradient_threshold', 'gradient_detected',
            'peak_ratio', 'peak_ratio_threshold', 'peak_ratio_detected',
            'max_peak', 'max_peak_threshold', 'max_peak_detected'
        ]
        
        for field in metric_fields:
            if field in metrics:
                row[f"{prefix}{field}"] = metrics[field]

    def save_results_csv(self, df):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_folder / f'results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        return csv_path

    def create_visualizations(self, results):
        vis_folder = self.output_folder / 'visualizations'
        vis_folder.mkdir(exist_ok=True)

        for result in results:
            if 'error' not in result:
                try:
                    if self.test_all_sensitivities and 'multi_sensitivity_results' in result:
                        create_multi_sensitivity_visualization(result, vis_folder)
                    elif 'p_map' in result and result['p_map'] is not None:
                        create_single_visualization(result, vis_folder)
                except Exception as e:
                    print(f"Warning: Could not create visualization for {result['file_name']}: {e}")

    def _print_summary(self, results, total_time, csv_path):
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


def create_single_visualization(result, vis_folder):
    filename = result['file_name']
    p_map = result['p_map']
    spectrum = result['spectrum']
    prediction_error = result['prediction_error']
    detected = result['detected']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{filename} - {"DETECTED" if detected else "NOT DETECTED"}',
                 fontsize=14, fontweight='bold',
                 color='red' if detected else 'green')

    # P-map
    im1 = axes[0, 0].imshow(p_map, cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title('Probability Map (P-Map)')
    axes[0, 0].set_xlabel('Pixel Column')
    axes[0, 0].set_ylabel('Pixel Row')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    # Prediction Error
    error_range = np.percentile(prediction_error, [5, 95])
    im2 = axes[0, 1].imshow(prediction_error, cmap='RdBu_r',
                            vmin=error_range[0], vmax=error_range[1])
    axes[0, 1].set_title('Prediction Error')
    axes[0, 1].set_xlabel('Pixel Column')
    axes[0, 1].set_ylabel('Pixel Row')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

    # Spectrum
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
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    # Error Distribution
    error_flat = prediction_error.flatten()
    n_bins = min(50, int(np.sqrt(len(error_flat))))
    axes[1, 1].hist(error_flat, bins=n_bins, alpha=0.7, edgecolor='black', density=True)
    axes[1, 1].axvline(np.mean(error_flat), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].set_title('Prediction Error Distribution')
    axes[1, 1].set_xlabel('Error Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    output_path = vis_folder / f'{filename.split(".")[0]}_kirchner_analysis.png'
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def create_multi_sensitivity_visualization(result, vis_folder):
    filename = result['file_name']
    multi_results = result['multi_sensitivity_results']
    detailed_metrics = result['detailed_metrics']
    
    # Use medium sensitivity results for main visualization
    p_map = multi_results['medium']['p_map']
    spectrum = multi_results['medium']['spectrum']
    prediction_error = multi_results['medium']['prediction_error']
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.6], hspace=0.4, wspace=0.3)
    
    # Title with detection summary
    detection_summary = []
    for sens in ['low', 'medium', 'high']:
        status = "✓" if multi_results[sens]['detected'] else "✗"
        detection_summary.append(f"{sens.upper()}: {status}")
    
    fig.suptitle(f'Multi-Sensitivity Analysis: {filename}\n{" | ".join(detection_summary)}', 
                 fontsize=14, fontweight='bold')

    # P-map
    ax_pmap = fig.add_subplot(gs[0, 0])
    im1 = ax_pmap.imshow(p_map, cmap='gray', vmin=0, vmax=1)
    ax_pmap.set_title('Probability Map')
    plt.colorbar(im1, ax=ax_pmap, shrink=0.8)

    # Spectrum
    ax_spectrum = fig.add_subplot(gs[0, 1])
    rows, cols = spectrum.shape
    spectrum_min = spectrum[spectrum > 0].min() if np.any(spectrum > 0) else 1e-6
    
    im2 = ax_spectrum.imshow(spectrum, cmap='inferno',
                      norm=LogNorm(vmin=spectrum_min, vmax=spectrum.max()))
    ax_spectrum.set_title('Frequency Spectrum')
    plt.colorbar(im2, ax=ax_spectrum, shrink=0.8)

    # Prediction Error
    ax_error = fig.add_subplot(gs[0, 2])
    error_range = np.percentile(prediction_error, [5, 95])
    im3 = ax_error.imshow(prediction_error, cmap='RdBu_r',
                          vmin=error_range[0], vmax=error_range[1])
    ax_error.set_title('Prediction Error')
    plt.colorbar(im3, ax=ax_error, shrink=0.8)

    # Metrics comparison charts
    if all(detailed_metrics[s] is not None for s in ['low', 'medium', 'high']):
        create_metrics_comparison_charts(fig, gs, detailed_metrics, multi_results)

    # Bottom: Detailed metrics table
    create_metrics_table(fig, gs, detailed_metrics, multi_results)

    output_path = vis_folder / f'{filename.split(".")[0]}_multi_sensitivity_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_metrics_comparison_charts(fig, gs, detailed_metrics, multi_results):
    sensitivities = ['low', 'medium', 'high']
    x_pos = np.arange(len(sensitivities))
    
    ax_grad = fig.add_subplot(gs[1, 0])
    gradients = [detailed_metrics[s]['max_gradient'] for s in sensitivities]
    thresholds = [detailed_metrics[s]['gradient_threshold'] for s in sensitivities]
    
    bars = ax_grad.bar(x_pos, gradients, alpha=0.7, label='Max Gradient')
    ax_grad.plot(x_pos, thresholds, 'r--', marker='o', label='Threshold')
    ax_grad.set_xlabel('Sensitivity Level')
    ax_grad.set_ylabel('Gradient Value')
    ax_grad.set_title('Maximum Gradient Comparison')
    ax_grad.set_xticks(x_pos)
    ax_grad.set_xticklabels(sensitivities)
    ax_grad.legend()
    ax_grad.grid(True, alpha=0.3)

    # Peak analysis
    ax_peak = fig.add_subplot(gs[1, 1])
    peaks = [detailed_metrics[s]['max_peak'] for s in sensitivities]
    peak_thresholds = [detailed_metrics[s]['max_peak_threshold'] for s in sensitivities]
    
    bars = ax_peak.bar(x_pos, peaks, alpha=0.7, label='Max Peak')
    ax_peak.plot(x_pos, peak_thresholds, 'r--', marker='o', label='Threshold')
    ax_peak.set_xlabel('Sensitivity Level')
    ax_peak.set_ylabel('Peak Strength')
    ax_peak.set_title('Peak Strength Comparison')
    ax_peak.set_xticks(x_pos)
    ax_peak.set_xticklabels(sensitivities)
    ax_peak.legend()
    ax_peak.grid(True, alpha=0.3)

    ax_summary = fig.add_subplot(gs[1, 2])
    detection_methods = ['Gradient', 'Peak Ratio', 'Max Peak', 'Final']
    
    for i, sens in enumerate(sensitivities):
        metrics = detailed_metrics[sens]
        detections = [
            metrics['gradient_detected'],
            metrics['peak_ratio_detected'],
            metrics['max_peak_detected'],
            multi_results[sens]['detected']
        ]
        
        y_offset = i * 0.25
        colors = ['green' if d else 'red' for d in detections]
        bars = ax_summary.barh([y + y_offset for y in range(len(detection_methods))], 
                             [1]*len(detection_methods), 
                             height=0.2, color=colors, alpha=0.7,
                             label=sens.capitalize() if i == 0 else "")
    
    ax_summary.set_yticks(range(len(detection_methods)))
    ax_summary.set_yticklabels(detection_methods)
    ax_summary.set_xlabel('Detection Result')
    ax_summary.set_title('Detection Methods Summary')
    ax_summary.set_xlim(0, 1.2)


def create_metrics_table(fig, gs, detailed_metrics, multi_results):
    """Create detailed metrics table at bottom of visualization."""
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    if all(detailed_metrics[s] is not None for s in ['low', 'medium', 'high']):
        table_data = []
        headers = ['Level', 'Max Grad', 'Grad Thresh', 'Grad Det', 
                  'Peak Ratio', 'Ratio Thresh', 'Ratio Det', 
                  'Max Peak', 'Peak Thresh', 'Peak Det', 'Final Result']
        
        for sensitivity in ['low', 'medium', 'high']:
            metrics = detailed_metrics[sensitivity]
            detected = multi_results[sensitivity]['detected']
            
            row = [
                sensitivity.upper(),
                f"{metrics['max_gradient']:.5f}",
                f"{metrics['gradient_threshold']:.5f}",
                "✓" if metrics['gradient_detected'] else "✗",
                f"{metrics['peak_ratio']:.3f}",
                f"{metrics['peak_ratio_threshold']:.3f}",
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
        table.set_fontsize(7)
        table.scale(1.2, 2.0)
        
        for i, row in enumerate(table_data):
            if "DETECTED" in row[-1]:
                table[(i+1, len(headers)-1)].set_facecolor('#ffdddd')
            else:
                table[(i+1, len(headers)-1)].set_facecolor('#ddffdd')


def quick_scan(input_folder, output_folder=None, sensitivity='medium', test_all_sensitivities=False):
    if output_folder is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        suffix = "_multi_sensitivity" if test_all_sensitivities else ""
        output_folder = f"results_{timestamp}{suffix}"

    processor = BatchProcessor(input_folder, output_folder, sensitivity, test_all_sensitivities=test_all_sensitivities)
    return processor.process_batch()


def quick_scan_all_sensitivities(input_folder, output_folder=None):
    return quick_scan(input_folder, output_folder, test_all_sensitivities=True)