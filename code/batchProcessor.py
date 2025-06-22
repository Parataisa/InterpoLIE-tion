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
    def __init__(self, input_folder, output_folder, sensitivity='medium', max_workers=4):
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
            
            detector_info = detector.get_detector_info()
            
            processing_time = time.time() - start_time
            print(f"    Result: {'DETECTED' if result['detected'] else 'CLEAN'} ({processing_time:.2f}s)")

            base_result = {
                'file_name': img_path.name,
                'detected': result['detected'],
                'processing_time': processing_time,
                'sensitivity': self.sensitivity,
                'p_map': result['p_map'],
                'spectrum': result['spectrum'],
                'prediction_error': result['prediction_error']
            }
            
            base_result.update({
                'lambda_param': detector_info['lambda_param'],
                'tau': detector_info['tau'],
                'sigma': detector_info['sigma'],
                'gradient_threshold': detector_info['gradient_threshold'],
            })
            
            return base_result
                
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            return {
                'file_name': img_path.name,
                'detected': None,
                'error': str(e)
            }

    def process_batch(self, save_visualizations=True):
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
                    create_single_visualization(result, vis_folder)
                except Exception as e:
                    print(f"Warning: Could not create visualization for {result['file_name']}: {e}")

    def _print_summary(self, results, total_time, csv_path):
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
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.5], hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'{filename} - {"DETECTED" if detected else "CLEAN"}',
                 fontsize=14, fontweight='bold',
                 color='red' if detected else 'green', y=0.95)

    # P-map
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(p_map, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('P-Map (Equation 21)')
    ax1.set_xlabel('Pixel Column')
    ax1.set_ylabel('Pixel Row')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Prediction Error
    ax2 = fig.add_subplot(gs[0, 1])
    error_range = np.percentile(prediction_error, [5, 95])
    im2 = ax2.imshow(prediction_error, cmap='RdBu_r',
                            vmin=error_range[0], vmax=error_range[1])
    ax2.set_title('Prediction Error (Equation 5)')
    ax2.set_xlabel('Pixel Column')
    ax2.set_ylabel('Pixel Row')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Enhanced Spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    rows, cols = spectrum.shape
    freq_x = np.linspace(-0.5, 0.5, cols)
    freq_y = np.linspace(-0.5, 0.5, rows)
    spectrum_min = spectrum[spectrum > 0].min() if np.any(spectrum > 0) else 1e-6

    im3 = ax3.imshow(spectrum, cmap='inferno',
                           norm=LogNorm(vmin=spectrum_min, vmax=spectrum.max()),
                           extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]],
                           origin='lower')
    ax3.set_title('Enhanced Spectrum')
    ax3.set_xlabel('Normalized Frequency f_x')
    ax3.set_ylabel('Normalized Frequency f_y')
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # Detection Results Table
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    # Get detection metrics
    from kirchner import KirchnerDetector
    temp_detector = KirchnerDetector()
    gradient_detected, max_gradient, gradient_map = temp_detector.detect_cumulative_periodogram(spectrum)
    
    # Create clearer table
    table_data = [
        ['Gradient Analysis (Section 5.2.2)', 
         f'{max_gradient:.4f}', 
         f'>{temp_detector.gradient_threshold:.4f}',
         'PASS' if gradient_detected else 'NOT DETECTED'],
         
        ['Final Decision', 
        f'Gradient = {max_gradient:.4f}', 
        f'>{temp_detector.gradient_threshold:.4f}', 
        'DETECTED' if detected else 'NOT DETECTED']
    ]
    
    headers = ['Method', 'Measured Value', 'Threshold', 'Result']
    
    table = ax_table.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0.1, 0.2, 0.8, 0.7])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    
    # Style the table
    cellDict = table.get_celld()
    n_rows, n_cols = len(table_data) + 1, len(headers)
    
    for i in range(n_rows):
        for j in range(n_cols):
            cell = cellDict.get((i, j))
            if cell:
                cell.set_linewidth(1)
                cell.set_edgecolor('gray')
                
                if i == 0:  # Header
                    cell.set_facecolor('#e8e8e8')
                    cell.set_text_props(weight='bold', size=11)
                else:
                    # Color code results
                    if j == 3:  # Result column
                        text = table_data[i-1][j]
                        if text in ['PASS', 'DETECTED']:
                            cell.set_facecolor('#d4edda')  # Light green
                        elif text in ['NOT DETECTED']:
                            cell.set_facecolor('#e7f3ff')  # Light blue
                    
                    cell.set_text_props(size=10)

    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.88)
    
    output_path = vis_folder / f'{filename.split(".")[0]}_analysis.png'
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def quick_scan(input_folder, output_folder=None, sensitivity='medium'):
    if output_folder is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_folder = f"results_fast_{timestamp}"

    processor = BatchProcessor(input_folder, output_folder, sensitivity)
    return processor.process_batch()