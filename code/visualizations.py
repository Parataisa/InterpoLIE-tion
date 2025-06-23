import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from PIL import Image
import os
import warnings

warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')
matplotlib.use('Agg')  


def load_image_safely(image_path, target_size=None):
    try:
        # Method 1: Try with PIL first (more robust)
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        if target_size:
            img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
        
        return img_array
        
    except Exception as e1:
        print(f"OpenCV could not load image: {image_path}")
        return None

def find_image_file(filename, search_paths=None):
    if search_paths is None:
        search_paths = ['.', 'img', '../img', '../../img']
    
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    
    for search_path in search_paths:
        search_dir = Path(search_path)
        if search_dir.exists():
            candidate = search_dir / filename
            if candidate.exists():
                return str(candidate)
            
            for file_path in search_dir.rglob(filename):
                if file_path.is_file():
                    return str(file_path)
    
    print(f"Could not find image file: {filename}")
    return None


def create_unified_visualization(result_data, output_path, visualization_type='batch'):
    filename = result_data['file_name']
    detected = result_data['detected']
    p_map = result_data['p_map']
    spectrum = result_data['spectrum']
    prediction_error = result_data['prediction_error']
    detailed_metrics = result_data.get('detailed_metrics', {})
    
    scaling_factor = result_data.get('scaling_factor', 1.0)
    interpolation = result_data.get('interpolation', 'original')
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.4], hspace=0.3, wspace=0.3)
    
    if visualization_type == 'scaling':
        title_color = 'red' if detected else 'green'
        status = "DETECTED" if detected else "CLEAN"
        fig.suptitle(f'{filename} - {status}\nScale: {scaling_factor:.1f}x, Method: {interpolation}',
                    fontsize=16, fontweight='bold', color=title_color, y=0.95)
    else:
        title_color = 'red' if detected else 'green'
        status = "DETECTED" if detected else "CLEAN"
        fig.suptitle(f'{filename} - {status}',
                    fontsize=16, fontweight='bold', color=title_color, y=0.95)
    
    target_size = (p_map.shape[1], p_map.shape[0]) 
    
    # Panel 1: Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    
    image_loaded = False
    image_file_path = result_data.get('file_path', '')
    
    search_paths = ['.', 'img', '../img', '../../img']
    if image_file_path:
        search_paths.insert(0, str(Path(image_file_path).parent))
    
    found_image_path = find_image_file(filename, search_paths)
    
    if found_image_path:
        img_array = load_image_safely(found_image_path, target_size)
        if img_array is not None:
            ax1.imshow(img_array)
            image_loaded = True
    
    if not image_loaded:
        print(f"Warning: Could not load original image for {filename}, using prediction error as fallback")
        prediction_error_resized = cv2.resize(prediction_error.astype(np.float32), 
                                            target_size, interpolation=cv2.INTER_LINEAR)
        error_range = np.percentile(prediction_error_resized, [1, 99])
        ax1.imshow(prediction_error_resized, cmap='gray', vmin=error_range[0], vmax=error_range[1])
    
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: P-Map
    ax2 = fig.add_subplot(gs[0, 1])
    p_map_enhanced = np.clip(p_map, 0, 1)
    gamma = 0.8
    p_map_dark = np.power(p_map_enhanced, gamma)
    
    im2 = ax2.imshow(p_map_dark, cmap='binary', vmin=0, vmax=1)
    ax2.set_title('P-Map (Equation 21)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Panel 3: Frequency Spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    rows, cols = spectrum.shape
    freq_x = np.linspace(-0.5, 0.5, cols)
    freq_y = np.linspace(-0.5, 0.5, rows)
    
    spectrum_processed = spectrum.copy()
    spectrum_min = spectrum_processed[spectrum_processed > 0].min() if np.any(spectrum_processed > 0) else 1e-10
    spectrum_log = np.log10(spectrum_processed + spectrum_min)
    spectrum_log_min = spectrum_log.min()
    spectrum_log_max = spectrum_log.max()
    
    im3 = ax3.imshow(spectrum_log, cmap='gray',
                    vmin=spectrum_log_min, vmax=spectrum_log_max * 0.8,
                    extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]],
                    origin='lower')
    ax3.set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Normalized Frequency f_x')
    ax3.set_ylabel('Normalized Frequency f_y')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    max_gradient = detailed_metrics.get('max_gradient', 0)
    gradient_threshold = detailed_metrics.get('gradient_threshold', 0.008) 
    
    table_data = [
        ['Gradient Analysis', 
         f'{max_gradient:.6f}', 
         f'>{gradient_threshold:.6f}',
         'DETECTED' if detected else 'NOT DETECTED']
    ]
    
    headers = ['Method', 'Measured Value', 'Threshold', 'Result']
    
    table = ax_table.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0.15, 0.3, 0.7, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    cellDict = table.get_celld()
    n_rows, n_cols = len(table_data) + 1, len(headers)
    
    for i in range(n_rows):
        for j in range(n_cols):
            cell = cellDict.get((i, j))
            if cell:
                cell.set_linewidth(1)
                cell.set_edgecolor('gray')
                
                if i == 0:  # Header row
                    cell.set_facecolor('#e8e8e8')
                    cell.set_text_props(weight='bold', size=11)
                else:
                    if j == 3:  # Result column
                        text = table_data[i-1][j]
                        if text == 'DETECTED':
                            cell.set_facecolor('#ffcccb')  # Light red
                        elif text == 'NOT DETECTED':
                            cell.set_facecolor('#d4edda')  # Light green
                    
                    cell.set_text_props(size=10)
    
    plt.subplots_adjust(left=0.08, bottom=0.12, right=0.95, top=0.88, wspace=0.3, hspace=0.4)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig)
    
    return str(output_path)


def create_batch_visualization(result, vis_folder):
    filename = result['file_name']
    base_name = filename.split('.')[0]
    output_path = vis_folder / f'{base_name}_analysis.png'
    
    return create_unified_visualization(result, output_path, visualization_type='batch')


def create_scaling_visualization(filename, p_map, spectrum, prediction_error, detected, 
                               scaling_factor, interpolation_method, detailed_metrics, 
                               output_folder, file_path=None):
    base_name = filename.split('.')[0]
    output_path = output_folder / f'{base_name}_scale{scaling_factor:.1f}_{interpolation_method}_analysis.png'
    
    result_data = {
        'file_name': filename,
        'file_path': file_path,
        'detected': detected,
        'p_map': p_map,
        'spectrum': spectrum,
        'prediction_error': prediction_error,
        'detailed_metrics': detailed_metrics,
        'scaling_factor': scaling_factor,
        'interpolation': interpolation_method
    }
    
    return create_unified_visualization(result_data, output_path, visualization_type='scaling')


def create_comparison_visualization(results_list, output_path, title="Detection Comparison"):
    n_results = len(results_list)
    if n_results == 0:
        return
    
    fig, axes = plt.subplots(3, n_results, figsize=(6*n_results, 16))
    if n_results == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    for i, result in enumerate(results_list):
        filename = result['file_name']
        detected = result['detected']
        p_map = result['p_map']
        spectrum = result['spectrum']
        
        target_size = (p_map.shape[1], p_map.shape[0])
        
        image_file_path = result.get('file_path', '')
        found_image_path = find_image_file(filename)
        
        # Row 1: Original Images
        ax1 = axes[0, i]
        if found_image_path:
            img_array = load_image_safely(found_image_path, target_size)
            if img_array is not None:
                ax1.imshow(img_array)
            else:
                prediction_error = result['prediction_error']
                error_resized = cv2.resize(prediction_error.astype(np.float32), 
                                         target_size, interpolation=cv2.INTER_LINEAR)
                error_range = np.percentile(error_resized, [1, 99])
                ax1.imshow(error_resized, cmap='gray', vmin=error_range[0], vmax=error_range[1])
        
        status = "DETECTED" if detected else "CLEAN"
        color = 'red' if detected else 'green'
        ax1.set_title(f'{filename}\n{status}', fontsize=10, fontweight='bold', color=color)
        ax1.axis('off')
        
        # Row 2: P-Maps
        ax2 = axes[1, i]
        p_map_enhanced = np.clip(p_map, 0, 1)
        gamma = 0.8
        p_map_dark = np.power(p_map_enhanced, gamma)
        im2 = ax2.imshow(p_map_dark, cmap='binary', vmin=0, vmax=1)
        ax2.set_title('P-Map', fontsize=10, fontweight='bold')
        ax2.axis('off')
        
        # Row 3: Spectra
        ax3 = axes[2, i]
        spectrum_processed = spectrum.copy()
        spectrum_min = spectrum_processed[spectrum_processed > 0].min() if np.any(spectrum_processed > 0) else 1e-10
        spectrum_log = np.log10(spectrum_processed + spectrum_min)
        im3 = ax3.imshow(spectrum_log, cmap='gray', vmin=spectrum_log.min(), vmax=spectrum_log.max() * 0.8)
        ax3.set_title('Spectrum', fontsize=10, fontweight='bold')
        ax3.axis('off')
    
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.90, wspace=0.1, hspace=0.3)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig)
    
    return str(output_path)


# Wrapper functions for backward compatibility
def save_scaling_visualization(filename, p_map, spectrum, prediction_error, detected, 
                             scaling_factor, interpolation_method, detailed_metrics, output_folder):
    return create_scaling_visualization(filename, p_map, spectrum, prediction_error, detected,
                                      scaling_factor, interpolation_method, detailed_metrics, output_folder)


def create_single_visualization(result, vis_folder):
    return create_batch_visualization(result, vis_folder)