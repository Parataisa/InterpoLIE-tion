import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from PIL import Image
import os
import warnings
from fileHandler import FileHandler

warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')
matplotlib.use('Agg')  
matplotlib.rcParams['figure.max_open_warning'] = 50  

def create_unified_visualization(result_data, output_path, visualization_type='batch', crop_center=False, downscale_size=512):
    filename = result_data['file_name']
    detected = result_data['detected']
    p_map = result_data['p_map']
    spectrum = result_data['spectrum']
    gradient_map = result_data.get('gradient_map')
    detailed_metrics = result_data.get('detailed_metrics', {})
    
    scaling_factor = result_data.get('scaling_factor', 1.0)
    interpolation = result_data.get('interpolation', 'original')
    rotation_angle = result_data.get('rotation_angle', 0.0)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.3], hspace=0.35, wspace=0.3)
    
    title_color = 'red' if detected else 'green'
    status = "DETECTED" if detected else "CLEAN"
    
    if visualization_type == 'scaling':
        fig.suptitle(f'{filename} - {status}\nScale: {scaling_factor:.1f}x, Method: {interpolation}',
                    fontsize=16, fontweight='bold', color=title_color, y=0.95)
    elif visualization_type == 'rotation':
        fig.suptitle(f'{filename} - {status}\nRotation: {rotation_angle:.1f}°, Method: {interpolation}',
                    fontsize=16, fontweight='bold', color=title_color, y=0.95)
    else:
        fig.suptitle(f'{filename} - {status}',
                    fontsize=16, fontweight='bold', color=title_color, y=0.95)
    
    # Panel 1: Original Image (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    file_handler = FileHandler(crop_center=crop_center, downscale_size=downscale_size)
    image_file_path = result_data.get('file_path', '')
    
    search_paths = ['.', 'img', '../img', '../../img']
    
    if image_file_path:
        file_path_obj = Path(image_file_path)
        if file_path_obj.exists():
            search_paths.insert(0, str(file_path_obj))
        search_paths.insert(0, str(file_path_obj.parent))
        
        if '_scale' in filename or '_rot' in filename:
            parent_dirs = file_path_obj.parents
            for parent in parent_dirs:
                search_paths.extend([
                    str(parent / 'scaled_images'),
                    str(parent / 'rotated_images'),
                    str(parent)
                ])
    
    found_image_path = file_handler.find_image_file(filename, search_paths)
    
    if found_image_path is None:
        try:
            if image_file_path and Path(image_file_path).exists():
                found_image_path = image_file_path
            else:
                img_array = np.ones((256, 256, 3), dtype=np.uint8) * 128
                ax1.text(0.5, 0.5, f'Image not found:\n{filename}', ha='center', va='center',
                        transform=ax1.transAxes, fontsize=12, fontweight='bold', color='red',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        except:
            img_array = np.ones((256, 256, 3), dtype=np.uint8) * 128
            ax1.text(0.5, 0.5, f'Image not found:\n{filename}', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=12, fontweight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    if found_image_path is not None:
        try:
            img_array = file_handler.load_image_rgb(found_image_path, target_size=None, apply_downscale=True)
        except Exception as e:
            img_array = np.ones((256, 256, 3), dtype=np.uint8) * 128
            ax1.text(0.5, 0.5, f'Error loading:\n{filename}\n{str(e)[:50]}...', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=10, fontweight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    ax1.imshow(img_array)
    ax1.set_aspect('equal')  
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: P-Map (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(p_map, cmap='binary_r', vmin=0, vmax=1, aspect='equal')
    ax2.set_title('P-Map (Equation 21)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Panel 3: Frequency Spectrum (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    rows, cols = spectrum.shape
    
    freq_x = np.linspace(-0.5, 0.5, cols)
    freq_y = np.linspace(-0.5, 0.5, rows)
    spectrum_processed = spectrum.copy()
    spectrum_min = spectrum_processed[spectrum_processed > 0].min() if np.any(spectrum_processed > 0) else 1e-10
    spectrum_log = np.log10(spectrum_processed + spectrum_min)
    spectrum_log_min = spectrum_log.min()
    spectrum_log_max = spectrum_log.max()
    
    plot_extent = [freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]]
    
    im3 = ax3.imshow(spectrum_log, cmap='binary_r',
                     vmin=spectrum_log_min, vmax=spectrum_log_max,
                     extent=plot_extent,
                     origin='lower',
                     aspect='equal')
    
    ax3.set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Normalized Frequency f_x')
    ax3.set_ylabel('Normalized Frequency f_y')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # Panel 4: Gradient Map (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    if gradient_map is not None:
        if gradient_map.max() > gradient_map.min():
            gradient_enhanced = np.power((gradient_map - gradient_map.min()) / 
                                       (gradient_map.max() - gradient_map.min()), 0.7)
        else:
            gradient_enhanced = gradient_map
            
        im4 = ax4.imshow(gradient_enhanced, cmap='gray', aspect='equal')
        ax4.set_title('Gradient Map |∇C(f)|', fontsize=12, fontweight='bold')
        plt.colorbar(im4, ax=ax4, shrink=0.8, label='Gradient Magnitude')
        
        max_grad_val = gradient_map.max()
        max_pos = np.unravel_index(gradient_map.argmax(), gradient_map.shape)
        ax4.plot(max_pos[1], max_pos[0], 'cyan', marker='x', markersize=12, markeredgewidth=3)
        ax4.text(0.02, 0.98, f'Max: {max_grad_val:.6f}', transform=ax4.transAxes, 
                fontsize=10, fontweight='bold', color='white', va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    else:
        ax4.text(0.5, 0.5, 'Gradient data\nnot available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14, fontweight='bold', color='gray')
        ax4.set_title('Gradient Map |∇C(f)|', fontsize=12, fontweight='bold')
        ax4.axis('off')
    
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    max_gradient = detailed_metrics.get('max_gradient', 0)
    gradient_threshold = detailed_metrics.get('gradient_threshold', 0.012) 
    
    table_data = [
        ['Gradient Analysis', 
         f'{max_gradient:.6f}', 
         f'>{gradient_threshold:.6f}',
         'DETECTED' if detected else 'NOT DETECTED']
    ]
    
    headers = ['Method', 'Measured Value', 'Threshold', 'Result']
    
    table = ax_table.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0.15, 0.2, 0.7, 0.6])
    
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
                
                if i == 0:
                    cell.set_facecolor('#e8e8e8')
                    cell.set_text_props(weight='bold', size=11)
                else:
                    if j == 3:
                        text = table_data[i-1][j]
                        if text == 'DETECTED':
                            cell.set_facecolor('#ffcccb')  # Light red
                        elif text == 'NOT DETECTED':
                            cell.set_facecolor('#d4edda')  # Light green
                    
                    cell.set_text_props(size=10)
    
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.88, wspace=0.3, hspace=0.4)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig) 
    plt.clf() 

    return str(output_path)

def create_batch_visualization(result, vis_folder, crop_center=False, downscale_size=512):
    filename = result['file_name']
    base_name = filename.split('.')[0]
    output_path = vis_folder / f'{base_name}_analysis.png'
    
    if 'gradient_map' not in result and 'detailed_metrics' in result:
        detailed_metrics = result['detailed_metrics']
        if isinstance(detailed_metrics, dict) and 'gradient_map' in detailed_metrics:
            result['gradient_map'] = detailed_metrics['gradient_map']
    
    return create_unified_visualization(result, output_path, visualization_type='batch', 
                                      crop_center=crop_center, downscale_size=downscale_size)

def create_scaling_visualization(filename, p_map, spectrum, prediction_error, detected, 
                               scaling_factor, interpolation_method, detailed_metrics, 
                               output_folder, file_path=None, crop_center=False, downscale_size=512, gradient_map=None):
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
        'interpolation': interpolation_method,
        'gradient_map': gradient_map
    }
    
    return create_unified_visualization(result_data, output_path, visualization_type='scaling', 
                                      crop_center=crop_center, downscale_size=downscale_size)

def create_rotation_visualization(filename, p_map, spectrum, prediction_error, detected, 
                                rotation_angle, interpolation_method, detailed_metrics, 
                                output_folder, file_path=None, crop_center=False, downscale_size=512, gradient_map=None):
    base_name = filename.split('.')[0]
    output_path = output_folder / f'{base_name}_rot{rotation_angle:03d}_{interpolation_method}_analysis.png'
    
    result_data = {
        'file_name': filename,
        'file_path': file_path,
        'detected': detected,
        'p_map': p_map,
        'spectrum': spectrum,
        'prediction_error': prediction_error,
        'detailed_metrics': detailed_metrics,
        'rotation_angle': rotation_angle,
        'interpolation': interpolation_method,
        'gradient_map': gradient_map
    }
    
    return create_unified_visualization(result_data, output_path, visualization_type='rotation', 
                                      crop_center=crop_center, downscale_size=downscale_size)