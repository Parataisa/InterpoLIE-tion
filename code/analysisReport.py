import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('Agg')

class AnalysisReports:
    @staticmethod
    def analyze_scaling_results(created_images, detection_results, output_path):
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

    @staticmethod
    def analyze_rotation_results(created_images, detection_results, output_path):
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
        
        rotation_analysis = merged_df.groupby(['rotation_angle', 'interpolation']).agg({
            'detected': ['count', 'sum', 'mean'],
            'processing_time': 'mean',
            'max_gradient': 'mean',
        }).round(6)
        
        rotation_analysis.columns = ['total_images', 'detected_count', 'detection_rate', 
                                    'avg_processing_time', 'avg_max_gradient']
        rotation_analysis = rotation_analysis.reset_index()
        
        detailed_results_path = output_path / 'rotation_results_detailed.csv'
        merged_df.to_csv(detailed_results_path, index=False)
        
        rotation_results_path = output_path / 'rotation_angle_analysis.csv'
        rotation_analysis.to_csv(rotation_results_path, index=False)
        
        return {
            'detailed_results': merged_df,
            'rotation_analysis': rotation_analysis,
            'total_images': len(merged_df),
            'overall_detection_rate': merged_df['detected'].mean() if len(merged_df) > 0 else 0.0
        }

    @staticmethod
    def create_scaling_report(analysis_results, output_path):
        detailed_df = analysis_results['detailed_results']
        scaling_df = analysis_results['scaling_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Kirchner Detector: Scaling Factor Analysis', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        AnalysisReports._plot_detection_vs_scaling(axes[0, 0], scaling_df)
        AnalysisReports._plot_category_analysis(axes[0, 1], detailed_df)
        AnalysisReports._plot_detection_heatmap(axes[1, 0], scaling_df)
        AnalysisReports._plot_gradient_vs_scaling(axes[1, 1], detailed_df, scaling_df)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)  
        plot_path = output_path / 'scaling_analysis_report.png'
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()

    @staticmethod
    def create_batch_analysis_report(results, output_folder):
        valid_results = [r for r in results if 'error' not in r and r.get('detected') is not None]
        
        if not valid_results:
            print("No valid results for analysis report")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Kirchner Detector: Batch Analysis Report', 
                    fontsize=18, fontweight='bold', y=0.98) 
        
        AnalysisReports._plot_spectrum_analysis(axes[0, 0], valid_results)
        AnalysisReports._plot_pmap_statistics(axes[0, 1], valid_results)
        AnalysisReports._plot_gradient_progression(axes[1, 0], valid_results)
        AnalysisReports._plot_detection_confidence(axes[1, 1], valid_results)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94) 
        plot_path = output_folder / 'batch_analysis_report.png'
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()

    @staticmethod
    def _plot_detection_vs_scaling(ax, scaling_df):
        if len(scaling_df) > 0:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            markers = ['o', 's', '^', 'D', 'v', 'P']
            linestyles = ['-', '--', '-.', ':', '-', '--']
            
            for i, interp_method in enumerate(scaling_df['interpolation'].unique()):
                method_data = scaling_df[scaling_df['interpolation'] == interp_method]
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                linestyle = linestyles[i % len(linestyles)]
                
                ax.plot(method_data['scaling_factor'], method_data['detection_rate'], 
                        marker=marker, linestyle=linestyle, label=interp_method, 
                        linewidth=2.5, markersize=8, color=color, markeredgecolor='white',
                        markeredgewidth=1, alpha=0.9)
            
            ax.set_xlabel('Scaling Factor', fontsize=12, fontweight='bold')
            ax.set_ylabel('Detection Rate', fontsize=12, fontweight='bold')
            ax.set_title('Detection Rate vs Scaling Factor', fontsize=14, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.set_ylim(-0.05, 1.05)
            
            ax.axvspan(min(scaling_df['scaling_factor']), 1.0, alpha=0.1, color='red', label='_downscaled')
            ax.axvspan(1.0, max(scaling_df['scaling_factor']), alpha=0.1, color='blue', label='_upscaled')

    @staticmethod
    def _plot_category_analysis(ax, detailed_df):
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
                
                bars = ax.bar(categories, detection_rates, color=bar_colors, alpha=0.8, 
                              edgecolor='white', linewidth=2)
                ax.set_ylabel('Detection Rate', fontsize=12, fontweight='bold')
                ax.set_title('Detection Rate by Image Category', fontsize=14, fontweight='bold')
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.4, axis='y')
                
                for bar, rate, total in zip(bars, detection_rates, totals):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{rate:.2f}\n({total} imgs)', ha='center', va='bottom', 
                            fontsize=11, fontweight='bold')

    @staticmethod
    def _plot_detection_heatmap(ax, scaling_df):
        if len(scaling_df) > 0 and len(scaling_df['interpolation'].unique()) > 1:
            try:
                pivot_data = scaling_df.pivot(index='interpolation', columns='scaling_factor', values='detection_rate')
                
                colors_heatmap = ['#8B0000', '#FF4500', '#FFD700', '#90EE90', '#006400']
                n_bins = 100
                cmap = LinearSegmentedColormap.from_list('custom', colors_heatmap, N=n_bins)
                
                im = ax.imshow(pivot_data.values, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                               interpolation='nearest')
                
                ax.set_xticks(range(len(pivot_data.columns)))
                ax.set_xticklabels([f'{x:.1f}' for x in pivot_data.columns], rotation=45, fontsize=10)
                ax.set_yticks(range(len(pivot_data.index)))
                ax.set_yticklabels(pivot_data.index, fontsize=10)
                ax.set_xlabel('Scaling Factor', fontsize=12, fontweight='bold')
                ax.set_ylabel('Interpolation Method', fontsize=12, fontweight='bold')
                ax.set_title('Detection Rate Heatmap', fontsize=14, fontweight='bold')
                
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        value = pivot_data.values[i, j]
                        if np.isnan(value):
                            continue
                        else:
                            text = f'{value:.2f}'
                            text_color = 'black'
                        
                        ax.text(j, i, text, ha="center", va="center", 
                                color=text_color, fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.3))
                
                cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                cbar.set_label('Detection Rate', fontsize=11, fontweight='bold')
                cbar.ax.tick_params(labelsize=10)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Heatmap unavailable\n({str(e)})', 
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

    @staticmethod
    def _plot_gradient_vs_scaling(ax, detailed_df, scaling_df):
        if len(detailed_df) > 0:
            valid_data = detailed_df.dropna(subset=['max_gradient', 'scaling_factor'])
            
            if len(valid_data) > 0:
                detected_data = valid_data[valid_data['detected'] == True]
                clean_data = valid_data[valid_data['detected'] == False]
                
                if len(clean_data) > 0:
                    ax.scatter(clean_data['scaling_factor'], clean_data['max_gradient'], 
                              c='lightgreen', alpha=0.6, s=25, label=f'Clean Images ({len(clean_data)})', 
                              marker='o', edgecolors='darkgreen', linewidths=0.5)
                
                if len(detected_data) > 0:
                    ax.scatter(detected_data['scaling_factor'], detected_data['max_gradient'], 
                              c='lightcoral', alpha=0.8, s=40, label=f'Detected Images ({len(detected_data)})', 
                              marker='^', edgecolors='darkred', linewidths=0.5)
                
                # Plot trend lines
                colors_trend = ['#000080', '#8B0000', '#006400', '#FF8C00', '#4B0082', '#8B4513']
                markers_trend = ['o', 's', '^', 'D', 'v', 'P']
                linestyles_trend = ['-', '--', '-.', ':', '-', '--']
                
                for i, interp_method in enumerate(scaling_df['interpolation'].unique()):
                    method_data = scaling_df[scaling_df['interpolation'] == interp_method]
                    color = colors_trend[i % len(colors_trend)]
                    marker = markers_trend[i % len(markers_trend)]
                    linestyle = linestyles_trend[i % len(linestyles_trend)]
                    
                    ax.plot(method_data['scaling_factor'], method_data['avg_max_gradient'], 
                            marker=marker, linestyle=linestyle, 
                            label=f'{interp_method} (avg)', linewidth=2.5, markersize=7, 
                            color=color, alpha=0.9, markeredgecolor='white', markeredgewidth=1)

                # Add threshold line
                if 'gradient_threshold' in valid_data.columns:
                    gradient_thresh = valid_data['gradient_threshold'].dropna().unique()
                    if len(gradient_thresh) > 0:
                        ax.axhline(y=gradient_thresh[0], color='black', linestyle='--', 
                                   linewidth=2.5, alpha=0.8, 
                                   label=f'Threshold: {gradient_thresh[0]:.4f}')

                ax.set_xlabel('Scaling Factor', fontsize=12, fontweight='bold')
                ax.set_ylabel('Max ∇C(f)', fontsize=12, fontweight='bold')
                ax.set_title('Individual Images & Average Gradients vs Scaling Factor', 
                             fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, 
                          frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.4, linestyle='--')
                ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

    @staticmethod
    def _plot_spectrum_analysis(ax, valid_results):
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
                
                im = ax.imshow(spectrum_diff, cmap='RdBu_r', 
                            extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]],
                            origin='lower', vmin=-2, vmax=2)
                ax.set_title(f'Spectrum Difference: Detected - Clean\n({len(detected_spectra)} vs {len(clean_spectra)} images)', 
                            fontsize=12, fontweight='bold')
                ax.set_xlabel('Normalized Frequency fx', fontsize=11)
                ax.set_ylabel('Normalized Frequency fy', fontsize=11)
                plt.colorbar(im, ax=ax, shrink=0.8, label='Log10 Difference')
                
            elif clean_spectra:
                avg_clean = np.mean(clean_spectra, axis=0)
                rows, cols = avg_clean.shape
                freq_x = np.linspace(-0.5, 0.5, cols)
                freq_y = np.linspace(-0.5, 0.5, rows)
                avg_clean_log = np.log10(avg_clean + avg_clean[avg_clean > 0].min())
                
                im = ax.imshow(avg_clean_log, cmap='gray', 
                            extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]],
                            origin='lower')
                ax.set_title(f'Average Clean Spectrum\n({len(clean_spectra)} images)', 
                            fontsize=12, fontweight='bold')
                ax.set_xlabel('Normalized Frequency fx', fontsize=11)
                ax.set_ylabel('Normalized Frequency fy', fontsize=11)
                plt.colorbar(im, ax=ax, shrink=0.8, label='Log10 Magnitude')
            else:
                ax.text(0.5, 0.5, 'No spectrum data available', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, f'Spectrum analysis failed:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

    @staticmethod
    def _plot_pmap_statistics(ax, valid_results):
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
                    ax.bar(x_pos - width/2, clean_means, width, label=f'Clean ({len(clean_stats)})', 
                        color='#2ca02c', alpha=0.8, edgecolor='darkgreen')
                
                if detected_stats:
                    detected_means = [np.mean([s[m] for s in detected_stats]) for m in metrics]
                    ax.bar(x_pos + width/2, detected_means, width, label=f'Detected ({len(detected_stats)})', 
                        color='#d62728', alpha=0.8, edgecolor='darkred')
                
                ax.set_xlabel('P-Map Metrics', fontsize=11, fontweight='bold')
                ax.set_ylabel('Average Value', fontsize=11, fontweight='bold')
                ax.set_title('P-Map Statistical Comparison', fontsize=12, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(metric_labels, rotation=45, ha='right')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No P-map data available', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, f'P-map analysis failed:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

    @staticmethod
    def _plot_gradient_progression(ax, valid_results):
        gradients = [r.get('max_gradient', 0) for r in valid_results if r.get('max_gradient') is not None]
        
        if gradients:
            image_indices = list(range(len(valid_results)))
            gradient_values = [r.get('max_gradient', 0) for r in valid_results if r.get('max_gradient') is not None]
            detection_status = [r['detected'] for r in valid_results if r.get('max_gradient') is not None]
            
            detected_indices = [i for i, detected in enumerate(detection_status) if detected]
            clean_indices = [i for i, detected in enumerate(detection_status) if not detected]
            detected_grads = [gradient_values[i] for i in detected_indices]
            clean_grads = [gradient_values[i] for i in clean_indices]
            
            if clean_indices:
                ax.scatter(clean_indices, clean_grads, 
                        c='lightgreen', alpha=0.6, s=60, label=f'Clean Images ({len(clean_indices)})', 
                        marker='o', edgecolors='darkgreen', linewidths=1.5)
            
            if detected_indices:
                ax.scatter(detected_indices, detected_grads, 
                        c='lightcoral', alpha=0.8, s=80, label=f'Detected Images ({len(detected_indices)})', 
                        marker='^', edgecolors='darkred', linewidths=1.5)
            
            # Rolling average
            window_size = max(3, len(gradient_values) // 8)
            rolling_gradient = []
            for i in range(len(gradient_values)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(gradient_values), i + window_size // 2 + 1)
                window_grads = gradient_values[start_idx:end_idx]
                rolling_gradient.append(sum(window_grads) / len(window_grads))
            
            ax.plot(image_indices[:len(rolling_gradient)], rolling_gradient, color='#000080', linestyle='-', 
                    linewidth=2.5, alpha=0.9, label=f'Rolling Average (window={window_size})')
            
            # Threshold line
            if valid_results and 'gradient_threshold' in valid_results[0]:
                threshold = valid_results[0]['gradient_threshold']
                ax.axhline(y=threshold, color='black', linestyle='--', 
                        linewidth=2.5, alpha=0.8, 
                        label=f'Threshold: {threshold:.6f}')

            ax.set_xlabel('Image Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('Max ∇C(f)', fontsize=12, fontweight='bold')
            ax.set_title('Individual Image Gradients vs Processing Order', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.4, linestyle='--')

    @staticmethod
    def _plot_detection_confidence(ax, valid_results):
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
                    ax.scatter(clean_margins, clean_times, c='lightgreen', alpha=0.6, s=60, 
                            label=f'Clean ({len(clean_margins)})', marker='o', 
                            edgecolors='darkgreen', linewidths=1.5)
                
                if detected_margins:
                    ax.scatter(detected_margins, detected_times, c='lightcoral', alpha=0.8, s=80, 
                            label=f'Detected ({len(detected_margins)})', marker='^', 
                            edgecolors='darkred', linewidths=1.5)
                
                ax.axvline(x=0, color='black', linestyle='--', linewidth=2.5, alpha=0.8, 
                        label='Detection Threshold')
                
                ax.set_xlabel('Detection Margin (Gradient - Threshold)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Processing Time (seconds)', fontsize=11, fontweight='bold')
                ax.set_title('Detection Confidence vs Processing Time', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                if processing_times:
                    avg_time = np.mean(processing_times)
                    summary_text = f'Avg Processing: {avg_time:.3f}s\n'
                    summary_text += f'Total Images: {len(valid_results)}\n'
                    summary_text += f'Detected: {sum(detection_status)}\n'
                    summary_text += f'Threshold: {threshold:.6f}'
                    
                    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
                            fontsize=10, fontweight='bold', va='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Insufficient data for\nconfidence analysis', 
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=14, fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, f'Confidence analysis failed:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)