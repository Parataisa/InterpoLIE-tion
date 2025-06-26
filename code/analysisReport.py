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
                'processing_time': result.get('processing_time', None),
                'max_gradient': result.get('max_gradient', None) 
            }
            
            if row['max_gradient'] is None and 'detailed_metrics' in result and result['detailed_metrics']:
                metrics = result['detailed_metrics']
                row.update({
                    'max_gradient': metrics.get('max_gradient', 0.0),
                    'gradient_threshold': metrics.get('gradient_threshold', 0.008),
                    'spectrum_mean': metrics.get('spectrum_mean', 0.0),
                    'spectrum_std': metrics.get('spectrum_std', 0.0),
                    'spectrum_max': metrics.get('spectrum_max', 0.0)
                })
            else:
                row.update({
                    'max_gradient': row['max_gradient'] or 0.0,
                    'gradient_threshold': result.get('gradient_threshold', 0.008),
                    'spectrum_mean': result.get('spectrum_mean', 0.0),
                    'spectrum_std': result.get('spectrum_std', 0.0),
                    'spectrum_max': result.get('spectrum_max', 0.0)
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
        
        if 'max_gradient' not in merged_df.columns:
            merged_df['max_gradient'] = 0.0
        else:
            merged_df['max_gradient'] = merged_df['max_gradient'].fillna(0.0)
        
        agg_dict = {
            'detected': ['count', 'sum', 'mean'],
            'processing_time': 'mean'
        }
        
        if 'max_gradient' in merged_df.columns and merged_df['max_gradient'].notna().any():
            agg_dict['max_gradient'] = 'mean'
        
        scaling_analysis = merged_df.groupby(['scaling_factor', 'interpolation']).agg(agg_dict).round(6)
        
        if len(agg_dict) == 3: 
            scaling_analysis.columns = ['total_images', 'detected_count', 'detection_rate', 
                                       'avg_processing_time', 'avg_max_gradient']
        else:
            scaling_analysis.columns = ['total_images', 'detected_count', 'detection_rate', 
                                       'avg_processing_time']
            scaling_analysis['avg_max_gradient'] = 0.0
        
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
                'processing_time': result.get('processing_time', None),
                'max_gradient': result.get('max_gradient', None)
            }
            
            if row['max_gradient'] is None and 'detailed_metrics' in result and result['detailed_metrics']:
                metrics = result['detailed_metrics']
                row.update({
                    'max_gradient': metrics.get('max_gradient', 0.0),
                    'gradient_threshold': metrics.get('gradient_threshold', 0.008),
                    'spectrum_mean': metrics.get('spectrum_mean', 0.0),
                    'spectrum_std': metrics.get('spectrum_std', 0.0),
                    'spectrum_max': metrics.get('spectrum_max', 0.0)
                })
            else:
                row.update({
                    'max_gradient': row['max_gradient'] or 0.0,
                    'gradient_threshold': result.get('gradient_threshold', 0.008),
                    'spectrum_mean': result.get('spectrum_mean', 0.0),
                    'spectrum_std': result.get('spectrum_std', 0.0),
                    'spectrum_max': result.get('spectrum_max', 0.0)
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
        
        if 'max_gradient' not in merged_df.columns:
            merged_df['max_gradient'] = 0.0
        else:
            merged_df['max_gradient'] = merged_df['max_gradient'].fillna(0.0)
        
        agg_dict = {
            'detected': ['count', 'sum', 'mean'],
            'processing_time': 'mean'
        }
        
        if 'max_gradient' in merged_df.columns and merged_df['max_gradient'].notna().any():
            agg_dict['max_gradient'] = 'mean'
        
        rotation_analysis = merged_df.groupby(['rotation_angle', 'interpolation']).agg(agg_dict).round(6)
        
        if len(agg_dict) == 3: 
            rotation_analysis.columns = ['total_images', 'detected_count', 'detection_rate', 
                                        'avg_processing_time', 'avg_max_gradient']
        else:
            rotation_analysis.columns = ['total_images', 'detected_count', 'detection_rate', 
                                        'avg_processing_time']
            rotation_analysis['avg_max_gradient'] = 0.0
        
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
    def create_batch_analysis_report(results, output_folder):
        valid_results = [r for r in results if 'error' not in r and r.get('detected') is not None]
        
        if not valid_results:
            print("No valid results for analysis report")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Kirchner Detector: Batch Analysis Report', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        AnalysisReports._plot_batch_detection_summary(axes[0, 0], valid_results)
        AnalysisReports._plot_batch_spectrum_analysis(axes[0, 1], valid_results)
        AnalysisReports._plot_batch_pmap_statistics(axes[1, 0], valid_results)
        AnalysisReports._plot_batch_gradient_progression(axes[1, 1], valid_results)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plot_path = output_folder / 'batch_analysis_report.png'
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
        plt.close()

    @staticmethod
    def create_scaling_report(analysis_results, output_path):
        detailed_df = analysis_results['detailed_results']
        scaling_df = analysis_results['scaling_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Kirchner Detector: Scaling Factor Analysis', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        AnalysisReports._plot_detection_vs_parameter(axes[0, 0], scaling_df, 'scaling_factor', 'Scaling Factor', 1.0)
        AnalysisReports._plot_category_analysis(axes[0, 1], detailed_df, ['original', 'upscaled', 'downscaled'])
        AnalysisReports._plot_parameter_heatmap(axes[1, 0], scaling_df, 'scaling_factor', 'Scaling Factor', '{:.1f}')
        AnalysisReports._plot_gradient_vs_parameter(axes[1, 1], detailed_df, scaling_df, 'scaling_factor', 'Scaling Factor', 1.0)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plot_path = output_path / 'scaling_analysis_report.png'
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
        plt.close()

    @staticmethod
    def create_rotation_report(analysis_results, output_path):
        detailed_df = analysis_results['detailed_results']
        rotation_df = analysis_results['rotation_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Kirchner Detector: Rotation Angle Analysis', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        AnalysisReports._plot_detection_vs_parameter(axes[0, 0], rotation_df, 'rotation_angle', 'Rotation Angle (degrees)', None)
        AnalysisReports._plot_category_analysis(axes[0, 1], detailed_df, ['original', 'rotated'])
        AnalysisReports._plot_parameter_heatmap(axes[1, 0], rotation_df, 'rotation_angle', 'Rotation Angle', '{:.0f}°')
        AnalysisReports._plot_gradient_vs_parameter(axes[1, 1], detailed_df, rotation_df, 'rotation_angle', 'Rotation Angle (degrees)', None)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plot_path = output_path / 'rotation_analysis_report.png'
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
        plt.close()

    @staticmethod
    def _plot_detection_vs_parameter(ax, analysis_df, param_col, param_label, reference_line=None):
        if len(analysis_df) > 0:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            markers = ['o', 's', '^', 'D', 'v', 'P']
            linestyles = ['-', '--', '-.', ':', '-', '--']
            
            for i, interp_method in enumerate(analysis_df['interpolation'].unique()):
                method_data = analysis_df[analysis_df['interpolation'] == interp_method]
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                linestyle = linestyles[i % len(linestyles)]
                
                ax.plot(method_data[param_col], method_data['detection_rate'], 
                        marker=marker, linestyle=linestyle, label=interp_method, 
                        linewidth=2.5, markersize=8, color=color, markeredgecolor='white',
                        markeredgewidth=1, alpha=0.9)
            
            ax.set_xlabel(param_label, fontsize=12, fontweight='bold')
            ax.set_ylabel('Detection Rate', fontsize=12, fontweight='bold')
            ax.set_title(f'Detection Rate vs {param_label}', fontsize=14, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.set_ylim(-0.05, 1.05)
            
            if reference_line is not None:
                ax.axvline(x=reference_line, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
                if param_col == 'scaling_factor':
                    ax.axvspan(min(analysis_df[param_col]), reference_line, alpha=0.1, color='red')
                    ax.axvspan(reference_line, max(analysis_df[param_col]), alpha=0.1, color='blue')
            else:
                param_range = max(analysis_df[param_col]) - min(analysis_df[param_col])
                ax.set_xlim(min(analysis_df[param_col]) - param_range*0.05, 
                           max(analysis_df[param_col]) + param_range*0.05)

    @staticmethod
    def _plot_category_analysis(ax, detailed_df, categories):
        if len(detailed_df) > 0:
            categories_data = {}
            for category in categories:
                cat_data = detailed_df[detailed_df['category'] == category]
                if len(cat_data) > 0:
                    detected = cat_data['detected'].sum()
                    total = len(cat_data)
                    categories_data[category] = {'detected': detected, 'total': total, 'rate': detected/total}
            
            if categories_data:
                categories_list = list(categories_data.keys())
                detection_rates = [categories_data[cat]['rate'] for cat in categories_list]
                totals = [categories_data[cat]['total'] for cat in categories_list]
                
                colors_cat = {
                    'original': '#808080', 'rotated': '#d62728', 'upscaled': '#2ca02c', 
                    'downscaled': '#d62728', 'clean': '#2ca02c', 'detected': '#d62728'
                }
                bar_colors = [colors_cat.get(cat, '#1f77b4') for cat in categories_list]
                
                bars = ax.bar(categories_list, detection_rates, color=bar_colors, alpha=0.8, 
                              edgecolor='white', linewidth=2)
                ax.set_ylabel('Detection Rate', fontsize=12, fontweight='bold')
                ax.set_title('Detection Rate by Category', fontsize=14, fontweight='bold')
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.4, axis='y')
                
                for bar, rate, total in zip(bars, detection_rates, totals):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{rate:.2f}\n({total} imgs)', ha='center', va='bottom', 
                            fontsize=11, fontweight='bold')

    @staticmethod
    def _plot_parameter_heatmap(ax, analysis_df, param_col, param_label, format_str):
        if len(analysis_df) > 0 and len(analysis_df['interpolation'].unique()) > 1:
            pivot_data = analysis_df.pivot(index='interpolation', columns=param_col, values='detection_rate')
            
            colors_heatmap = ['#8B0000', '#FF4500', '#FFD700', '#90EE90', '#006400']
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('custom', colors_heatmap, N=n_bins)
            
            im = ax.imshow(pivot_data.values, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                            interpolation='nearest')
            
            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_xticklabels([format_str.format(x) for x in pivot_data.columns], rotation=45, fontsize=10)
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels(pivot_data.index, fontsize=10)
            ax.set_xlabel(param_label, fontsize=12, fontweight='bold')
            ax.set_ylabel('Interpolation Method', fontsize=12, fontweight='bold')
            ax.set_title('Detection Rate Heatmap', fontsize=14, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label('Detection Rate', fontsize=11, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)

    @staticmethod
    def _plot_gradient_vs_parameter(ax, detailed_df, analysis_df, param_col, param_label, reference_line=None):
        if len(detailed_df) > 0:
            valid_data = detailed_df.dropna(subset=['max_gradient', param_col])
            
            if len(valid_data) > 0:
                detected_data = valid_data[valid_data['detected'] == True]
                clean_data = valid_data[valid_data['detected'] == False]
                
                if len(clean_data) > 0:
                    ax.scatter(clean_data[param_col], clean_data['max_gradient'], 
                              c='lightgreen', alpha=0.6, s=25, label=f'Clean Images ({len(clean_data)})', 
                              marker='o', edgecolors='darkgreen', linewidths=0.5)
                
                if len(detected_data) > 0:
                    ax.scatter(detected_data[param_col], detected_data['max_gradient'], 
                              c='lightcoral', alpha=0.8, s=40, label=f'Detected Images ({len(detected_data)})', 
                              marker='^', edgecolors='darkred', linewidths=0.5)
                
                colors_trend = ['#000080', '#8B0000', '#006400', '#FF8C00', '#4B0082', '#8B4513']
                markers_trend = ['o', 's', '^', 'D', 'v', 'P']
                linestyles_trend = ['-', '--', '-.', ':', '-', '--']
                
                for i, interp_method in enumerate(analysis_df['interpolation'].unique()):
                    method_data = analysis_df[analysis_df['interpolation'] == interp_method]
                    color = colors_trend[i % len(colors_trend)]
                    marker = markers_trend[i % len(markers_trend)]
                    linestyle = linestyles_trend[i % len(linestyles_trend)]
                    
                    ax.plot(method_data[param_col], method_data['avg_max_gradient'], 
                            marker=marker, linestyle=linestyle, 
                            label=f'{interp_method} (avg)', linewidth=2.5, markersize=7, 
                            color=color, alpha=0.9, markeredgecolor='white', markeredgewidth=1)

                if 'gradient_threshold' in valid_data.columns:
                    gradient_thresh = valid_data['gradient_threshold'].dropna().unique()
                    if len(gradient_thresh) > 0:
                        ax.axhline(y=gradient_thresh[0], color='black', linestyle='--', 
                                   linewidth=2.5, alpha=0.8, 
                                   label=f'Threshold: {gradient_thresh[0]:.4f}')

                if reference_line is not None:
                    ax.axvline(x=reference_line, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

                ax.set_xlabel(param_label, fontsize=12, fontweight='bold')
                ax.set_ylabel('Max ∇C(f)', fontsize=12, fontweight='bold')
                ax.set_title(f'Gradients vs {param_label}', fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, 
                          frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.4, linestyle='--')

    @staticmethod
    def _plot_batch_detection_summary(ax, valid_results):
        detected_count = sum(1 for r in valid_results if r['detected'])
        clean_count = len(valid_results) - detected_count
        
        categories = ['Clean', 'Detected']
        counts = [clean_count, detected_count]
        colors = ['#2ca02c', '#d62728']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_title('Detection Summary', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.4, axis='y')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = count / len(valid_results) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + len(valid_results)*0.01,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
        
        if valid_results:
            avg_time = np.mean([r.get('processing_time', 0) for r in valid_results])
            summary_text = f'Total: {len(valid_results)} images\n'
            summary_text += f'Avg Time: {avg_time:.3f}s\n'
            summary_text += f'Throughput: {len(valid_results)/sum([r.get("processing_time", 0) for r in valid_results]):.1f} img/s'
            
            ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, 
                    fontsize=10, fontweight='bold', va='top', ha='right',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

    @staticmethod
    def _plot_batch_spectrum_analysis(ax, valid_results):
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
    def _plot_batch_pmap_statistics(ax, valid_results):
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
    def _plot_batch_gradient_progression(ax, valid_results):
        try:
            gradient_data = []
            for i, result in enumerate(valid_results):
                max_grad = result.get('max_gradient')
                if max_grad is not None and not (isinstance(max_grad, float) and np.isnan(max_grad)):
                    gradient_data.append({
                        'index': i,
                        'max_gradient': max_grad,
                        'detected': result.get('detected', False),
                        'filename': result.get('file_name', f'image_{i}')
                    })
            
            if not gradient_data:
                ax.text(0.5, 0.5, 'No valid gradient data available', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, fontweight='bold')
                return
            
            indices = [d['index'] for d in gradient_data]
            gradients = [d['max_gradient'] for d in gradient_data]
            detected_flags = [d['detected'] for d in gradient_data]
            
            clean_indices = [indices[i] for i, detected in enumerate(detected_flags) if not detected]
            clean_gradients = [gradients[i] for i, detected in enumerate(detected_flags) if not detected]
            detected_indices = [indices[i] for i, detected in enumerate(detected_flags) if detected]
            detected_gradients = [gradients[i] for i, detected in enumerate(detected_flags) if detected]
            
            if clean_indices:
                ax.scatter(clean_indices, clean_gradients, 
                          c='lightgreen', alpha=0.7, s=60, label=f'Clean Images ({len(clean_indices)})', 
                          marker='o', edgecolors='darkgreen', linewidths=1.5)
            
            if detected_indices:
                ax.scatter(detected_indices, detected_gradients, 
                          c='lightcoral', alpha=0.8, s=80, label=f'Detected Images ({len(detected_indices)})', 
                          marker='^', edgecolors='darkred', linewidths=1.5)
            
            if len(gradients) > 3:
                window_size = max(3, len(gradients) // 10)
                rolling_gradient = []
                rolling_indices = []
                
                for i in range(len(gradients)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(gradients), i + window_size // 2 + 1)
                    window_grads = gradients[start_idx:end_idx]
                    rolling_gradient.append(sum(window_grads) / len(window_grads))
                    rolling_indices.append(indices[i])
                
                ax.plot(rolling_indices, rolling_gradient, color='#000080', linestyle='-', 
                        linewidth=2.5, alpha=0.9, label=f'Rolling Average (window={window_size})')
            
            thresholds = [result.get('gradient_threshold') for result in valid_results 
                         if result.get('gradient_threshold') is not None]
            if thresholds:
                threshold = thresholds[0]
                ax.axhline(y=threshold, color='black', linestyle='--', 
                          linewidth=2.5, alpha=0.8, 
                          label=f'Threshold: {threshold:.6f}')
            
            stats_text = f'Total gradients: {len(gradient_data)}\n'
            stats_text += f'Mean: {np.mean(gradients):.6f}\n'
            stats_text += f'Max: {np.max(gradients):.6f}\n'
            stats_text += f'Min: {np.min(gradients):.6f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
            
            ax.set_xlabel('Image Processing Order', fontsize=12, fontweight='bold')
            ax.set_ylabel('Max ∇C(f)', fontsize=12, fontweight='bold')
            ax.set_title('Gradient Progression Through Batch', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.4, linestyle='--')
            
            if gradients:
                y_margin = (max(gradients) - min(gradients)) * 0.1
                ax.set_ylim(min(gradients) - y_margin, max(gradients) + y_margin)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Gradient progression failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))