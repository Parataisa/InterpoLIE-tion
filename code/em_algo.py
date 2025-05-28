import numpy as np
import matplotlib.pyplot as plt
import cv2
import psutil
import gc
import time

from PIL import Image
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
from skimage.transform import resize, rotate
from skimage import data
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from functools import partial

# Exposing Digital Forgeries by Detecting Traces of Resampling
# by Alin C. Popescu and Hany Farid

IMAGE_NAME = 'image_4'  

def process_chunk_worker(args):
    padded_img_chunk, start_row, start_col, end_row, end_col, alpha, sigma, K, num_weights = args
    
    chunk_rows = end_row - start_row
    chunk_cols = end_col - start_col
    n_pixels = chunk_rows * chunk_cols
    
    neighborhoods = np.zeros((n_pixels, num_weights))
    center_values = np.zeros(n_pixels)
    
    idx = 0
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            if (i+2*K+1 <= padded_img_chunk.shape[0] and j+2*K+1 <= padded_img_chunk.shape[1]):
                neighborhood = padded_img_chunk[i:i + 2*K + 1, j:j + 2*K + 1]
                
                if neighborhood.size == (2*K+1)**2:
                    neighborhood_flat = neighborhood.flatten()
                    center_idx = len(neighborhood_flat) // 2
                    
                    center_values[idx] = neighborhood_flat[center_idx]
                    neighbors = np.concatenate([
                        neighborhood_flat[:center_idx],
                        neighborhood_flat[center_idx + 1:]
                    ])
                    
                    if len(neighbors) == num_weights:
                        neighborhoods[idx] = neighbors
                        idx += 1
    
    if idx < n_pixels:
        neighborhoods = neighborhoods[:idx]
        center_values = center_values[:idx]
        n_pixels = idx
    
    if n_pixels == 0:
        return {
            'alpha_numerator': np.zeros_like(alpha),
            'alpha_denominator': np.zeros((num_weights, num_weights)),
            'sigma_numerator': 0.0,
            'sigma_denominator': 0.0,
            'total_weight': 0.0,
            'n_pixels': 0
        }
    
    residuals = center_values - np.dot(neighborhoods, alpha)
    
    likelihood_M1 = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * \
                    np.exp(-0.5 * (residuals / sigma) ** 2)
    
    residual_range = np.max(residuals) - np.min(residuals) + 1e-8
    likelihood_M2 = np.ones_like(residuals) / residual_range
    
    prior_M1 = prior_M2 = 0.5
    
    numerator = likelihood_M1 * prior_M1
    denominator = likelihood_M1 * prior_M1 + likelihood_M2 * prior_M2
    
    posterior_M1 = numerator / (denominator + 1e-8)
    
    W = np.diag(posterior_M1)
    XTW = neighborhoods.T @ W
    alpha_denominator = XTW @ neighborhoods
    alpha_numerator = XTW @ center_values
    
    weighted_residuals_sq = posterior_M1 * (residuals ** 2)
    sigma_numerator = np.sum(weighted_residuals_sq)
    sigma_denominator = np.sum(posterior_M1)
    total_weight = np.sum(posterior_M1)
    
    return {
        'alpha_numerator': alpha_numerator,
        'alpha_denominator': alpha_denominator,
        'sigma_numerator': sigma_numerator,
        'sigma_denominator': sigma_denominator,
        'total_weight': total_weight,
        'n_pixels': n_pixels
    }

def process_pmap_row_worker(args):
    padded_img_row, row_idx, K, alpha, sigma, cols = args
    
    row_neighborhoods = []
    row_centers = []
    
    for j in range(K, cols - K):
        neighborhood = padded_img_row[:, j-K:j+K+1]
        neighborhood_flat = neighborhood.flatten()
        center_idx = len(neighborhood_flat) // 2
        
        center_val = neighborhood_flat[center_idx]
        neighbors = np.concatenate([
            neighborhood_flat[:center_idx],
            neighborhood_flat[center_idx + 1:]
        ])
        
        row_neighborhoods.append(neighbors)
        row_centers.append(center_val)
    
    if row_neighborhoods:
        row_neighborhoods = np.array(row_neighborhoods)
        row_centers = np.array(row_centers)
        
        residuals = row_centers - np.dot(row_neighborhoods, alpha)
        
        likelihood_M1 = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * \
                        np.exp(-0.5 * (residuals / sigma) ** 2)
        
        residual_range = np.max(residuals) - np.min(residuals) + 1e-8
        likelihood_M2 = np.ones_like(residuals) / residual_range
        
        prior_M1 = prior_M2 = 0.5
        
        numerator = likelihood_M1 * prior_M1
        denominator = likelihood_M1 * prior_M1 + likelihood_M2 * prior_M2
        
        posterior_M1 = numerator / (denominator + 1e-8)
        
        return row_idx, posterior_M1
    
    return row_idx, None

class ParallelMemoryEfficientEMDetector:
    def __init__(self, neighborhood_size=5, max_iterations=20, tolerance=1e-4, 
                 chunk_size=64, max_memory_mb=1000, n_workers=12):
        
        self.K = neighborhood_size // 2 
        self.neighborhood_size = neighborhood_size
        self.max_iter = max_iterations
        self.tolerance = tolerance
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.n_workers = n_workers
        print(f"🚀 Using {self.n_workers} parallel workers")
        
        self.alpha = None
        self.sigma = None
        self.num_weights = neighborhood_size ** 2 - 1 
        
        self.total_pixels_processed = 0
        self.memory_usage_mb = 0
    
    def detect_resampling(self, img):
        if isinstance(img, str):
            img = self._load_image(img)
        
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        print(f"🧠 Processing {img.shape} image with parallel EM algorithm")
        print(f"📦 Chunk size: {self.chunk_size}×{self.chunk_size}")
        print(f"💾 Memory limit: {self.max_memory_mb} MB")
        print(f"⚡ Workers: {self.n_workers}")
        
        start_time = time.time()
        p_map = self._parallel_em_algorithm(img)
        total_time = time.time() - start_time
        
        print(f"⏱️ Total processing time: {total_time:.2f} seconds")
        
        spectrum = self._analyze_spectrum(p_map)
        is_resampled = self._detect_peaks(spectrum)
        
        return p_map, spectrum, is_resampled
    
    def _parallel_em_algorithm(self, img):
        rows, cols = img.shape
        
        self._initialize_parameters()
        
        print(f"🚀 Starting parallel EM with {self.num_weights} predictor weights")
        
        padded_img = np.pad(img, self.K, mode='reflect')
        
        for iteration in range(self.max_iter):
            alpha_old = self.alpha.copy()
            
            iter_start_time = time.time()
            chunk_stats = self._parallel_process_image_chunks(padded_img, iteration)
            iter_time = time.time() - iter_start_time
            
            self._update_global_parameters(chunk_stats)
            
            alpha_change = np.linalg.norm(self.alpha - alpha_old)
            
            if iteration % 5 == 0:
                gc.collect()
            
            print(f"Iteration {iteration + 1:2d}: α change = {alpha_change:.6f}, "
                  f"σ = {self.sigma:.6f}, time = {iter_time:.2f}s, "
                  f"Memory = {self._get_memory_usage():.1f} MB")
            
            if alpha_change < self.tolerance:
                print(f"✅ Converged after {iteration + 1} iterations")
                break
        
        p_map = self._parallel_generate_final_p_map(img)
        
        print(f"📊 Processed {self.total_pixels_processed:,} pixels total")
        
        return p_map
    
    def _initialize_parameters(self):
        self.alpha = np.random.normal(0, 0.01, self.num_weights)
        self.sigma = 0.1
        self.total_pixels_processed = 0
    
    def _parallel_process_image_chunks(self, padded_img, iteration):
        rows, cols = padded_img.shape
        
        chunks = []
        step_size = max(1, self.chunk_size // 2)
        
        for start_row in range(self.K, rows - self.K, step_size):
            for start_col in range(self.K, cols - self.K, step_size):
                end_row = min(start_row + self.chunk_size, rows - self.K)
                end_col = min(start_col + self.chunk_size, cols - self.K)
                
                chunk_rows = end_row - start_row
                chunk_cols = end_col - start_col
                
                if chunk_rows < 5 or chunk_cols < 5:
                    continue
                
                extract_start_row = max(0, start_row - self.K)
                extract_end_row = min(rows, end_row + self.K)
                extract_start_col = max(0, start_col - self.K)
                extract_end_col = min(cols, end_col + self.K)
                
                padded_img_chunk = padded_img[extract_start_row:extract_end_row, 
                                             extract_start_col:extract_end_col]
                
                rel_start_row = start_row - extract_start_row
                rel_start_col = start_col - extract_start_col
                rel_end_row = rel_start_row + chunk_rows
                rel_end_col = rel_start_col + chunk_cols
                
                chunks.append((padded_img_chunk, rel_start_row, rel_start_col, 
                              rel_end_row, rel_end_col, self.alpha.copy(), 
                              self.sigma, self.K, self.num_weights))
        
        print(f"🔄 Processing {len(chunks)} chunks in parallel...")
        
        start_time = time.time()
        
        chunk_results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_chunk = {executor.submit(process_chunk_worker, chunk): i 
                              for i, chunk in enumerate(chunks)}
            
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    chunk_results.append(result)
                except Exception as e:
                    print(f"⚠️ Chunk processing failed: {e}")
        
        parallel_time = time.time() - start_time
        
        return self._aggregate_chunk_results(chunk_results, parallel_time, len(chunks))
    
    def _aggregate_chunk_results(self, chunk_results, parallel_time, num_chunks):
        alpha_numerator = np.zeros(self.num_weights)
        alpha_denominator = np.zeros((self.num_weights, self.num_weights))
        sigma_numerator = 0.0
        sigma_denominator = 0.0
        total_weight = 0.0
        total_pixels = 0
        
        for result in chunk_results:
            alpha_numerator += result['alpha_numerator']
            alpha_denominator += result['alpha_denominator']
            sigma_numerator += result['sigma_numerator']
            sigma_denominator += result['sigma_denominator']
            total_weight += result['total_weight']
            total_pixels += result['n_pixels']
        
        self.total_pixels_processed += total_pixels
        
        print(f"⚡ Parallel processing: {num_chunks} chunks in {parallel_time:.2f}s")
        
        return {
            'alpha_numerator': alpha_numerator,
            'alpha_denominator': alpha_denominator,
            'sigma_numerator': sigma_numerator,
            'sigma_denominator': sigma_denominator,
            'total_weight': total_weight,
            'chunk_count': len(chunk_results)
        }
    
    def _update_global_parameters(self, chunk_stats):
        try:
            reg_term = 1e-6 * np.eye(chunk_stats['alpha_denominator'].shape[0])
            regularized_denominator = chunk_stats['alpha_denominator'] + reg_term
            
            self.alpha = np.linalg.solve(regularized_denominator, chunk_stats['alpha_numerator'])
            
        except np.linalg.LinAlgError:
            print("⚠️  Numerical instability in α update, using smaller step")
            if chunk_stats['total_weight'] > 0:
                gradient = chunk_stats['alpha_numerator'] / (chunk_stats['total_weight'] + 1e-8)
                self.alpha = 0.9 * self.alpha + 0.1 * gradient
        
        if chunk_stats['sigma_denominator'] > 0:
            new_sigma = np.sqrt(chunk_stats['sigma_numerator'] / chunk_stats['sigma_denominator'])
            self.sigma = np.clip(new_sigma, 0.01, 1.0)
    
    def _parallel_generate_final_p_map(self, img):
        """Generate final probability map using parallel row processing"""
        rows, cols = img.shape
        p_map = np.zeros((rows, cols))
        
        padded_img = np.pad(img, self.K, mode='reflect')
        
        print("🎨 Generating final probability map in parallel...")
        
        row_args = []
        for i in range(self.K, rows - self.K):
            row_start = i - self.K
            row_end = i + self.K + 1
            padded_img_row = padded_img[row_start:row_end, :]
            
            row_args.append((padded_img_row, i, self.K, self.alpha.copy(), 
                           self.sigma, cols))
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_row = {executor.submit(process_pmap_row_worker, arg): i 
                           for i, arg in enumerate(row_args)}
            
            for future in as_completed(future_to_row):
                try:
                    row_idx, posterior_probs = future.result()
                    if posterior_probs is not None:
                        p_map[row_idx, self.K:cols-self.K] = posterior_probs
                except Exception as e:
                    print(f"⚠️ Row processing failed: {e}")
        
        pmap_time = time.time() - start_time
        print(f"⚡ P-map generation: {len(row_args)} rows in {pmap_time:.2f}s")
        
        p_map = gaussian_filter(p_map, sigma=0.5)
        
        return p_map
    
    def _analyze_spectrum(self, p_map):
        p_map_hp = p_map - gaussian_filter(p_map, sigma=3)
        
        rows, cols = p_map_hp.shape
        window_r = np.hanning(rows).reshape(-1, 1)
        window_c = np.hanning(cols).reshape(1, -1)
        window = window_r @ window_c
        p_map_windowed = p_map_hp * window
        
        spectrum = np.abs(fft2(p_map_windowed))
        spectrum = fftshift(spectrum)
        
        spectrum = spectrum / (np.max(spectrum) + 1e-8)
        
        return spectrum
    
    def _detect_peaks(self, spectrum):
        rows, cols = spectrum.shape
        center_r, center_c = rows // 2, cols // 2
        
        exclude_radius = min(rows, cols) // 8
        mask = np.ones((rows, cols), dtype=bool)
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_c)**2 + (y - center_r)**2)
        mask[distance < exclude_radius] = False
        
        spectrum_masked = spectrum[mask]
        mean_val = np.mean(spectrum_masked)
        std_val = np.std(spectrum_masked)
        
        threshold = mean_val + 2.5 * std_val
        
        num_peaks = np.sum(spectrum_masked > threshold)
        peak_ratio = num_peaks / len(spectrum_masked)
        max_peak = np.max(spectrum_masked)
        
        is_resampled = (num_peaks > 10) and (peak_ratio > 0.001) and (max_peak > 0.1)
        
        print(f"🔍 Peak detection: {num_peaks} peaks, ratio: {peak_ratio:.4f}, max: {max_peak:.4f}")
        
        return is_resampled
    
    def _get_memory_usage(self):
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage_mb = memory_mb
        return memory_mb
    
    def _load_image(self, image_path):
        img = Image.open(image_path).convert('L')
        return np.array(img) / 255.0
    
    def test_single_image(self, image_path_or_array, max_size):
        print("🚀 Starting Parallel Memory-Efficient Full EM Resampling Detection...")
        print("="*70)
        
        if isinstance(image_path_or_array, str):
            display_img = self._load_image(image_path_or_array)
        else:
            display_img = image_path_or_array
            if len(display_img.shape) == 3:
                display_img = np.mean(display_img, axis=2)

        if max_size != 0 and max(display_img.shape) > max_size:
            scale_factor = max_size / max(display_img.shape)
            new_shape = (int(display_img.shape[0] * scale_factor), 
                        int(display_img.shape[1] * scale_factor))
            display_img = resize(display_img, new_shape, anti_aliasing=True)
            print(f"📏 Resized image to {new_shape} for efficiency")
        
        initial_memory = self._get_memory_usage()
        print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
        
        overall_start_time = time.time()
        p_map, spectrum, is_resampled = self.detect_resampling(display_img)
        total_processing_time = time.time() - overall_start_time
        
        final_memory = self._get_memory_usage()
        print(f"💾 Final memory usage: {final_memory:.1f} MB")
        print(f"💾 Peak memory increase: {final_memory - initial_memory:.1f} MB")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(display_img, cmap='gray')
        axes[0, 0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(p_map, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('EM Probability Map\n(Red = High correlation)', fontsize=14)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
        
        im2 = axes[0, 2].imshow(np.log(spectrum + 1e-6), cmap='viridis')
        axes[0, 2].set_title('Log Frequency Spectrum\n(Peaks = Periodic patterns)', fontsize=14)
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
        
        result_color = 'red' if is_resampled else 'green'
        result_text = '✓ RESAMPLING DETECTED' if is_resampled else '✗ NO RESAMPLING'
        
        axes[1, 0].text(0.5, 0.7, result_text, ha='center', va='center', 
                        fontsize=16, fontweight='bold', color=result_color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                 edgecolor=result_color, linewidth=2))
        
        param_text = f"""Parallel EM Results:
Workers: {self.n_workers}
Neighborhood: {self.neighborhood_size}×{self.neighborhood_size}
Chunk Size: {self.chunk_size}×{self.chunk_size}
Final σ: {self.sigma:.4f}
α weights: {len(self.alpha)}
Max |α|: {np.max(np.abs(self.alpha)):.4f}

Performance:
Total Time: {total_processing_time:.2f}s
Pixels Processed: {self.total_pixels_processed:,}
Peak Memory: {final_memory:.1f} MB"""
        
        axes[1, 0].text(0.5, 0.3, param_text, ha='center', va='center',
                        fontsize=9, fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Results & Performance', fontsize=14, fontweight='bold')
        
        axes[1, 1].hist(p_map.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('Probability Distribution', fontsize=14)
        axes[1, 1].set_xlabel('P(Correlated | pixel)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(np.mean(p_map), color='blue', linestyle='--', 
                          label=f'Mean: {np.mean(p_map):.3f}')
        axes[1, 1].legend()
        
        if len(self.alpha) <= 25:  
            full_matrix = np.zeros((self.neighborhood_size, self.neighborhood_size))
            center = self.neighborhood_size // 2
            
            idx = 0
            for i in range(self.neighborhood_size):
                for j in range(self.neighborhood_size):
                    if i == center and j == center:
                        full_matrix[i, j] = 0  
                    else:
                        if idx < len(self.alpha):  
                            full_matrix[i, j] = self.alpha[idx]
                        idx += 1
            
            im3 = axes[1, 2].imshow(full_matrix, cmap='RdBu_r', vmin=-np.max(np.abs(self.alpha)), 
                                   vmax=np.max(np.abs(self.alpha)))
            axes[1, 2].set_title('Learned α Weights\n(Blue=negative, Red=positive)', fontsize=14)
            plt.colorbar(im3, ax=axes[1, 2], shrink=0.8)
            
            axes[1, 2].set_xticks(range(self.neighborhood_size))
            axes[1, 2].set_yticks(range(self.neighborhood_size))
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].plot(center, center, 'ko', markersize=10, label='Center (0)')
            axes[1, 2].legend()
        else:
            axes[1, 2].hist(self.alpha, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 2].set_title('α Weight Distribution', fontsize=14)
            axes[1, 2].set_xlabel('Weight Value')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Parallel Memory-Efficient EM Algorithm - Resampling Detection', 
                    fontsize=16, fontweight='bold', y=0.98)
        fig_filename = f'results/em_detection_results_{IMAGE_NAME}.png'
        plt.draw()
        try:
            plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Figure saved as '{fig_filename}'")
        except Exception as e:
            print(f"⚠️ Could not save figure: {e}")
        
        plt.show()
        
        print("\n" + "="*70)
        print("PARALLEL MEMORY-EFFICIENT EM DETECTION RESULTS")
        print("="*70)
        print(f"Resampling Detected: {is_resampled}")
        print(f"Image Shape: {display_img.shape}")
        print(f"Total Processing Time: {total_processing_time:.2f} seconds")
        print(f"Total Pixels Processed: {self.total_pixels_processed:,}")
        print(f"Average Probability: {np.mean(p_map):.4f}")
        print(f"Probability Std: {np.std(p_map):.4f}")
        print("\nEM Algorithm Results:")
        print(f"Final σ (noise std): {self.sigma:.4f}")
        print(f"Number of α weights: {len(self.alpha)}")
        print(f"Max |α| weight: {np.max(np.abs(self.alpha)):.4f}")
        print(f"α weight range: [{np.min(self.alpha):.4f}, {np.max(self.alpha):.4f}]")
        print("\nParallel Performance:")
        print(f"Workers Used: {self.n_workers}")
        print(f"Chunk Size: {self.chunk_size}×{self.chunk_size}")
        print(f"Memory Limit: {self.max_memory_mb} MB")
        print(f"Peak Memory Usage: {final_memory:.1f} MB")
        print("="*70)

def run_parallel_em_demo():
    print("🚀 Parallel Memory-Efficient Full EM Algorithm Demo")
    print("="*70)
    
    detector = ParallelMemoryEfficientEMDetector(
        neighborhood_size=5,     
        max_iterations=15,       
        tolerance=1e-4,          
        chunk_size=32,           
        max_memory_mb=2048,      
        n_workers=24           
    )
    
    try:
        print("\n📷 Testing parallel processing on external image")
        detector.test_single_image(f'img/{IMAGE_NAME}.jpg', max_size=512)
    except Exception as e:
        print(f"External image test failed: {e}")

if __name__ == "__main__":
    run_parallel_em_demo()