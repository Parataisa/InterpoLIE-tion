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

IMAGE_NAME = 'image_2'  

class MemoryEfficientEMDetector:
    def __init__(self, neighborhood_size=5, max_iterations=20, tolerance=1e-4, 
                 chunk_size=64, max_memory_mb=1000):
        
        self.K = neighborhood_size // 2 
        self.neighborhood_size = neighborhood_size
        self.max_iter = max_iterations
        self.tolerance = tolerance
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb

        # EM parameters
        self.alpha = None
        self.sigma = None
        self.num_weights = neighborhood_size ** 2 - 1 
        
        # Statistics
        self.total_pixels_processed = 0
        self.memory_usage_mb = 0
        self.signal_range = None  # For proper M2 modeling
    
    def _initialize_parameters(self, img):
        """Initialize parameters with better defaults based on image statistics"""
        # Better alpha initialization - small but not too small
        self.alpha = np.random.normal(0, 0.01, self.num_weights)
        
        # Initialize sigma based on image noise estimate
        # Use Laplacian of Gaussian to estimate noise
        img_filtered = gaussian_filter(img, sigma=0.5)
        noise_estimate = np.std(img - img_filtered)
        self.sigma = max(0.1, min(0.3, noise_estimate * 2))
        
        # Calculate signal range for M2 model
        self.signal_range = img.max() - img.min()
        
        print(f"   Initialized: α_std = 0.01, σ = {self.sigma:.4f}")
        print(f"   Signal range: {self.signal_range:.4f}")

    def memory_efficient_em_algorithm(self, img):
        rows, cols = img.shape
        
        # Initialize parameters based on image
        self._initialize_parameters(img)
        
        print(f"🚀 Starting FIXED EM with {self.num_weights} predictor weights")
        print(f"   Image size: {rows}×{cols}")
        print(f"   Initialized: α_range=[{np.min(self.alpha):.4f}, {np.max(self.alpha):.4f}], σ={self.sigma:.4f}")
        
        padded_img = np.pad(img, self.K, mode='reflect')
        
        max_pixels_per_batch = self.chunk_size
        total_pixels = (rows - 2*self.K) * (cols - 2*self.K)
        print(f"   Total pixels: {total_pixels:,}, batch size: {max_pixels_per_batch:,}")
        
        for iteration in range(self.max_iter):
            alpha_old = self.alpha.copy()
            sigma_old = self.sigma
            iter_start_time = time.time()
            
            total_alpha_numerator = np.zeros(self.num_weights)
            total_alpha_denominator = np.zeros((self.num_weights, self.num_weights))
            total_sigma_numerator = 0.0
            total_sigma_denominator = 0.0
            total_pixels_processed = 0
            
            batch_count = 0
            current_batch_neighborhoods = []
            current_batch_centers = []
            
            for i in range(self.K, rows - self.K):
                for j in range(self.K, cols - self.K):
                    neighborhood = padded_img[i:i + 2*self.K + 1, j:j + 2*self.K + 1]
                    
                    if neighborhood.size == (2*self.K + 1)**2:
                        neighborhood_flat = neighborhood.flatten()
                        center_idx = len(neighborhood_flat) // 2
                        
                        center_val = neighborhood_flat[center_idx]
                        neighbors = np.concatenate([
                            neighborhood_flat[:center_idx],
                            neighborhood_flat[center_idx + 1:]
                        ])
                        
                        if len(neighbors) == self.num_weights:
                            current_batch_neighborhoods.append(neighbors)
                            current_batch_centers.append(center_val)
                            
                            if len(current_batch_neighborhoods) >= max_pixels_per_batch:
                                self._process_em_batch_fixed(
                                    current_batch_neighborhoods,
                                    current_batch_centers,
                                    total_alpha_numerator,
                                    total_alpha_denominator,
                                    total_sigma_numerator,
                                    total_sigma_denominator,
                                    iteration
                                )
                                
                                total_pixels_processed += len(current_batch_neighborhoods)
                                batch_count += 1
                                
                                current_batch_neighborhoods = []
                                current_batch_centers = []
                                
                                if batch_count % 3 == 0:
                                    gc.collect()
            
            # Process remaining batch
            if current_batch_neighborhoods:
                self._process_em_batch_fixed(
                    current_batch_neighborhoods,
                    current_batch_centers,
                    total_alpha_numerator,
                    total_alpha_denominator,
                    total_sigma_numerator,
                    total_sigma_denominator,
                    iteration
                )
                total_pixels_processed += len(current_batch_neighborhoods)
                batch_count += 1
            
            print(f"      Processed {batch_count} batches, {total_pixels_processed:,} pixels")
            
            # Update parameters
            self._update_parameters_safely(total_alpha_numerator, total_alpha_denominator,
                                        total_sigma_numerator, total_sigma_denominator)
            
            # Check convergence
            alpha_change = np.linalg.norm(self.alpha - alpha_old)
            sigma_change = abs(self.sigma - sigma_old)
            iter_time = time.time() - iter_start_time
            
            print(f"   Iteration {iteration + 1:2d}: α change = {alpha_change:.6f}, "
                f"σ change = {sigma_change:.6f}, σ = {self.sigma:.6f}, time = {iter_time:.2f}s")
            
            if alpha_change < self.tolerance and sigma_change < self.tolerance * 0.1:
                print(f"✅ Converged after {iteration + 1} iterations")
                break
        
        p_map = self._generate_final_p_map_fixed(img)
        
        return p_map

    def _process_em_batch_fixed(self, neighborhoods_list, centers_list,
                               total_alpha_numerator, total_alpha_denominator,
                               total_sigma_numerator, total_sigma_denominator,
                               iteration):
        """FIXED batch processing with proper likelihood models"""
        
        neighborhoods = np.array(neighborhoods_list)
        center_values = np.array(centers_list)
        
        # Compute residuals
        residuals = center_values - np.dot(neighborhoods, self.alpha)
        
        # Model M1: Gaussian likelihood (correlated pixels)
        likelihood_M1 = (1.0 / (np.sqrt(2 * np.pi) * self.sigma)) * \
                        np.exp(-0.5 * (residuals / self.sigma) ** 2)
        
        # Model M2: UNIFORM likelihood over signal range (uncorrelated pixels)
        # This is the key fix - M2 should be uniform, not a fraction of M1
        likelihood_M2 = np.ones_like(residuals) / self.signal_range
        
        # Prior probabilities (can be adjusted)
        prior_M1 = 0.7  # Assume most pixels are correlated
        prior_M2 = 0.3  # Some pixels are outliers
        
        # Posterior probabilities using Bayes rule
        numerator_M1 = likelihood_M1 * prior_M1
        numerator_M2 = likelihood_M2 * prior_M2
        denominator = numerator_M1 + numerator_M2 + 1e-10  # Avoid division by zero
        
        posterior_M1 = numerator_M1 / denominator
        posterior_M2 = numerator_M2 / denominator
        
        # Debug: Print statistics for first few iterations
        if iteration < 3:
            print(f"      Batch stats - M1 posterior: mean={np.mean(posterior_M1):.3f}, "
                  f"std={np.std(posterior_M1):.3f}")
            print(f"      Residual stats: mean={np.mean(np.abs(residuals)):.4f}, "
                  f"std={np.std(residuals):.4f}")
        
        # Update alpha (weighted least squares)
        for i in range(len(neighborhoods)):
            weight = posterior_M1[i]
            x = neighborhoods[i]
            y = center_values[i]
            
            total_alpha_denominator += weight * np.outer(x, x)
            total_alpha_numerator += weight * x * y
        
        # Update sigma (weighted variance)
        weighted_residuals_sq = posterior_M1 * (residuals ** 2)
        total_sigma_numerator += np.sum(weighted_residuals_sq)
        total_sigma_denominator += np.sum(posterior_M1)
        
        return len(neighborhoods)

    def _update_parameters_safely(self, alpha_num, alpha_denom, sigma_num, sigma_denom):
        """Update parameters with better numerical stability"""
        
        # Update alpha with regularization
        try:
            # Add regularization to prevent singular matrices
            reg_term = 1e-6 * np.eye(alpha_denom.shape[0])
            regularized_denom = alpha_denom + reg_term
            new_alpha = np.linalg.solve(regularized_denom, alpha_num)

            # Clip alpha to prevent extreme values
            alpha_norm = np.linalg.norm(new_alpha)
            if alpha_norm > 1.0:  # More conservative clipping
                new_alpha = new_alpha * (1.0 / alpha_norm)
                print(f"      ⚠️ Alpha norm clipped: {alpha_norm:.3f} → 1.0")
            
            self.alpha = new_alpha
            
        except np.linalg.LinAlgError:
            print("      ⚠️ Numerical instability, using gradient step")
            if sigma_denom > 0:
                gradient = alpha_num / (sigma_denom + 1e-8)
                learning_rate = 0.1  # Conservative learning rate
                self.alpha = (1 - learning_rate) * self.alpha + learning_rate * gradient
        
        # Update sigma with bounds
        if sigma_denom > 0:
            new_sigma = np.sqrt(sigma_num / sigma_denom)
            # Keep sigma in reasonable bounds
            new_sigma = np.clip(new_sigma, 0.05, 0.5)
            
            # Use momentum for stability
            momentum = 0.8
            self.sigma = momentum * self.sigma + (1 - momentum) * new_sigma

    def _generate_final_p_map_fixed(self, img):
        """Generate final probability map with fixed likelihood calculation"""
        rows, cols = img.shape
        p_map = np.zeros((rows, cols))
        
        padded_img = np.pad(img, self.K, mode='reflect')
        
        print("🎨 Generating FIXED probability map...")
        print(f"   Using σ = {self.sigma:.6f}, signal_range = {self.signal_range:.4f}")
        
        # Fixed M2 likelihood - uniform over signal range
        likelihood_M2_value = 1.0 / self.signal_range
        prior_M1 = 0.7
        prior_M2 = 0.3
        
        print(f"   M2 likelihood value: {likelihood_M2_value:.6f}")
        
        for i in range(self.K, rows - self.K):
            for j in range(self.K, cols - self.K):
                neighborhood = padded_img[i-self.K:i+self.K+1, j-self.K:j+self.K+1]
                neighborhood_flat = neighborhood.flatten()
                center_idx = len(neighborhood_flat) // 2
                
                center_val = neighborhood_flat[center_idx]
                neighbors = np.concatenate([
                    neighborhood_flat[:center_idx],
                    neighborhood_flat[center_idx + 1:]
                ])
                
                residual = center_val - np.dot(neighbors, self.alpha)
                
                # M1: Gaussian likelihood
                likelihood_M1 = (1.0 / (np.sqrt(2 * np.pi) * self.sigma)) * \
                              np.exp(-0.5 * (residual / self.sigma) ** 2)
                
                # M2: Uniform likelihood
                likelihood_M2 = likelihood_M2_value
                
                # Posterior probability
                numerator_M1 = likelihood_M1 * prior_M1
                numerator_M2 = likelihood_M2 * prior_M2
                denominator = numerator_M1 + numerator_M2 + 1e-10
                
                posterior_M1 = numerator_M1 / denominator
                p_map[i, j] = posterior_M1
        
        # Light smoothing to reduce noise
        p_map = gaussian_filter(p_map, sigma=0.5)
        
        print(f"⚡ FIXED p-map completed")
        print(f"   Statistics: mean={np.mean(p_map):.4f}, std={np.std(p_map):.4f}")
        print(f"   Min/Max: [{np.min(p_map):.4f}, {np.max(p_map):.4f}]")
        
        return p_map

    def detect_resampling(self, img):
        """Main detection function with fixed implementation"""
        if isinstance(img, str):
            img = self._load_image(img)
        
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        
        # Normalize to [0,1] range
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        print(f"\n🧠 Processing {img.shape} image with FIXED EM algorithm")
        print(f"🎯 Neighborhood: {self.neighborhood_size}×{self.neighborhood_size}")
        print(f"⚖️ Using PROPER uniform M2 likelihood model")
        
        start_time = time.time()
        p_map = self.memory_efficient_em_algorithm(img)
        total_time = time.time() - start_time
        
        print(f"⏱️ Total processing time: {total_time:.2f} seconds")
        
        spectrum = self._analyze_spectrum(p_map)
        is_resampled = self._detect_peaks(spectrum)
        
        return p_map, spectrum, is_resampled

    def _analyze_spectrum(self, p_map):
        """Analyze frequency spectrum for periodic patterns"""
        # High-pass filter to remove DC component
        p_map_hp = p_map - gaussian_filter(p_map, sigma=3)
        
        # Apply window to reduce edge effects
        rows, cols = p_map_hp.shape
        window_r = np.hanning(rows).reshape(-1, 1)
        window_c = np.hanning(cols).reshape(1, -1)
        window = window_r @ window_c
        p_map_windowed = p_map_hp * window
        
        # Compute FFT
        spectrum = np.abs(fft2(p_map_windowed))
        spectrum = fftshift(spectrum)
        
        # Normalize
        spectrum = spectrum / (np.max(spectrum) + 1e-8)
        
        return spectrum
    
    def _detect_peaks(self, spectrum):
        """Detect periodic peaks indicating resampling"""
        rows, cols = spectrum.shape
        center_r, center_c = rows // 2, cols // 2
        
        # Exclude center region (DC and low frequencies)
        exclude_radius = min(rows, cols) // 8
        mask = np.ones((rows, cols), dtype=bool)
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_c)**2 + (y - center_r)**2)
        mask[distance < exclude_radius] = False
        
        spectrum_masked = spectrum[mask]
        
        if len(spectrum_masked) == 0:
            return False
        
        # Adaptive threshold based on distribution
        sorted_vals = np.sort(spectrum_masked)
        percentile_95 = sorted_vals[int(0.95 * len(sorted_vals))]
        percentile_99 = sorted_vals[int(0.99 * len(sorted_vals))]
        
        threshold = percentile_95 * 1.1
        strong_threshold = percentile_99 * 0.8
        
        num_peaks = np.sum(spectrum_masked > threshold)
        strong_peaks = np.sum(spectrum_masked > strong_threshold)
        peak_ratio = num_peaks / len(spectrum_masked)
        max_peak = np.max(spectrum_masked)
        
        # More conservative detection criteria
        is_resampled = (
            (num_peaks > 8) and 
            (peak_ratio > 0.005) and 
            (max_peak > 0.5) and
            (strong_peaks >= 1)
        )
        
        print(f"🔍 Peak detection: {num_peaks} peaks, "
              f"ratio: {peak_ratio:.4f}, max: {max_peak:.4f}, "
              f"strong peaks: {strong_peaks}")
        
        return is_resampled
    
    def _get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage_mb = memory_mb
        return memory_mb
    
    def _load_image(self, image_path):
        """Load image from file"""
        img = Image.open(image_path).convert('L')
        return np.array(img) / 255.0
    
    def test_single_image(self, image_path_or_array, max_size):
        """Test the fixed algorithm on a single image"""
        print("🚀 Starting FIXED EM Resampling Detection...")
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
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Input image
        axes[0, 0].imshow(display_img, cmap='gray')
        axes[0, 0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Probability map
        im1 = axes[0, 1].imshow(p_map, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('FIXED EM Probability Map\n(Red = High correlation)', fontsize=14)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
        
        # Spectrum
        im2 = axes[0, 2].imshow(np.log(spectrum + 1e-6), cmap='viridis')
        axes[0, 2].set_title('Log Frequency Spectrum\n(Peaks = Periodic patterns)', fontsize=14)
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
        
        # Results
        result_color = 'red' if is_resampled else 'green'
        result_text = '✓ RESAMPLING DETECTED' if is_resampled else '✗ NO RESAMPLING'
        
        axes[1, 0].text(0.5, 0.7, result_text, ha='center', va='center', 
                        fontsize=16, fontweight='bold', color=result_color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                 edgecolor=result_color, linewidth=2))
        
        param_text = f"""FIXED EM Results:
Neighborhood: {self.neighborhood_size}×{self.neighborhood_size}
Signal Range: {self.signal_range:.4f}
Final σ: {self.sigma:.4f}
α weights: {len(self.alpha)}
Max |α|: {np.max(np.abs(self.alpha)):.4f}

Performance:
Total Time: {total_processing_time:.2f}s
Pixels Processed: {self.total_pixels_processed:,}
Peak Memory: {final_memory:.1f} MB"""
        
        axes[1, 0].text(0.5, 0.3, param_text, ha='center', va='center',
                        fontsize=9, fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        axes[1, 0].axis('off')
        axes[1, 0].set_title('FIXED Results & Performance', fontsize=14, fontweight='bold')
        
        # Probability histogram
        axes[1, 1].hist(p_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].set_title('FIXED Probability Distribution', fontsize=14)
        axes[1, 1].set_xlabel('P(Correlated | pixel)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(np.mean(p_map), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(p_map):.3f}')
        axes[1, 1].legend()
        
        # Alpha weights visualization
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
            axes[1, 2].set_title('FIXED α Weights\n(Blue=negative, Red=positive)', fontsize=14)
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
        plt.suptitle('FIXED Memory-Efficient EM Algorithm - Resampling Detection', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        fig_filename = f'results/fixed_em_detection_results_{IMAGE_NAME}.png'
        plt.draw()
        try:
            plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Figure saved as '{fig_filename}'")
        except Exception as e:
            print(f"⚠️ Could not save figure: {e}")
        
        plt.show()
        
        print("\n" + "="*70)
        print("FIXED EM DETECTION RESULTS")
        print("="*70)
        print(f"Resampling Detected: {is_resampled}")
        print(f"Image Shape: {display_img.shape}")
        print(f"Total Processing Time: {total_processing_time:.2f} seconds")
        print(f"Total Pixels Processed: {self.total_pixels_processed:,}")
        print(f"Average Probability: {np.mean(p_map):.4f}")
        print(f"Probability Std: {np.std(p_map):.4f}")
        print(f"Probability Range: [{np.min(p_map):.4f}, {np.max(p_map):.4f}]")
        print("\n EM Algorithm Results:")
        print(f"Signal Range: {self.signal_range:.4f}")
        print(f"Final σ (noise std): {self.sigma:.4f}")
        print(f"Number of α weights: {len(self.alpha)}")
        print(f"Max |α| weight: {np.max(np.abs(self.alpha)):.4f}")
        print(f"α weight range: [{np.min(self.alpha):.4f}, {np.max(self.alpha):.4f}]")
        print("\nPerformance:")
        print(f"Chunk Size: {self.chunk_size}")
        print(f"Memory Limit: {self.max_memory_mb} MB")
        print(f"Peak Memory Usage: {final_memory:.1f} MB")
        print("="*70)

def run_em_demo():
    print("🚀 Memory-Efficient Full EM Algorithm Demo")
    print("="*70)
    
    detector = MemoryEfficientEMDetector(
        neighborhood_size=5,
        max_iterations=16,
        tolerance=1e-4,
        chunk_size=2048 * 16,
        max_memory_mb=2048
    )
    
    try:
        print("\n📷 Testing processing on external image")
        detector.test_single_image(f'img/{IMAGE_NAME}.jpg', max_size=400)
    except Exception as e:
        print(f"External image test failed: {e}")

if __name__ == "__main__":
    run_em_demo()