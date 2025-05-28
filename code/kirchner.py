import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
from skimage.transform import resize, rotate
from skimage import data

# Fast and Reliable Resampling Detection by Spectral Analysis of Fixed Linear Predictor Residue
# by Matthias Kirchner

class FastResamplingDetector:
    def __init__(self, lambda_param=1.0, tau=2.0, sigma=1.0):

        self.lambda_param = lambda_param
        self.tau = tau
        self.sigma = sigma
        
        # Fixed predictor coefficients (second-derivative like)
        # Based on Kirchner's optimal coefficients
        self.predictor = np.array([
            [-0.25,  0.50, -0.25],
            [ 0.50,  0.00,  0.50],
            [-0.25,  0.50, -0.25]
        ])
    
    def detect_resampling(self, img):

        if isinstance(img, str):
            img = self._load_image(img)
        
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        prediction_error = self._calculate_prediction_error(img)
        
        p_map = self._generate_p_map(prediction_error)
        
        spectrum = self._analyze_spectrum(p_map)
        is_resampled = self._detect_peaks(spectrum)
        
        return p_map, spectrum, is_resampled, prediction_error
    
    def _calculate_prediction_error(self, img):

        rows, cols = img.shape
        
        prediction_error = cv2.filter2D(img, -1, self.predictor, borderType=cv2.BORDER_REFLECT)
        
        return prediction_error
    
    def _generate_p_map(self, prediction_error):
        abs_error = np.abs(prediction_error)
        
        p_map = self.lambda_param * np.exp(-(abs_error ** self.tau) / self.sigma)
        
        p_map = (p_map - p_map.min()) / (p_map.max() - p_map.min() + 1e-8)
        
        return p_map
    
    def _analyze_spectrum(self, p_map):
        p_map_centered = p_map - np.mean(p_map)
        
        rows, cols = p_map_centered.shape
        window_r = np.hanning(rows).reshape(-1, 1)
        window_c = np.hanning(cols).reshape(1, -1)
        window = window_r @ window_c
        p_map_windowed = p_map_centered * window
        
        spectrum = np.abs(fft2(p_map_windowed))
        spectrum = fftshift(spectrum)
        
        spectrum = spectrum / (np.max(spectrum) + 1e-8)
        
        return spectrum
    
    def _detect_peaks(self, spectrum):
        rows, cols = spectrum.shape
        center_r, center_c = rows // 2, cols // 2
        
        exclude_radius = min(rows, cols) // 10
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_c)**2 + (y - center_r)**2)
        mask = distance >= exclude_radius
        
        first_quadrant = spectrum[center_r:, center_c:]
        first_quadrant_mask = mask[center_r:, center_c:]
        
        spectrum_values = first_quadrant[first_quadrant_mask]
        spectrum_sorted = np.sort(spectrum_values)
        
        cumulative_sum = np.cumsum(spectrum_sorted)
        cumulative_sum = cumulative_sum / cumulative_sum[-1]  
        
        if len(cumulative_sum) > 1:
            gradients = np.diff(cumulative_sum)
            max_gradient = np.max(gradients)
        else:
            max_gradient = 0
        
        mean_spectrum = np.mean(spectrum_values)
        std_spectrum = np.std(spectrum_values)
        max_peak = np.max(spectrum_values)
        
        threshold = mean_spectrum + 3 * std_spectrum
        num_peaks = np.sum(spectrum_values > threshold)
        peak_ratio = num_peaks / len(spectrum_values) if len(spectrum_values) > 0 else 0
        
        is_resampled = (max_gradient > 0.02) or (peak_ratio > 0.005 and max_peak > 0.15)
        
        print(f"Peak detection: max_gradient={max_gradient:.4f}, peak_ratio={peak_ratio:.4f}, max_peak={max_peak:.4f}")
        
        return is_resampled
    
    def _load_image(self, image_path):
        img = Image.open(image_path).convert('L')
        return np.array(img) / 255.0
    
    def test_single_image(self, image_path_or_array):
        print("🔍 Starting Fast Resampling Detection...")
        print("="*60)
        
        p_map, spectrum, is_resampled, prediction_error = self.detect_resampling(image_path_or_array)
        
        if isinstance(image_path_or_array, str):
            display_img = self._load_image(image_path_or_array)
        else:
            display_img = image_path_or_array
            if len(display_img.shape) == 3:
                display_img = np.mean(display_img, axis=2)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(display_img, cmap='gray')
        axes[0, 0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Prediction error
        im0 = axes[0, 1].imshow(prediction_error, cmap='RdBu_r', 
                               vmin=np.percentile(prediction_error, 5),
                               vmax=np.percentile(prediction_error, 95))
        axes[0, 1].set_title('Prediction Error\n(Blue=negative, Red=positive)', fontsize=14)
        axes[0, 1].axis('off')
        plt.colorbar(im0, ax=axes[0, 1], shrink=0.8)
        
        im1 = axes[0, 2].imshow(p_map, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title('P-map\n(Red = High correlation prob.)', fontsize=14)
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2], shrink=0.8)
        
        im2 = axes[1, 0].imshow(np.log(spectrum + 1e-6), cmap='viridis')
        axes[1, 0].set_title('Log Frequency Spectrum\n(Peaks = Periodic patterns)', fontsize=14)
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)
        
        result_color = 'red' if is_resampled else 'green'
        result_text = '✓ RESAMPLING DETECTED' if is_resampled else '✗ NO RESAMPLING'
        
        axes[1, 1].text(0.5, 0.7, result_text, ha='center', va='center', 
                        fontsize=16, fontweight='bold', color=result_color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                 edgecolor=result_color, linewidth=2))
        
        param_text = f"""Algorithm: Kirchner Fast Detection
Predictor: Fixed 3×3 kernel
λ: {self.lambda_param}
τ: {self.tau}
σ: {self.sigma}
Computational Time: ~40× faster than EM"""
        
        axes[1, 1].text(0.5, 0.3, param_text, ha='center', va='center',
                        fontsize=10, fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Detection Result & Parameters', fontsize=14, fontweight='bold')
        
        axes[1, 2].hist(prediction_error.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 2].set_title('Prediction Error Distribution', fontsize=14)
        axes[1, 2].set_xlabel('Prediction Error')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        error_mean = np.mean(prediction_error)
        error_std = np.std(prediction_error)
        axes[1, 2].axvline(error_mean, color='red', linestyle='--', 
                          label=f'Mean: {error_mean:.4f}')
        axes[1, 2].axvline(error_mean + 2*error_std, color='orange', linestyle='--', 
                          label=f'+2σ: {error_mean + 2*error_std:.4f}')
        axes[1, 2].axvline(error_mean - 2*error_std, color='orange', linestyle='--', 
                          label=f'-2σ: {error_mean - 2*error_std:.4f}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.suptitle('Fast Resampling Detection (Kirchner Method)', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.show()
        
        print("\n" + "="*60)
        print("FAST RESAMPLING DETECTION RESULTS")
        print("="*60)
        print(f"Resampling Detected: {is_resampled}")
        print(f"Image Shape: {display_img.shape}")
        print(f"Prediction Error Stats:")
        print(f"  Mean: {np.mean(prediction_error):.6f}")
        print(f"  Std:  {np.std(prediction_error):.6f}")
        print(f"  Min:  {np.min(prediction_error):.6f}")
        print(f"  Max:  {np.max(prediction_error):.6f}")
        print(f"P-map Stats:")
        print(f"  Mean: {np.mean(p_map):.4f}")
        print(f"  Std:  {np.std(p_map):.4f}")
        print(f"  Min:  {np.min(p_map):.4f}")
        print(f"  Max:  {np.max(p_map):.4f}")
        print(f"Spectrum Stats:")
        print(f"  Max:  {np.max(spectrum):.4f}")
        print(f"  Mean: {np.mean(spectrum):.4f}")
        print("="*60)

def run_demo():
    print("🚀 Fast Resampling Detection Demo")
    print("="*60)
    
    detector = FastResamplingDetector(
        lambda_param=1.0,
        tau=2.0,
        sigma=1.0
    )
    
    try:
        print("\n🔍 Testing External Image...")
        detector.test_single_image('img/image_1.jpg')
    except Exception as e:
        print(f"External image test failed: {e}")

if __name__ == "__main__":
    run_demo()