"""
Kirchner fast resampling detector implementation
Based on: "Fast and reliable resampling detection by spectral analysis of fixed linear predictor residue" (2008)
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.ndimage import convolve
from pathlib import Path
from fileHandler import FileHandler
from tqdm import tqdm

class KirchnerDetector:
    def __init__(self, sensitivity='medium', lambda_param=1.0, tau=2.0, sigma=1.0, downscale_size=512, downscale=True):
        # Preset filter coefficients from Equation 25
        self.predictor_filter = np.array([
            [-0.25, 0.50, -0.25],
            [0.50,  0.00,  0.50],
            [-0.25, 0.50, -0.25]
        ])
        
        # P-map parameters from Equation 21
        self.lambda_param = lambda_param
        self.tau = tau  
        self.sigma = sigma

        self.file_handler = FileHandler(downscale_size, downscale)
        
        sensitivity_params = {
            'low':    {'gradient_threshold': 0.010}, 
            'medium': {'gradient_threshold': 0.015},  
            'high':   {'gradient_threshold': 0.030}    
        }

        params = sensitivity_params.get(sensitivity, sensitivity_params['medium'])
        self.gradient_threshold = params['gradient_threshold']
        self.sensitivity = sensitivity

    def detect(self, img_input, skip_internal_downscale=False):
        try:
            if isinstance(img_input, (str, Path)):
                tqdm.write(f"      Loading image: {Path(img_input).name}")
                image = self.file_handler.load_image(img_input, apply_downscale=not skip_internal_downscale)
                tqdm.write(f"      Image loaded, size: {image.shape}")
            else:
                image = img_input.astype(np.float32)
                tqdm.write(f"      Using pre-loaded image, size: {image.shape}")
            
            tqdm.write(f"      Running Kirchner detection...")
            results = self.detect_resampling(image)
            detected = results['detected']
            tqdm.write(f"      Detection result: {'DETECTED' if detected else 'CLEAN'}")

            return {
                'detected': results['detected'],
                'p_map': results['p_map'],
                'spectrum': results['spectrum'],
                'prediction_error': results['prediction_error']
            }
        except Exception as e:
            tqdm.write(f"      ERROR in detect(): {e}")
            raise

    def detect_resampling(self, image):
        tqdm.write(f"        Step 1: Input preparation")
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        
        tqdm.write(f"        Step 2: Linear prediction with preset coefficients (Equation 25)")
        predicted = convolve(image, self.predictor_filter, mode='reflect')
        
        tqdm.write(f"        Step 3: Prediction error computation")
        prediction_error = image - predicted
        
        tqdm.write(f"        Step 4: P-map generation (Equation 21: p = lambda * exp(-|e|^tau / sigma))")
        p_map = self.generate_p_map(prediction_error)
        
        tqdm.write(f"        Step 5: Spectrum computation with contrast function gamma")
        spectrum = self.compute_spectrum(p_map)
        
        tqdm.write(f"        Step 6: Cumulative periodogram detection (Equation 23-24)")
        detected, max_gradient, gradient_map = self.detect_cumulative_periodogram(spectrum)
        
        tqdm.write(f"          Max gradient: {max_gradient:.6f}, Threshold: {self.gradient_threshold:.6f}")
        
        return {
            'p_map': p_map,
            'spectrum': spectrum,
            'prediction_error': prediction_error,
            'detected': detected,
            'max_gradient': max_gradient,
            'gradient_map': gradient_map
        }

    def generate_p_map(self, prediction_error):
        abs_error = np.abs(prediction_error)
        p_map = self.lambda_param * np.exp(-(abs_error ** self.tau) / self.sigma)
        return p_map

    def compute_spectrum(self, p_map, gamma=0.8):
        p_map_zero_mean = p_map - np.mean(p_map)
        
        # DFT computation
        fft_result = fft2(p_map_zero_mean)
        magnitude_spectrum = np.abs(fftshift(fft_result))
        
        # Contrast function: radial weighting window (attenuates low frequencies)
        h, w = p_map.shape
        center_h, center_w = h // 2, w // 2
        y_coords, x_coords = np.mgrid[-center_h:h-center_h, -center_w:w-center_w]
        radial_dist = np.sqrt(y_coords**2 + x_coords**2)
        max_radius = np.max(radial_dist)
        
        # Radial weighting (higher weights for higher frequencies)
        if max_radius > 0:
            radial_weight = radial_dist / max_radius
        else:
            radial_weight = np.ones_like(radial_dist)
        
        weighted_spectrum = magnitude_spectrum * radial_weight
        gamma_corrected = weighted_spectrum ** gamma
        
        return gamma_corrected

    def detect_cumulative_periodogram(self, spectrum):
        # Remove DC component at center
        spectrum = spectrum.copy()
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        spectrum[center_h, center_w] = 0
        
        #First quadrant of a p-map's DFT (0 <= f <= b)"
        first_quadrant = spectrum[center_h:, center_w:]
        
        # Equation 23: C(f) = sum|P(f')|^2 / sum_total|P(f')|^2
        energy = first_quadrant ** 2
        total_energy = np.sum(energy)
        
        if total_energy == 0:
            return False, 0.0
        
        cumulative_matrix = np.zeros_like(energy)
        for i in range(energy.shape[0]):
            for j in range(energy.shape[1]):
                cumulative_matrix[i, j] = np.sum(energy[:i+1, :j+1])
        
        C_matrix = cumulative_matrix / total_energy
        
        # Equation 24: delta' = max |delta C(f)|
        grad_y, grad_x = np.gradient(C_matrix)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        max_gradient = np.max(gradient_magnitude)
        
        detected = max_gradient > self.gradient_threshold
        
        return detected, max_gradient, gradient_magnitude

    def extract_detection_metrics(self, spectrum):
        gradient_detected, max_gradient, gradient_map = self.detect_cumulative_periodogram(spectrum)
        
        return {
            'max_gradient': max_gradient,
            'gradient_map': gradient_map,
            'spectrum_mean': np.mean(spectrum),
            'spectrum_std': np.std(spectrum),
            'spectrum_max': np.max(spectrum),
            'gradient_method_detected': gradient_detected,
            'gradient_threshold': self.gradient_threshold
        }