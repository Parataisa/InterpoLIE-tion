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
            'low':    {'gradient_threshold': 0.004}, 
            'medium': {'gradient_threshold': 0.009},  
            'high':   {'gradient_threshold': 0.015}    
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
        tqdm.write(f"          Detection result: {'DETECTED' if detected else 'CLEAN'}")
        
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

    def compute_spectrum(self, p_map):
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
        local_mean = convolve(p_map, kernel, mode='reflect')
        contrast_p_map = p_map - local_mean
        
        contrast_p_map = np.sign(contrast_p_map) * np.abs(contrast_p_map) ** 0.8
        contrast_p_map = contrast_p_map - np.mean(contrast_p_map)
        
        h, w = contrast_p_map.shape
        window_h = np.hanning(h).reshape(-1, 1)
        window_w = np.hanning(w).reshape(1, -1)
        window = window_h * window_w
        windowed_p_map = contrast_p_map * window
        
        fft_result = fft2(windowed_p_map)
        spectrum = np.abs(fftshift(fft_result))
        
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
        
        return spectrum

    def detect_cumulative_periodogram(self, spectrum):
        spectrum = spectrum.copy()
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        spectrum[center_h, center_w] = 0
        
        first_quadrant = spectrum[center_h:, center_w:]
        
        energy = first_quadrant ** 2
        total_energy = np.sum(energy)
        
        if total_energy == 0:
            return False, 0.0, np.zeros_like(first_quadrant)
        
        h_quad, w_quad = first_quadrant.shape
        y_coords, x_coords = np.meshgrid(np.arange(h_quad), np.arange(w_quad), indexing='ij')
        
        radial_dist = np.sqrt(x_coords**2 + y_coords**2)
        max_dist = np.sqrt((h_quad-1)**2 + (w_quad-1)**2)
        radial_dist_norm = radial_dist / max_dist
        
        flat_energy = energy.flatten()
        flat_radial = radial_dist_norm.flatten()
        
        sort_indices = np.argsort(flat_radial)
        sorted_energy = flat_energy[sort_indices]
        
        cumulative_energy = np.cumsum(sorted_energy)
        C_values = cumulative_energy / total_energy
        
        C_2d = np.zeros_like(energy)
        for i, idx in enumerate(sort_indices):
            row, col = idx // w_quad, idx % w_quad
            C_2d[row, col] = C_values[i]
        
        grad_y, grad_x = np.gradient(C_2d)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        max_grad = np.max(gradient_magnitude)
        
        detected = max_grad > self.gradient_threshold
        return detected, max_grad, gradient_magnitude

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