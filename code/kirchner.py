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

PRINT_OUTPUT = False

class KirchnerDetector:
    def __init__(self, sensitivity='medium', lambda_param=1.0, tau=2.0, sigma=1.0, downscale_size=512, downscale=True, max_gradient=None):
        # Preset filter coefficients from Equation 25
        self.predictor_filter = np.array([
            [-0.25, 0.50, -0.25],
            [0.50,  0.00,  0.50],
            [-0.25, 0.50, -0.25]
        ], dtype=np.float32)
        
        # Optimized Sobel operators for gradient computation (Section 5.2.2)
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        self.lambda_param = lambda_param
        self.tau = tau
        self.sigma = sigma
        self.file_handler = FileHandler(downscale_size, downscale)
        
        if max_gradient is not None:
            self.gradient_threshold = max_gradient
            self.custom_threshold = True
        else:
            sensitivity_params = {
                'low':    {'gradient_threshold': 0.05}, 
                'medium': {'gradient_threshold': 0.11},  
                'high':   {'gradient_threshold': 0.30}    
            }
            params = sensitivity_params.get(sensitivity, sensitivity_params['medium'])
            self.gradient_threshold = params['gradient_threshold']
            self.custom_threshold = False

    def detect(self, img_input, skip_internal_downscale=False, save_intermediate_steps=False):
        try:
            if isinstance(img_input, (str, Path)):
                tqdm.write(f"      Loading image: {Path(img_input).name}")
                image = self.file_handler.load_image(img_input, apply_downscale=not skip_internal_downscale)
            else:
                image = img_input.astype(np.float32)
                    
            if PRINT_OUTPUT:
                tqdm.write(f"      Running Kirchner detection...")
                if self.custom_threshold:
                    tqdm.write(f"      Using custom threshold: {self.gradient_threshold:.8f}")
                else:
                    tqdm.write(f"      Using default threshold: {self.gradient_threshold:.8f}")
                    
            results = self.detect_resampling(image, save_intermediate_steps)
            detected = results['detected']
                    
            if PRINT_OUTPUT:
                tqdm.write(f"      Detection result: {'DETECTED' if detected else 'CLEAN'}")

            return {
                'detected': results['detected'],
                'p_map': results['p_map'],
                'spectrum': results['spectrum'],
                'prediction_error': results['prediction_error'],
                'max_gradient': results['max_gradient'],
                'gradient_map': results['gradient_map'] 
            }
        except Exception as e:
            tqdm.write(f"      ERROR in detect(): {e}")
            raise

    def detect_resampling(self, image, save_intermediate_steps=False):
        if PRINT_OUTPUT:
            tqdm.write(f"        Step 1: Input preparation")
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)

        if PRINT_OUTPUT:
            tqdm.write(f"        Step 2: Linear prediction with preset coefficients (Equation 25)")
        predicted = cv2.filter2D(image, -1, self.predictor_filter, borderType=cv2.BORDER_REFLECT)
        
        if PRINT_OUTPUT:
            tqdm.write(f"        Step 3: Prediction error computation")
        prediction_error = image - predicted
             
        if PRINT_OUTPUT:   
            tqdm.write(f"        Step 4: P-map generation (Equation 21: p = lambda * exp(-|e|^tau / sigma))")
        p_map = self.generate_p_map(prediction_error)
                
        if PRINT_OUTPUT:
            tqdm.write(f"        Step 5: Spectrum computation with contrast function gamma")
        spectrum = self.compute_spectrum(p_map)
                
        if PRINT_OUTPUT:
            tqdm.write(f"        Step 6: Cumulative periodogram detection (Equation 23-24)")
        detected, max_gradient, gradient_map = self.detect_cumulative_periodogram(spectrum)
                
        if PRINT_OUTPUT:
            threshold_type = "custom" if self.custom_threshold else "default"
            tqdm.write(f"          Max gradient: {max_gradient:.6f}, Threshold: {self.gradient_threshold:.6f} ({threshold_type})")        
        
        if save_intermediate_steps:
            self.save_intermediate_results(image, predicted, prediction_error, 
                                         p_map, spectrum, gradient_map)
        return {
            'p_map': p_map,
            'spectrum': spectrum,
            'prediction_error': prediction_error,
            'detected': detected,
            'max_gradient': max_gradient,
            'gradient_map': gradient_map
        }

    def generate_p_map(self, prediction_error):
        # Paper equation 21: p = lambda * exp(-|e|^tau / sigma)
        abs_error = np.abs(prediction_error)
        np.power(abs_error, self.tau, out=abs_error)
        abs_error /= self.sigma
        np.negative(abs_error, out=abs_error)
        np.exp(abs_error, out=abs_error)
        abs_error *= self.lambda_param
        return abs_error

    def compute_spectrum(self, p_map, gamma=0.8):
        # Remove DC component
        p_map_zero_mean = p_map - np.mean(p_map)
        
        # FFT computation
        fft_result = fft2(p_map_zero_mean)
        magnitude_spectrum = np.abs(fftshift(fft_result))
        
        # Optimized radial weighting using ogrid
        h, w = p_map.shape
        center_h, center_w = h // 2, w // 2
        
        # Create radial distance map 
        y, x = np.ogrid[-center_h:h-center_h, -center_w:w-center_w]
        radial_dist = np.sqrt(y*y + x*x)
        max_radius = np.max(radial_dist)
        
        if max_radius > 0:
            radial_weight = radial_dist / max_radius
            weighted_spectrum = magnitude_spectrum * radial_weight
        else:
            weighted_spectrum = magnitude_spectrum
        
        return np.power(weighted_spectrum, gamma)

    def detect_cumulative_periodogram(self, spectrum):
        # Remove DC component
        spectrum = spectrum.copy()
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        spectrum[center_h, center_w] = 0
        
        # First quadrant of a p-map's DFT (0 <= f <= b) 
        first_quadrant = spectrum[center_h:, center_w:]
        
        # Equation 23: C(f) = sum|P(f')|^2 / sum_total|P(f')|^2
        energy = first_quadrant ** 2
        total_energy = np.sum(energy)
        
        if total_energy == 0:
            return False, 0.0, np.zeros_like(first_quadrant)
        
        # Fast cumulative computation using numpy
        cumulative_matrix = np.cumsum(np.cumsum(energy, axis=0), axis=1)
        C_matrix = cumulative_matrix / total_energy
        
        # Equation 24: delta' = max |grad C(f)| using Sobel operator (Paper Section 5.2.2)
        grad_x = cv2.filter2D(C_matrix, -1, self.sobel_x)
        grad_y = cv2.filter2D(C_matrix, -1, self.sobel_y)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        max_gradient = np.max(gradient_magnitude)
        detected = max_gradient > self.gradient_threshold
        
        return detected, max_gradient, gradient_magnitude

    def save_intermediate_results(self, image, predicted, prediction_error, 
                                p_map, spectrum, gradient_map):
        self.file_handler.save_presentation_image(image, 'demo/presentation', 'image.png', 'standard')
        self.file_handler.save_presentation_image(predicted, 'demo/presentation', 'predicted.png', 'standard')
        self.file_handler.save_presentation_image(prediction_error, 'demo/presentation', 'prediction_error.png', 'prediction_error')
        self.file_handler.save_presentation_image(p_map, 'demo/presentation', 'p_map.png', 'p_map')
        self.file_handler.save_presentation_image(spectrum, 'demo/presentation', 'spectrum.png', 'spectrum')
        self.file_handler.save_presentation_image(gradient_map, 'demo/presentation', 'gradient_map.png', 'gradient')

    def extract_detection_metrics(self, spectrum):
        gradient_detected, max_gradient, gradient_map = self.detect_cumulative_periodogram(spectrum)
        
        return {
            'max_gradient': max_gradient,
            'gradient_map': gradient_map,
            'spectrum_mean': np.mean(spectrum),
            'spectrum_std': np.std(spectrum),
            'spectrum_max': np.max(spectrum),
            'gradient_method_detected': gradient_detected,
            'gradient_threshold': self.gradient_threshold  # Always return the threshold used
        }