"""
Kirchner fast resampling detector implementation
Based on: "Fast and reliable resampling detection by spectral analysis 
of fixed linear predictor residue" (2008) - Section 5 Fast Detection

Key Implementation:
- Section 5.1: Fast computation with preset coefficients (Equation 25)
- Section 5.2.2: Cumulative periodogram detection (Equation 24)
- Equation 21: P-map formula p = lambda * exp(-|e|^tau / sigma)
"""
import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.ndimage import convolve
from pathlib import Path
from fileHandler import FileHandler


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
            'low':    {'gradient_threshold': 0.0040}, 
            'medium': {'gradient_threshold': 0.008},  
            'high':   {'gradient_threshold': 0.010}    
        }

        params = sensitivity_params.get(sensitivity, sensitivity_params['medium'])
        self.gradient_threshold = params['gradient_threshold']
        self.sensitivity = sensitivity

    def detect(self, img_input, skip_internal_downscale=False):
        try:
            if isinstance(img_input, (str, Path)):
                print(f"      Loading image: {Path(img_input).name}")
                image = self.file_handler.load_image(img_input, apply_downscale=not skip_internal_downscale)
                print(f"      Image loaded, size: {image.shape}")
            else:
                image = img_input.astype(np.float32)
                print(f"      Using pre-loaded image, size: {image.shape}")
            
            print(f"      Running Kirchner detection...")
            results = self.detect_resampling_fast(image)
            detected = results['detected']
            print(f"      Detection result: {'DETECTED' if detected else 'CLEAN'}")

            
            return {
                'detected': results['detected'],
                'p_map': results['p_map'],
                'spectrum': results['spectrum'],
                'prediction_error': results['prediction_error']
            }
        except Exception as e:
            print(f"      ERROR in detect(): {e}")
            raise

    def detect_resampling_fast(self, image):
        print(f"        Step 1: Input preparation")
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        
        # Apply preset linear predictor
        print(f"        Step 2: Linear prediction")
        predicted = convolve(image, self.predictor_filter, mode='reflect')
        
        # Calculate prediction error
        print(f"        Step 3: Prediction error")
        prediction_error = image - predicted
        
        # Generate P-map using Equation 21
        print(f"        Step 4: P-map generation")
        p_map = self.generate_p_map_fast(prediction_error)
        
        # Spectral analysis
        print(f"        Step 5: Spectrum computation")
        spectrum = self.compute_spectrum_simple(p_map)
        
        # Simple detection decision
        print(f"        Step 6: Detection decision")
        detected = self.make_decision_simple(spectrum)
        
        return {
            'p_map': p_map,
            'spectrum': spectrum,
            'prediction_error': prediction_error,
            'detected': detected
        }

    def generate_p_map_fast(self, prediction_error):
        """P-map generation using Equation 21."""
        abs_error = np.abs(prediction_error)
        # Clip to avoid overflow
        abs_error = np.clip(abs_error, 0, 10)
        p_map = self.lambda_param * np.exp(-(abs_error ** self.tau) / self.sigma)
        return p_map

    def compute_spectrum_simple(self, p_map):
        # Contrast function Γ: subtract local mean
        kernel = np.ones((5, 5), dtype=np.float32) / 25
        local_mean = convolve(p_map, kernel, mode='reflect')
        contrast_p_map = p_map - local_mean
        
        fft_result = fft2(contrast_p_map)
        spectrum = np.abs(fftshift(fft_result))
        
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
        
        return spectrum

    def make_decision_simple(self, spectrum):
        gradient_detected, max_gradient, gradient_map = self.detect_cumulative_periodogram(spectrum)
        print(f"          Gradient: {max_gradient:.6f}")
        return gradient_detected

    def detect_cumulative_periodogram(self, spectrum):
        # Remove DC component
        spectrum = spectrum.copy()
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        spectrum[center_h, center_w] = 0

        energy = spectrum ** 2
        total_energy = np.sum(energy)

        if total_energy == 0:
            return False, 0.0

        # Compute cumulative sum in both directions (upper-left quadrant)
        cumulative_energy = np.cumsum(np.cumsum(energy, axis=0), axis=1)
        
        # Normalize to form cumulative periodogram
        C = cumulative_energy / total_energy

        # Compute gradient magnitude of the cumulative periodogram
        grad_y, grad_x = np.gradient(C)
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

    def get_detector_info(self):
        return {
            'method': 'robust_fast',
            'sensitivity': self.sensitivity,
            'lambda_param': self.lambda_param,
            'tau': self.tau,
            'sigma': self.sigma,
            'gradient_threshold': self.gradient_threshold,
            'predictor_filter': self.predictor_filter.tolist()
        }