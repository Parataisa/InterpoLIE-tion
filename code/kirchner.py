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
from PIL import Image
from pathlib import Path


class KirchnerDetector:
    def __init__(self, sensitivity='medium', lambda_param=1.0, tau=2.0, sigma=1.0):
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
        
        sensitivity_params = {
            'low':      {'peak_threshold_T': 5.0,  'gradient_threshold': 0.030, 'min_peaks': 5}, 
            'medium':   {'peak_threshold_T': 4.0,  'gradient_threshold': 0.025, 'min_peaks': 3},  
            'high':     {'peak_threshold_T': 3.0,  'gradient_threshold': 0.020, 'min_peaks': 2}    
        }
            
        params = sensitivity_params.get(sensitivity, sensitivity_params['medium'])
        self.peak_threshold_T = params['peak_threshold_T']  
        self.gradient_threshold = params['gradient_threshold']
        self.min_peaks = params['min_peaks']
        self.sensitivity = sensitivity

    def detect(self, img_path):
        try:
            print(f"      Loading image: {Path(img_path).name}")
            image = self.load_image(img_path)
            print(f"      Image loaded, size: {image.shape}")
            
            print(f"      Running Kirchner detection...")
            results = self.detect_resampling_fast(image)
            print(f"      Detection result: {'DETECTED' if results['detected'] else 'CLEAN'}")
            
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
        image = image.astype(np.float64)
        
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
        # Apply 2D FFT and shift zero frequency to center
        fft_result = fft2(p_map)
        spectrum = np.abs(fftshift(fft_result))
        
        # Simple normalization instead of complex contrast function
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
        
        return spectrum

    def make_decision_simple(self, spectrum):
        # Method 1: Simple peak counting
        peaks = self.detect_peaks_fast(spectrum)
        peak_detected = len(peaks) >= self.min_peaks
        
        # Method 2: Simple gradient analysis
        gradient_detected, max_gradient = self.detect_cumulative_periodogram(spectrum)
        
        final_detected = peak_detected or gradient_detected
        
        print(f"          Peaks found: {len(peaks)}, Gradient: {max_gradient:.6f}")
        print(f"          Peak detection: {peak_detected}, Gradient detection: {gradient_detected}")
        
        return final_detected

    def detect_peaks_fast(self, spectrum):
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Define a reasonable search area
        search_size = min(h, w) // 4
        y_start = max(center_h - search_size, 5)
        y_end = min(center_h + search_size, h - 5)
        x_start = max(center_w - search_size, 5)
        x_end = min(center_w + search_size, w - 5)
        
        search_region = spectrum[y_start:y_end, x_start:x_end]
        
        if search_region.size == 0:
            return []
        
        threshold = np.mean(search_region) + 2 * np.std(search_region)
        
        peak_mask = search_region > threshold
        peak_locations = np.where(peak_mask)
        
        peaks = []
        max_peaks = 50  
        
        for i in range(min(len(peak_locations[0]), max_peaks)):
            y_local = peak_locations[0][i]
            x_local = peak_locations[1][i]
            
            # Convert back to global coordinates
            y_global = y_start + y_local
            x_global = x_start + x_local
            
            # Skip if too close to DC component
            if abs(y_global - center_h) < 3 and abs(x_global - center_w) < 3:
                continue
            
            strength = spectrum[y_global, x_global]
            
            # Simple local maximum check
            local_area = spectrum[max(0, y_global-1):min(h, y_global+2),
                               max(0, x_global-1):min(w, x_global+2)]
            
            if strength >= np.max(local_area):
                local_mean = np.mean(local_area)
                ratio = strength / local_mean if local_mean > 0 else 0
                
                if ratio > self.peak_threshold_T:
                    peaks.append({
                        'position': (y_global, x_global),
                        'strength': strength,
                        'ratio': ratio
                    })
        
        return peaks

    def detect_cumulative_periodogram(self, spectrum):
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        first_quadrant = spectrum[center_h:, center_w:]
        
        if first_quadrant.size < 10:
            return False, 0.0
        
        # Remove DC component
        first_quadrant = first_quadrant.copy()
        first_quadrant[0, 0] = 0
        
        # Simple radial sampling
        qh, qw = first_quadrant.shape
        max_radius = min(qh, qw) // 2
        
        if max_radius < 3:
            return False, 0.0
        
        # Sample along radial lines
        radial_samples = []
        for angle in np.linspace(0, np.pi/2, 8):  # 8 angles in first quadrant
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            for r in range(1, max_radius):
                y = int(r * sin_a)
                x = int(r * cos_a)
                
                if 0 <= y < qh and 0 <= x < qw:
                    radial_samples.append((r, first_quadrant[y, x]))
        
        if len(radial_samples) < 5:
            return False, 0.0
        
        # Sort by radius
        radial_samples.sort(key=lambda x: x[0])
        
        # Calculate cumulative energy
        energies = [sample[1]**2 for sample in radial_samples]
        total_energy = sum(energies)
        
        if total_energy == 0:
            return False, 0.0
        
        cumulative = []
        cum_sum = 0
        for energy in energies:
            cum_sum += energy
            cumulative.append(cum_sum / total_energy)
        
        # Calculate simple gradients
        gradients = []
        for i in range(1, len(cumulative)):
            r_diff = radial_samples[i][0] - radial_samples[i-1][0]
            cum_diff = cumulative[i] - cumulative[i-1]
            
            if r_diff > 0:
                gradients.append(abs(cum_diff / r_diff))
        
        if not gradients:
            return False, 0.0
        
        max_gradient = max(gradients)
        detected = max_gradient > self.gradient_threshold
        
        return detected, max_gradient

    def load_image(self, img_path):
        try:
            img = np.array(Image.open(img_path).convert('L'))
            img = img.astype(np.float64)

            # downscale to 256x256
            if img.shape[0] > 256 or img.shape[1] > 256:
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

    def extract_detection_metrics(self, spectrum):
        peaks = self.detect_peaks_fast(spectrum)
        gradient_detected, max_gradient = self.detect_cumulative_periodogram(spectrum)
        
        peak_detected = len(peaks) >= self.min_peaks
        final_detected = peak_detected or gradient_detected
        
        return {
            'detected': final_detected,
            'peak_count': len(peaks),
            'max_peak_strength': max([p['strength'] for p in peaks]) if peaks else 0,
            'max_gradient': max_gradient,
            'spectrum_mean': np.mean(spectrum),
            'spectrum_std': np.std(spectrum),
            'spectrum_max': np.max(spectrum),
            'peak_method_detected': peak_detected,
            'gradient_method_detected': gradient_detected,
            'peak_threshold_T': self.peak_threshold_T,
            'gradient_threshold': self.gradient_threshold,
            'min_peaks': self.min_peaks
        }

    def get_detector_info(self):
        return {
            'method': 'robust_fast',
            'sensitivity': self.sensitivity,
            'lambda_param': self.lambda_param,
            'tau': self.tau,
            'sigma': self.sigma,
            'peak_threshold_T': self.peak_threshold_T,
            'gradient_threshold': self.gradient_threshold,
            'min_peaks': self.min_peaks,
            'predictor_filter': self.predictor_filter.tolist()
        }