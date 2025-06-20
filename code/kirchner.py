"""
Kirchner fast resampling detector implementation
Based on: "Fast and reliable resampling detection by spectral analysis 
of fixed linear predictor residue" (2008)

Key Mathematical Foundations:
- Resampling: s(omega * chi) = sum h(omega * chi - chi_k) * s(chi_k) [Eq. 3]
- Prediction Error: e(omega * chi) = s(omega * chi) - sigma alpha_k * s(omega * chi + omega * k) [Eq. 5]
- Variance Periodicity: Var[e(x)] = Var[e(x + 1)] [Theorem 1]
- P-map: p = lambda_param * exp(-|e|^tau / sigma) [Eq. 21]
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.ndimage import convolve
from PIL import Image
from pathlib import Path


class KirchnerDetector:
    def __init__(self, sensitivity='medium', lambda_param=1.0, tau=2.0, sigma=1.0, threshold_factor=None):
        # Fixed 3x3 linear predictor from paper (Eq. 25)
        self.predictor_filter = np.array([
            [-0.25, 0.50, -0.25],
            [0.50,  0.00,  0.50],
            [-0.25, 0.50, -0.25]
        ])
        
        # P-map parameters from paper (Eq. 21)
        self.lambda_param = lambda_param
        self.tau = tau
        self.sigma = sigma
        self.threshold_factor = threshold_factor
        
        # Detection thresholds based on sensitivity
        sensitivity_params = {
            'low':    {'peak_threshold': 0.05, 'min_peaks': 1, 'gradient_threshold': 0.02},
            'medium': {'peak_threshold': 0.10, 'min_peaks': 2, 'gradient_threshold': 0.05}, 
            'high':   {'peak_threshold': 0.15, 'min_peaks': 3, 'gradient_threshold': 0.08}
        }
        
        params = sensitivity_params.get(sensitivity, sensitivity_params['medium'])
        self.peak_threshold = params['peak_threshold']
        self.min_peaks = params['min_peaks']
        self.gradient_threshold = params['gradient_threshold']
        self.sensitivity = sensitivity

    def detect(self, img_path):
        try:
            print(f"      Loading image: {Path(img_path).name}")
            image = self.load_image(img_path)
            print(f"      Image shape: {image.shape}")
            
            print(f"      Running 6-step Kirchner detection...")
            results = self.detect_resampling(image)
            print(f"      Detection result: {results['detected']}")
            
            return {
                'detected': results['detected'],
                'p_map': results['p_map'],
                'spectrum': results['spectrum'],
                'prediction_error': results['prediction_error']
            }
        except Exception as e:
            print(f"      ERROR in detect(): {e}")
            raise

    def detect_resampling(self, image):
        # Step 1: Input preparation
        print(f"        Step 1: Input preparation")
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float64)
        
        # Step 2: Apply fixed linear predictor (Eq. 4 with preset coefficients)
        print(f"        Step 2: Applying fixed linear predictor")
        predicted = convolve(image, self.predictor_filter, mode='reflect')
        
        # Step 3: Calculate prediction error (Eq. 5)
        print(f"        Step 3: Computing prediction error")
        prediction_error = image - predicted
        
        # Step 4: Generate P-map using contrast function (Eq. 21)
        print(f"        Step 4: Generating P-map")
        p_map = self.generate_p_map(prediction_error)
        
        # Step 5: Spectral analysis via DFT
        print(f"        Step 5: Computing frequency spectrum")
        spectrum = self.compute_spectrum(p_map)
        
        # Step 6: Peak detection and decision
        print(f"        Step 6: Peak detection and decision")
        peaks = self.detect_characteristic_peaks(spectrum)
        detected = self.make_decision(peaks)
        
        return {
            'p_map': p_map,
            'spectrum': spectrum,
            'prediction_error': prediction_error,
            'peaks': peaks,
            'detected': detected
        }

    def load_image(self, img_path):
        try:
            img = np.array(Image.open(img_path))
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            
            # Resize if too large (for performance)
            if max(img.shape) > 2048:
                scale = 2048 / max(img.shape)
                h, w = int(img.shape[0] * scale), int(img.shape[1] * scale)
                img = cv2.resize(img, (w, h))
            
            # Normalize to [0,1]
            img = img.astype(np.float64)
            if img.max() > 1:
                img /= 255.0
                
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

    def generate_p_map(self, prediction_error):
        """
        Step 4: Generate contrast-enhanced probability map using Kirchner's formula.
        
        P-map formula: p = lambda * exp(-|e|^tau / sigma) [Eq. 21]
        """
        abs_error = np.abs(prediction_error)
        p_map = self.lambda_param * np.exp(-(abs_error ** self.tau) / self.sigma)
        
        print(f"          P-map range: [{np.min(p_map):.6f}, {np.max(p_map):.6f}]")
        print(f"          P-map mean: {np.mean(p_map):.6f}")
        
        return p_map

    def compute_spectrum(self, p_map):
        """
        Step 5: Compute frequency spectrum using 2D DFT.
        """
        # Apply 2D FFT and shift zero frequency to center
        fft_result = fft2(p_map)
        spectrum = np.abs(fftshift(fft_result))
        
        # Normalize spectrum
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
        
        print(f"          Spectrum range: [{np.min(spectrum):.6f}, {np.max(spectrum):.6f}]")
        return spectrum

    def detect_characteristic_peaks(self, spectrum):
        """
        Step 6a: Detect characteristic peaks in frequency spectrum.
        """
        print(f"          Detecting characteristic peaks...")
        
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        peaks = []
        search_radius = min(h, w) // 4
        
        # Search for peaks excluding DC component (center)
        for i in range(max(5, center_h - search_radius), 
                      min(h - 5, center_h + search_radius), 3):
            for j in range(max(5, center_w - search_radius), 
                          min(w - 5, center_w + search_radius), 3):
                
                # Skip DC component
                if abs(i - center_h) < 5 and abs(j - center_w) < 5:
                    continue
                
                current_value = spectrum[i, j]
                
                local_region = spectrum[max(0, i-2):min(h, i+3), 
                                     max(0, j-2):min(w, j+3)]
                
                if (current_value == np.max(local_region) and 
                    current_value > self.peak_threshold):
                    
                    freq_x = (j - center_w) / w
                    freq_y = (i - center_h) / h
                    
                    peaks.append({
                        'position': (i, j),
                        'strength': current_value,
                        'frequency': (freq_x, freq_y)
                    })
        
        peaks.sort(key=lambda p: p['strength'], reverse=True)
        
        print(f"          Found {len(peaks)} characteristic peaks")
        if peaks:
            print(f"          Strongest peak: {peaks[0]['strength']:.6f}")
        
        return peaks

    def make_decision(self, peaks):
        """
        Step 6b: Make final resampling decision based on peak analysis.
        """
        if len(peaks) < self.min_peaks:
            print(f"          Decision: NOT DETECTED (insufficient peaks: {len(peaks)} < {self.min_peaks})")
            return False
        
        strong_peaks = [p for p in peaks if p['strength'] > self.peak_threshold]
        
        if len(strong_peaks) >= self.min_peaks:
            print(f"          Decision: DETECTED ({len(strong_peaks)} strong peaks found)")
            return True
        else:
            print(f"          Decision: NOT DETECTED (only {len(strong_peaks)} strong peaks)")
            return False

    def compute_cumulative_periodogram(self, spectrum):
        """
        Compute cumulative periodogram from spectrum (Paper Section 5.2.2).
        
        The cumulative periodogram is used for automatic detection by analyzing
        the maximum gradient, which indicates sharp transitions due to periodic artifacts.
        """
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Use first quadrant only (0 <= f <= 0.5) as in paper
        quad = spectrum[center_h:, center_w:]
        
        # Flatten and compute cumulative energy distribution
        values = quad.flatten()
        total_energy = np.sum(values**2)
        
        if total_energy == 0:
            return np.zeros_like(values)
        
        # Calculate cumulative periodogram C(f) = sum(|P(f')|^2) / sum_total
        cumulative = np.cumsum(values**2) / total_energy
        
        return cumulative

    def extract_detection_metrics(self, spectrum):
        """
        Extract comprehensive detection metrics for analysis and multi-sensitivity testing.
        
        This function computes multiple detection criteria as described in the Kirchner paper:
        1. Cumulative periodogram gradient analysis
        2. Peak ratio analysis  
        3. Maximum peak strength analysis
        """
        peaks = self.detect_characteristic_peaks(spectrum)
        
        # Cumulative periodogram analysis (Paper Section 5.2.2)
        cumulative = self.compute_cumulative_periodogram(spectrum)
        
        # Calculate maximum gradient delta' = max|delta C(f)| (Paper Eq. 24)
        if len(cumulative) > 1:
            gradients = np.diff(cumulative)
            max_gradient = np.max(np.abs(gradients)) if len(gradients) > 0 else 0
        else:
            max_gradient = 0
        
        # Detection based on gradient threshold
        gradient_detected = max_gradient > self.gradient_threshold
        
        # Peak ratio analysis (strong peaks vs total peaks)
        strong_peaks = [p for p in peaks if p['strength'] > self.peak_threshold]
        peak_ratio = len(strong_peaks) / max(1, len(peaks)) if peaks else 0
        peak_ratio_threshold = 0.5  # At least 50% of peaks should be strong
        peak_ratio_detected = peak_ratio >= peak_ratio_threshold
        
        # Maximum peak analysis
        max_peak = max([p['strength'] for p in peaks]) if peaks else 0
        max_peak_threshold = self.peak_threshold
        max_peak_detected = max_peak > max_peak_threshold
        
        # Final detection decision (primary criterion)
        detected = len(strong_peaks) >= self.min_peaks
        
        return {
            # Basic metrics
            'peak_count': len(peaks),
            'kirchner_peaks': len(peaks) > 0,
            'max_peak_strength': max_peak,
            'spectrum_mean': np.mean(spectrum),
            'spectrum_std': np.std(spectrum),
            'spectrum_max': np.max(spectrum),
            'detected': detected,
            
            # Advanced metrics for multi-sensitivity analysis
            'max_gradient': max_gradient,
            'gradient_threshold': self.gradient_threshold,
            'gradient_detected': gradient_detected,
            'peak_ratio': peak_ratio,
            'peak_ratio_threshold': peak_ratio_threshold,
            'peak_ratio_detected': peak_ratio_detected,
            'max_peak': max_peak,
            'max_peak_threshold': max_peak_threshold,
            'max_peak_detected': max_peak_detected
        }

    def get_detector_info(self):
        return {
            'sensitivity': self.sensitivity,
            'lambda_param': self.lambda_param,
            'tau': self.tau,
            'sigma': self.sigma,
            'peak_threshold': self.peak_threshold,
            'min_peaks': self.min_peaks,
            'gradient_threshold': self.gradient_threshold,
            'predictor_filter': self.predictor_filter.tolist()
        }