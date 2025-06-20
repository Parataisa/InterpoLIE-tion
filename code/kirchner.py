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
        
        # Step 6: Peak detection and decision using both paper methods
        print(f"        Step 6: Peak detection and decision")
        peaks = self.detect_characteristic_peaks(spectrum)
        detected = self.make_decision(peaks, spectrum)
        
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
        Step 6a: Detect characteristic peaks using Kirchner paper method.
        
        Paper method: "peaks n times greater than a local average magnitude"
        "T is selected to be 10" (Section 3.4)
        """
        print(f"          Detecting characteristic peaks (Kirchner paper method)...")
        
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Paper parameters
        T = 10  # Threshold factor from paper: "T is selected to be 10"
        search_radius = min(h, w) // 4  # Search radius
        
        print(f"          Using paper method: T={T}, search_radius={search_radius}")
        print(f"          Spectrum range: [{np.min(spectrum):.6f}, {np.max(spectrum):.6f}]")
        
        peaks = []
        candidate_count = 0
        
        # Search for peaks excluding DC component
        for i in range(max(5, center_h - search_radius), 
                      min(h - 5, center_h + search_radius), 3):
            for j in range(max(5, center_w - search_radius), 
                          min(w - 5, center_w + search_radius), 3):
                
                # Skip DC component (center)
                if abs(i - center_h) < 5 and abs(j - center_w) < 5:
                    continue
                
                current_value = spectrum[i, j]
                candidate_count += 1
                
                # Get local region for average calculation (Kirchner method)
                local_region = spectrum[max(0, i-2):min(h, i+3), 
                                     max(0, j-2):min(w, j+3)]
                
                # Calculate local average excluding center point (paper method)
                local_flat = local_region.flatten()
                if len(local_flat) > 1:
                    center_idx = len(local_flat) // 2
                    # Exclude center point from average
                    local_values = np.concatenate([local_flat[:center_idx], local_flat[center_idx+1:]])
                    local_average = np.mean(local_values) if len(local_values) > 0 else 0
                else:
                    local_average = 0
                
                # Kirchner paper conditions
                is_local_max = (current_value == np.max(local_region))
                is_significant = (local_average > 0 and current_value > T * local_average)
                
                if is_local_max and is_significant:
                    freq_x = (j - center_w) / w
                    freq_y = (i - center_h) / h
                    ratio = current_value / local_average if local_average > 0 else float('inf')
                    
                    peaks.append({
                        'position': (i, j),
                        'strength': current_value,
                        'frequency': (freq_x, freq_y),
                        'local_average': local_average,
                        'ratio': ratio
                    })
                    
                    print(f"            Peak found: strength={current_value:.6f}, "
                          f"local_avg={local_average:.6f}, ratio={ratio:.1f}")
        
        # Sort by strength as in original
        peaks.sort(key=lambda p: p['strength'], reverse=True)
        
        print(f"          Kirchner method: {len(peaks)} peaks from {candidate_count} candidates")
        if peaks:
            print(f"          Strongest peak: {peaks[0]['strength']:.6f} (ratio: {peaks[0]['ratio']:.1f})")
        
        return peaks

    def make_decision(self, peaks, spectrum):
        """
        Step 6b: Make final resampling decision using both paper methods.
        
        Method 1: Peak-based detection (original method)
        Method 2: Cumulative periodogram analysis (Section 5.2.2)
        """
        print(f"          Making decision using Kirchner paper methods...")
        
        # Method 1: Traditional peak-based detection
        strong_peaks = [p for p in peaks if p['strength'] > self.peak_threshold]
        peak_detected = len(strong_peaks) >= self.min_peaks
        
        print(f"          Peak method: {len(strong_peaks)} strong peaks -> {'DETECTED' if peak_detected else 'NOT DETECTED'}")
        
        # Method 2: Cumulative periodogram method (Section 5.2.2)
        periodogram_detected, max_gradient = self.detect_with_cumulative_periodogram(spectrum)
        
        # Final decision: Use OR logic (either method can detect)
        final_detected = peak_detected or periodogram_detected
        
        if final_detected:
            methods = []
            if peak_detected:
                methods.append("peaks")
            if periodogram_detected:
                methods.append("periodogram")
            print(f"          DETECTED via: {', '.join(methods)}")
        else:
            print(f"          NOT DETECTED by either method")
        
        return final_detected

    def detect_with_cumulative_periodogram(self, spectrum):
        """
        Kirchner paper Section 5.2.2: Automatic detection via cumulative periodograms.
        
        "delta' = max |delta C(f)|" (Equation 24)
        "If delta' exceeds a specific threshold delta'_T, the signal is flagged as resampled"
        """
        print(f"          Cumulative periodogram method (Section 5.2.2)...")
        
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Use first quadrant only as described in paper
        first_quadrant = spectrum[center_h:, center_w:]
        
        # Apply contrast function gamma (high-pass filter + gamma correction)
        qh, qw = first_quadrant.shape
        y, x = np.ogrid[:qh, :qw]
        
        # Radial distance from quadrant center
        r = np.sqrt((x - qw//2)**2 + (y - qh//2)**2)
        max_r = np.sqrt((qw//2)**2 + (qh//2)**2)
        
        # High-pass radial filter (attenuates low frequencies)
        if max_r > 0:
            radial_filter = np.clip(r / max_r, 0, 1)
        else:
            radial_filter = np.ones_like(first_quadrant)
        
        # Apply contrast function (gamma correction)
        gamma = 2.0
        gamma_corrected = (first_quadrant * radial_filter) ** gamma
        
        # Calculate cumulative periodogram C(f) (Equation 23)
        flat_values = gamma_corrected.flatten()
        total_energy = np.sum(flat_values**2)
        
        if total_energy == 0:
            print(f"            No energy in spectrum")
            return False, 0.0
        
        # Cumulative energy distribution
        cumulative = np.cumsum(flat_values**2) / total_energy
        
        # Calculate maximum gradient delta' = max|∇C(f)| (Equation 24)
        if len(cumulative) > 1:
            gradients = np.diff(cumulative)
            max_gradient = np.max(np.abs(gradients))
        else:
            max_gradient = 0.0
        
        print(f"            Max gradient delta': {max_gradient:.6f}")
        print(f"            Threshold delta'_T: {self.gradient_threshold:.6f}")
        
        # Decision based on gradient threshold
        detected = max_gradient > self.gradient_threshold
        
        print(f"            Periodogram result: {'DETECTED' if detected else 'NOT DETECTED'}")
        
        return detected, max_gradient

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
        
        # Final detection decision using both methods
        peak_decision = len(strong_peaks) >= self.min_peaks
        periodogram_decision, _ = self.detect_with_cumulative_periodogram(spectrum)
        detected = peak_decision or periodogram_decision
        
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
            'max_peak_detected': max_peak_detected,
            
            # Paper method results
            'peak_method_detected': peak_decision,
            'periodogram_method_detected': periodogram_decision
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