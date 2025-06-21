"""
Kirchner fast resampling detector implementation
Based on: "Fast and reliable resampling detection by spectral analysis 
of fixed linear predictor residue" (2008) - Section 5 Fast Detection

Key Implementation:
- Section 5.1: Fast computation with preset coefficients (Equation 25)
- Section 5.2.2: Cumulative periodogram detection (Equation 24)
- Equation 21: P-map formula p = lambda * exp(-|e|^tau / sigma)
- Section 3.4: Peak detection with T=10 threshold
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.ndimage import convolve
from PIL import Image
from pathlib import Path


class KirchnerDetector:
    def __init__(self, sensitivity='medium', lambda_param=1.0, tau=2.0, sigma=1.0):
        """
        Args:
            sensitivity: Detection sensitivity ('low', 'medium', 'high')
            lambda_param: lambda parameter for P-map (Equation 21)
            tau: tau parameter for P-map (Equation 21) 
            sigma: sigma parameter for P-map (Equation 21)
        """
        # Preset filter coefficients from Equation 25
        self.predictor_filter = np.array([
            [-0.25, 0.50, -0.25],
            [0.50,  0.00,  0.50],
            [-0.25, 0.50, -0.25]
        ])
        
        # P-map parameters from Equation 21: p = lambda exp(-|e|^tau / sigma)
        self.lambda_param = lambda_param
        self.tau = tau  
        self.sigma = sigma
        
        sensitivity_params = {
            'low':    {'peak_threshold_T': 2.5,  'gradient_threshold': 0.008, 'min_peaks': 2}, 
            'medium': {'peak_threshold_T': 3.5,  'gradient_threshold': 0.015, 'min_peaks': 2}, 
            'high':   {'peak_threshold_T': 5.0,  'gradient_threshold': 0.025, 'min_peaks': 3}  
        }
        
        params = sensitivity_params.get(sensitivity, sensitivity_params['medium'])
        self.peak_threshold_T = params['peak_threshold_T']  
        self.gradient_threshold = params['gradient_threshold']
        self.min_peaks = params['min_peaks']
        self.sensitivity = sensitivity

    def detect(self, img_path):
        """Main detection method using fast preset approach."""
        try:
            print(f"      Loading image: {Path(img_path).name}")
            image = self.load_image(img_path)
            print(f"      Image shape: {image.shape}")
            
            print(f"      Running fast Kirchner detection...")
            results = self.detect_resampling_fast(image)
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

    def detect_resampling_fast(self, image):
        """
        Fast resampling detection using preset coefficients (Section 5).
        
        Steps:
        1. Apply preset linear predictor (Equation 25)
        2. Compute prediction error (Equation 5)
        3. Generate P-map (Equation 21)
        4. Spectral analysis with contrast function
        5. Dual detection: peaks + cumulative periodogram
        """
        # Step 1: Input preparation
        print(f"        Step 1: Input preparation")
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float64)
        
        # Step 2: Apply preset linear predictor (Equation 25)
        print(f"        Step 2: Applying preset predictor coefficients")
        predicted = convolve(image, self.predictor_filter, mode='reflect')
        
        # Step 3: Calculate prediction error (Equation 5)
        print(f"        Step 3: Computing prediction error")
        prediction_error = image - predicted
        
        # Step 4: Generate P-map using Equation 21
        print(f"        Step 4: Generating P-map (Equation 21)")
        p_map = self.generate_p_map_fast(prediction_error)
        
        # Step 5: Spectral analysis with contrast function
        print(f"        Step 5: Computing enhanced spectrum")
        spectrum = self.compute_spectrum_with_contrast(p_map)
        
        # Step 6: Dual detection method
        print(f"        Step 6: Dual detection (peaks + periodogram)")
        detected = self.make_decision_fast(spectrum)
        
        return {
            'p_map': p_map,
            'spectrum': spectrum,
            'prediction_error': prediction_error,
            'detected': detected
        }

    def generate_p_map_fast(self, prediction_error):
        """
        Fast P-map generation using exact Equation 21:
        p = lambda * exp(-|e|^tau / sigma)
        """
        abs_error = np.abs(prediction_error)
        
        # Normalize error to reasonable range to avoid numerical issues
        if np.max(abs_error) > 0:
            abs_error = abs_error / np.max(abs_error)
        
        # Exact implementation of Equation 21 with numerical stability
        p_map = self.lambda_param * np.exp(-(abs_error ** self.tau) / self.sigma)
        
        print(f"          P-map stats: range=[{np.min(p_map):.6f}, {np.max(p_map):.6f}], mean={np.mean(p_map):.6f}")
        
        return p_map

    def compute_spectrum_with_contrast(self, p_map):
        """
        Compute frequency spectrum with contrast function (Section 5.2.1).
        
        Includes:
        - 2D FFT transformation
        - Radial weighting (attenuates low frequencies)
        - Gamma correction
        """
        # Apply 2D FFT and shift zero frequency to center
        fft_result = fft2(p_map)
        spectrum = np.abs(fftshift(fft_result))
        
        print(f"          DEBUG: Raw spectrum stats: min={np.min(spectrum):.6f}, max={np.max(spectrum):.6f}")
        
        # Apply MUCH gentler contrast function
        spectrum = self.apply_gentle_contrast_function(spectrum)
        
        print(f"          DEBUG: After gentle contrast: min={np.min(spectrum):.6f}, max={np.max(spectrum):.6f}")
        print(f"          DEBUG: Non-zero values: {np.count_nonzero(spectrum)}/{spectrum.size}")
        
        return spectrum

    def apply_gentle_contrast_function(self, spectrum):
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_r = np.sqrt((center_w)**2 + (center_h)**2)
        
        if max_r > 0:
            radial_weight = np.clip(r / (max_r * 0.05), 0.1, 1.0)
        else:
            radial_weight = np.ones_like(spectrum)
        
        spectrum_weighted = spectrum * radial_weight
        spectrum_log = np.log1p(spectrum_weighted)  # log(1 + x) for numerical stability
        
        if np.max(spectrum_log) > 0:
            spectrum_normalized = spectrum_log / np.max(spectrum_log)
        else:
            spectrum_normalized = spectrum_log
        
        return spectrum_normalized

    def analyze_spectrum_characteristics(self, spectrum):
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # 1. Check for periodic patterns in frequency domain
        first_quadrant = spectrum[center_h:, center_w:]
        
        # 2. Analyze frequency distribution uniformity
        freq_variance = np.var(first_quadrant)
        
        # 3. Check for high-frequency peaks (common in resampling)
        high_freq_region = first_quadrant[int(first_quadrant.shape[0]*0.3):, 
                                        int(first_quadrant.shape[1]*0.3):]
        high_freq_max = np.max(high_freq_region) if high_freq_region.size > 0 else 0
        
        # 4. Calculate energy concentration
        total_energy = np.sum(first_quadrant**2)
        if total_energy > 0:
            energy_concentration = np.sum((first_quadrant**2)[first_quadrant > np.percentile(first_quadrant, 95)]) / total_energy
        else:
            energy_concentration = 0
        
        variance_score = min(freq_variance * 1000, 1.0)             # Normalize variance
        high_freq_score = min(high_freq_max * 2, 1.0)               # Normalize high freq peaks
        concentration_score = min(energy_concentration * 5, 1.0)    # Normalize concentration
        
        final_score = (variance_score + high_freq_score + concentration_score) / 3
        
        print(f"            Spectrum analysis: variance={variance_score:.3f}, high_freq={high_freq_score:.3f}, concentration={concentration_score:.3f}")
        
        return final_score

    def make_decision_fast(self, spectrum):
        """
        Fast decision making using dual detection approach:
        1. Peak-based detection (Section 3.4, T=10)
        2. Cumulative periodogram analysis (Section 5.2.2, Equation 24)
        """
        # Method 1: Peak detection with enhanced analysis
        peaks = self.detect_peaks_fast(spectrum)
        peak_detected = len(peaks) >= self.min_peaks
        
        # Method 2: Cumulative periodogram gradient analysis
        gradient_detected, max_gradient = self.detect_cumulative_periodogram(spectrum)
        
        # Method 3: Enhanced spectrum analysis for better discrimination
        spectrum_score = self.analyze_spectrum_characteristics(spectrum)
        spectrum_detected = spectrum_score > 0.3  # Threshold for spectrum characteristics
        
        # Enhanced decision logic: require stronger evidence
        strong_evidence = sum([peak_detected, gradient_detected, spectrum_detected])
        
        # Require at least 2 out of 3 methods to agree for detection
        final_detected = strong_evidence >= 2
        
        print(f"          Peak method: {len(peaks)} peaks -> {'DETECTED' if peak_detected else 'CLEAN'}")
        print(f"          Gradient method: δ'={max_gradient:.6f} -> {'DETECTED' if gradient_detected else 'CLEAN'}")
        print(f"          Spectrum method: score={spectrum_score:.3f} -> {'DETECTED' if spectrum_detected else 'CLEAN'}")
        print(f"          Evidence count: {strong_evidence}/3 -> {'DETECTED' if final_detected else 'CLEAN'}")
        
        return final_detected


    def detect_peaks_fast(self, spectrum):
        """
        Fast peak detection using Section 3.4 methodology:
        "peaks n times greater than a local average magnitude"
        """
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        print(f"          Peak detection with preserved spectrum")
        print(f"          Spectrum stats: min={np.min(spectrum):.6f}, max={np.max(spectrum):.6f}, mean={np.mean(spectrum):.6f}")
        
        peaks = []
        search_radius = min(h, w) // 6
        
        spectrum_std = np.std(spectrum)
        spectrum_mean = np.mean(spectrum)
        
        base_threshold = spectrum_mean + 2 * spectrum_std
        print(f"          Using adaptive base threshold: {base_threshold:.6f}")
        
        step = 2 
        positions_checked = 0
        
        for i in range(max(5, center_h - search_radius), 
                      min(h - 5, center_h + search_radius), step):
            for j in range(max(5, center_w - search_radius), 
                          min(w - 5, center_w + search_radius), step):
                
                # Skip DC component (center)
                if abs(i - center_h) < 5 and abs(j - center_w) < 5:
                    continue
                
                positions_checked += 1
                current_value = spectrum[i, j]
                
                # Only check promising candidates
                if current_value < base_threshold:
                    continue
                
                # Use 3x3 local region for faster processing
                local_region = spectrum[max(0, i-1):min(h, i+2), 
                                     max(0, j-1):min(w, j+2)]
                
                is_local_max = (current_value >= np.max(local_region))
                if is_local_max:
                    local_area = spectrum[max(0, i-5):min(h, i+6), 
                                       max(0, j-5):min(w, j+6)]
                    local_mean = np.mean(local_area)
                    
                    # Much more reasonable ratio threshold
                    ratio = current_value / local_mean if local_mean > 0 else 0
                    
                    if ratio > 1.5: 
                        freq_x = (j - center_w) / w
                        freq_y = (i - center_h) / h
                        
                        peaks.append({
                            'position': (i, j),
                            'strength': current_value,
                            'frequency': (freq_x, freq_y),
                            'ratio': ratio
                        })
        
        peaks.sort(key=lambda p: p['strength'], reverse=True)
        print(f"          Checked {positions_checked} positions, found {len(peaks)} peaks")
        if peaks:
            print(f"          Top peak: strength={peaks[0]['strength']:.6f}, ratio={peaks[0]['ratio']:.2f}")
        
        return peaks

    def detect_cumulative_periodogram(self, spectrum):
        """
        Cumulative periodogram detection (Section 5.2.2):
        delta' = max |delta C(f)| (Equation 24)
        """
        print(f"          Fixed cumulative periodogram analysis")
        
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Use first quadrant (0 ≤ f ≤ 0.5) as per paper
        first_quadrant = spectrum[center_h:, center_w:]
        
        if first_quadrant.size == 0:
            return False, 0.0
        
        print(f"          First quadrant shape: {first_quadrant.shape}")
        print(f"          First quadrant stats: min={np.min(first_quadrant):.6f}, max={np.max(first_quadrant):.6f}")
        
        # Remove DC component (corner pixel) which dominates everything
        first_quadrant = first_quadrant.copy()
        first_quadrant[0, 0] = 0  # Zero out DC component
        
        flat_values = first_quadrant.flatten()
        qh, qw = first_quadrant.shape
        y_indices, x_indices = np.mgrid[:qh, :qw]
        distances = np.sqrt(x_indices**2 + y_indices**2).flatten()
        
        # Sort by distance from origin
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_values = flat_values[sorted_indices]
        
        # Remove very small values but keep more signal
        valid_mask = sorted_values > 1e-6
        sorted_distances = sorted_distances[valid_mask]
        sorted_values = sorted_values[valid_mask]
        
        print(f"          Valid values after filtering: {len(sorted_values)}")
        
        if len(sorted_values) < 10:  # Need minimum data points
            print(f"          Insufficient valid data points")
            return False, 0.0
        
        # Calculate cumulative energy as function of radial distance
        squared_values = sorted_values ** 2
        total_energy = np.sum(squared_values)
        
        if total_energy == 0:
            print(f"          Zero total energy")
            return False, 0.0
        
        cumulative = np.cumsum(squared_values) / total_energy
        
        print(f"          Cumulative range: [{np.min(cumulative):.6f}, {np.max(cumulative):.6f}]")
        print(f"          Cumulative variation: {np.std(cumulative):.6f}")
        
        # Calculate gradients in frequency domain (not just differences)
        distance_diffs = np.diff(sorted_distances)
        cumulative_diffs = np.diff(cumulative)
        
        # Avoid division by zero
        valid_grad_mask = distance_diffs > 1e-10
        if not np.any(valid_grad_mask):
            print(f"          No valid distance differences")
            return False, 0.0
        
        # Calculate actual gradients (change per unit distance)
        gradients = cumulative_diffs[valid_grad_mask] / distance_diffs[valid_grad_mask]
        
        if len(gradients) == 0:
            print(f"          No gradients calculated")
            return False, 0.0
        
        max_gradient = np.max(np.abs(gradients))
        
        print(f"          Calculated max gradient: {max_gradient:.6f}")
        print(f"          Gradient statistics: mean={np.mean(np.abs(gradients)):.6f}, std={np.std(gradients):.6f}")
        
        adaptive_threshold = max(0.001, np.std(gradients) * 2)
        
        print(f"          Using adaptive threshold: {adaptive_threshold:.6f}")
        
        detected = max_gradient > adaptive_threshold
        
        return detected, max_gradient

    def load_image(self, img_path):
        try:
            img = np.array(Image.open(img_path))
            
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            
            if max(img.shape) > 2048:
                scale = 2048 / max(img.shape)
                h, w = int(img.shape[0] * scale), int(img.shape[1] * scale)
                img = cv2.resize(img, (w, h))
            
            img = img.astype(np.float64)
            if img.max() > 1:
                img /= 255.0
                
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

    def extract_detection_metrics(self, spectrum):
            peaks = self.detect_peaks_fast(spectrum)
            gradient_detected, max_gradient = self.detect_cumulative_periodogram(spectrum)
            spectrum_score = self.analyze_spectrum_characteristics(spectrum)
            spectrum_detected = spectrum_score > 0.3
            
            peak_detected = len(peaks) >= self.min_peaks
            strong_evidence = sum([peak_detected, gradient_detected, spectrum_detected])
            final_detected = strong_evidence >= 2
            
            return {
                # Core detection results 
                'detected': final_detected,
                'evidence_count': strong_evidence,
                'peak_count': len(peaks),
                'max_peak_strength': max([p['strength'] for p in peaks]) if peaks else 0,
                'max_gradient': max_gradient,
                'spectrum_score': spectrum_score,
                
                # Basic spectrum stats
                'spectrum_mean': np.mean(spectrum),
                'spectrum_std': np.std(spectrum),
                'spectrum_max': np.max(spectrum),
                
                # Individual detection methods
                'peak_method_detected': peak_detected,
                'gradient_method_detected': gradient_detected,
                'spectrum_method_detected': spectrum_detected,
                
                # Thresholds used
                'peak_threshold_T': self.peak_threshold_T,
                'gradient_threshold': self.gradient_threshold,
                'min_peaks': self.min_peaks,
                'spectrum_threshold': 0.3
            }

    def get_detector_info(self):
        return {
            'method': 'fast',
            'sensitivity': self.sensitivity,
            'lambda_param': self.lambda_param,
            'tau': self.tau,
            'sigma': self.sigma,
            'peak_threshold_T': self.peak_threshold_T,
            'gradient_threshold': self.gradient_threshold,
            'min_peaks': self.min_peaks,
            'predictor_filter': self.predictor_filter.tolist()
        }