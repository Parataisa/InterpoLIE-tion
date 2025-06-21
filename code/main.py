import os
import sys
import time
import glob
import pandas as pd
from pathlib import Path

from kirchner import KirchnerDetector
from batchProcessor import BatchProcessor, quick_scan

def run_scaling_test(input_folder, scaling_factors=None, sensitivity='medium', output_folder=None):
    try:
        from scalingTestSuite import ScalingTestSuite
        test_suite = ScalingTestSuite(scaling_factors=scaling_factors)
        return test_suite.run_scaling_test(input_folder, output_folder, sensitivity, KirchnerDetector)
    except ImportError:
        print("ScalingTestSuite not available. Please ensure scalingTestSuite.py is in the same directory.")
        return None
    except Exception as e:
        print(f"Error running scaling test: {e}")
        import traceback
        traceback.print_exc()
        return None


def single_image():
    img_files = glob.glob('img/*.jpg') + glob.glob('img/*.png') + glob.glob('img/*.jpeg')
    if not img_files:
        print("No images found in img/ folder")
        return None
    
    test_img = img_files[0]
    print(f"Testing single image: {test_img}")
    
    try:
        detector = KirchnerDetector(sensitivity='high')
        result = detector.detect(test_img)
        print(f"SUCCESS: Detection result = {result['detected']}")
        
        metrics = detector.extract_detection_metrics(result['spectrum'])
        print(f"Metrics extracted successfully: {list(metrics.keys())}")
        
        return result
    except Exception as e:
        print(f"ERROR testing single image: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_demo():
    if not os.path.exists('img'):
        print("No 'img' folder found for demo")
        print("Please create an 'img' folder with test images")
        return None
        
    print("Running Fast Kirchner Detector Demo...")
    print("Based on: Fast and Reliable Resampling Detection by Spectral Analysis")
    print("of Fixed Linear Predictor Residue (Kirchner, 2008) - Section 5")
    
    print("\n" + "="*70)
    print("KIRCHNER FAST RESAMPLING DETECTOR")
    print("="*70)
    
    print("\n=== Debug Test: Single Image ===")
    single_result = single_image()
    if single_result is None:
        print("Single image test failed, stopping demo")
        return None
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    print("\n=== Fast Batch Processing Demo ===")
    output_folder_batch = f'demo_batch_{timestamp}'
    try:
        results_batch = quick_scan('img', output_folder_batch, sensitivity='medium')
        print(f"✓ Fast batch processing completed! Results in: {output_folder_batch}")
        if not results_batch.empty:
            detected = results_batch['detected'].sum()
            total = len(results_batch)
            print(f"✓ Detected resampling in {detected}/{total} images")
        else:
            print("⚠ No results generated")
    except Exception as e:
        print(f"✗ Batch processing demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Scaling Test Demo ===")
    scaling_output = f'demo_scaling_{timestamp}'
    try:
        demo_scaling_factors = [0.7, 0.8, 0.9, 1.2, 1.5, 2.0]
        results_scaling = run_scaling_test('img', 
                                         scaling_factors=demo_scaling_factors,
                                         sensitivity='medium',
                                         output_folder=scaling_output)
        if results_scaling:
            print(f"✓ Scaling test demo completed! Results in: {scaling_output}")
            if 'overall_detection_rate' in results_scaling:
                print(f"✓ Overall detection rate: {results_scaling['overall_detection_rate']:.3f}")
        else:
            print("⚠ Scaling test not available or failed")
    except Exception as e:
        print(f"✗ Scaling test demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    return output_folder_batch

def main():
    run_demo()
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)