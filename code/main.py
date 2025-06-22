import os
import sys
import time
import glob
import pandas as pd
from pathlib import Path

from kirchner import KirchnerDetector
from batchProcessor import BatchProcessor, quick_scan
from scalingTestSuite import ScalingTestSuite

def run_scaling_test(input_folder, scaling_factors=None, sensitivity='medium', output_folder=None):
    try:
        test_suite = ScalingTestSuite(scaling_factors=scaling_factors)
        return test_suite.run_scaling_test(input_folder, output_folder, sensitivity, KirchnerDetector)
    except Exception as e:
        print(f"Error running scaling test: {e}")
        return None

def run_demo(sensitivity='medium'):
    if not os.path.exists('img'):
        print("Please create an 'img' folder with test images")
        return None
        
    print("Running Kirchner Fast Resampling Detector Demo")
    print("Based on: Kirchner 2008 - Section 5 Fast Detection")
    
    print("\n" + "="*60)
    print("KIRCHNER RESAMPLING DETECTOR")
    print("="*60)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    root_demo_folder = f'kirchner_demo_{timestamp}'
    Path(root_demo_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 All results will be saved to: {root_demo_folder}")
    
    print("\n=== Batch Processing ===")
    output_folder_batch = Path(root_demo_folder) / 'batch_results'
    try:
        results_batch = quick_scan('img', str(output_folder_batch), sensitivity=sensitivity)
        print(f"✓ Batch processing completed! Results in: {output_folder_batch}")
        if not results_batch.empty:
            detected = results_batch['detected'].sum()
            total = len(results_batch)
            print(f"✓ Detection results: {detected}/{total} images flagged")
        else:
            print("⚠ No results generated")
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
    
    print("\n=== Scaling Test ===")
    scaling_output = Path(root_demo_folder) / 'scaling_test'
    try:
        demo_scaling_factors = [0.7, 0.8, 0.9, 1.2, 1.5, 2.0]
        results_scaling = run_scaling_test('img', 
                                         scaling_factors=demo_scaling_factors,
                                         sensitivity=sensitivity,
                                         output_folder=str(scaling_output))
        if results_scaling:
            print(f"✓ Scaling test completed! Results in: {scaling_output}")
            if 'overall_detection_rate' in results_scaling:
                print(f"✓ Overall detection rate: {results_scaling['overall_detection_rate']:.3f}")
        else:
            print("⚠ Scaling test not available or failed")
    except Exception as e:
        print(f"✗ Scaling test failed: {e}")
    
    print(f"\n🎯 Demo completed! All results organized in: {root_demo_folder}")
    print(f"   📂 Batch results: {output_folder_batch}")
    print(f"   📂 Scaling test: {scaling_output}")
    
    return root_demo_folder

def main():
    run_demo(sensitivity='medium')
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)