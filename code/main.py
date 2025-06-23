import os
import sys
import time
import glob
import pandas as pd
from pathlib import Path

from kirchner import KirchnerDetector
from batchProcessor import BatchProcessor, quick_scan
from scalingTestSuite import ScalingTestSuite

IMAGE_FOLDER_PATH = 'img'

def run_scaling_test(input_folder, scaling_factors=None, sensitivity='medium', output_folder=None, create_visualizations=True):
    try:
        test_suite = ScalingTestSuite(scaling_factors=scaling_factors)
        return test_suite.run_scaling_test(input_folder, output_folder, sensitivity, KirchnerDetector, create_visualizations)
    except Exception as e:
        print(f"Error running scaling test: {e}")
        return None

def run_demo(sensitivity='medium'):
    if not os.path.exists(IMAGE_FOLDER_PATH):
        print(f"Error: Image folder '{IMAGE_FOLDER_PATH}' not found.")
        return None
        
    print("Running Kirchner Fast Resampling Detector Demo")
    
    print("\n" + "="*60)
    print("KIRCHNER RESAMPLING DETECTOR")
    print("="*60)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    root_demo_folder = f'kirchner_demo_{timestamp}'
    Path(root_demo_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 All results will be saved to: {root_demo_folder}")
    
    print("\n=== Batch Processing with Gradient-Focused Analysis ===")
    output_folder_batch = Path(root_demo_folder) / 'batch_results'
    try:
        results_batch = quick_scan(IMAGE_FOLDER_PATH, str(output_folder_batch), 
                                 sensitivity=sensitivity)
        print(f"✓ Batch processing completed! Results in: {output_folder_batch}")
        print(f"✓ Gradient-focused analysis: {output_folder_batch}/detailed_batch_analysis_report.png")
        if not results_batch.empty:
            detected = results_batch['detected'].sum()
            total = len(results_batch)
            print(f"✓ Detection results: {detected}/{total} images flagged")
        else:
            print("⚠ No results generated")
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
    
    print("\n=== Scaling Test with Individual Data Analysis ===")
    scaling_output = Path(root_demo_folder) / 'scaling_test'
    try:
        demo_scaling_factors = [0.4, 0.5, 0.6, 0.8, 0.9, 1.1, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0]
        results_scaling = run_scaling_test(IMAGE_FOLDER_PATH, 
                                         scaling_factors=demo_scaling_factors,
                                         sensitivity=sensitivity,
                                         output_folder=str(scaling_output),
                                         create_visualizations=True)
    except Exception as e:
        print(f"✗ Scaling test failed: {e}")
    
    print(f"\n🎯 Demo completed! All results organized in: {root_demo_folder}")

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