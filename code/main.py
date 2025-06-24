import os
import sys
import time
import cv2

from pathlib import Path
from batchProcessor import quick_scan
from scalingTestSuite import run_scaling_test
from rotationTestSuite import run_rotation_test

# Configuration
IMAGE_FOLDER_PATH = 'img'
DOWNSCALE_SIZE = 1024  
DOWNSCALE = True
CROP_CENTER = False

RUN_BATCH_PROCESSING = True
BATCH_VISUALIZATION = True
SAVE_INTERMEDIATE_STEPS = True

RUN_SCALING_TEST = True
SCALING_VISUALIZATION = True

RUN_ROTATION_TEST = True
ROTATION_VISUALIZATION = True

SCALING_FACTORS = [0.2, 0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 1.8, 2.0]
ROTATION_ANGLES = [5, 10, 15, 30, 45, 60, 90, 180, 270]

INTERPOLATION_METHODS = {
    #'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    #'cubic': cv2.INTER_CUBIC,
    #'lanczos': cv2.INTER_LANCZOS4
}

def run_demo(sensitivity='medium'):
    if not os.path.exists(IMAGE_FOLDER_PATH):
        print(f"Error: Image folder '{IMAGE_FOLDER_PATH}' not found.")
        return None
        
    print("Running Kirchner Fast Resampling Detector Demo")
    print("\n" + "="*60)
    print("KIRCHNER RESAMPLING DETECTOR")
    print("="*60)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    root_demo_folder = f'demo/{IMAGE_FOLDER_PATH}_{timestamp}'
    Path(root_demo_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 All results will be saved to: {root_demo_folder}")
    
    if RUN_BATCH_PROCESSING:
        print("\n=== Batch Processing with Gradient-Focused Analysis ===")
        output_folder_batch = Path(root_demo_folder) / 'batch_results'
        try:
            results_batch = quick_scan(
                IMAGE_FOLDER_PATH, 
                str(output_folder_batch), 
                sensitivity=sensitivity, 
                downscale_size=DOWNSCALE_SIZE, 
                downscale=DOWNSCALE, 
                crop_center=CROP_CENTER,
                save_visualizations=BATCH_VISUALIZATION,
                save_intermediate_steps=SAVE_INTERMEDIATE_STEPS
            )
            print(f"✓ Batch processing completed! Results in: {output_folder_batch}")
            print(f"✓ Analysis report: {output_folder_batch}/batch_analysis_report.png")
            if not results_batch.empty:
                detected = results_batch['detected'].sum()
                total = len(results_batch)
                print(f"✓ Detection results: {detected}/{total} images flagged")
            else:
                print("⚠ No results generated")
        except Exception as e:
            print(f"✗ Batch processing failed: {e}")
    if RUN_SCALING_TEST:
        print("\n=== Scaling Test with Individual Data Analysis ===")
        scaling_output = Path(root_demo_folder) / 'scaling_test'
        try:
            results_scaling = run_scaling_test(
                IMAGE_FOLDER_PATH, 
                scaling_factors=SCALING_FACTORS,
                interpolation_methods=INTERPOLATION_METHODS,
                sensitivity=sensitivity,
                output_folder=str(scaling_output),
                create_visualizations=SCALING_VISUALIZATION,
                downscale_size=DOWNSCALE_SIZE,
                downscale=DOWNSCALE,
                crop_center=CROP_CENTER
            )
            print(f"✓ Scaling test completed! Results in: {scaling_output}")
            if results_scaling and 'overall_detection_rate' in results_scaling:
                print(f"✓ Overall detection rate: {results_scaling['overall_detection_rate']:.3f}")
        except Exception as e:
            print(f"✗ Scaling test failed: {e}")

    if RUN_ROTATION_TEST:
        print("\n=== Rotation Test with Angular Analysis ===")
        rotation_output = Path(root_demo_folder) / 'rotation_test'
        try:
            results_rotation = run_rotation_test(
                IMAGE_FOLDER_PATH,
                rotation_angles=ROTATION_ANGLES,
                interpolation_methods=INTERPOLATION_METHODS,
                sensitivity=sensitivity,
                output_folder=str(rotation_output),
                create_visualizations=ROTATION_VISUALIZATION,
                downscale_size=DOWNSCALE_SIZE,
                downscale=DOWNSCALE,
                crop_center=CROP_CENTER
            )
            print(f"✓ Rotation test completed! Results in: {rotation_output}")
            if results_rotation and 'overall_detection_rate' in results_rotation:
                print(f"✓ Overall detection rate: {results_rotation['overall_detection_rate']:.3f}")
        except Exception as e:
            print(f"✗ Rotation test failed: {e}")
    
    print(f"\n🎯 Demo completed! All results organized in: {root_demo_folder}")
    
    return root_demo_folder

def main():
    try:
        result_folder = run_demo(sensitivity='medium')
        
        if result_folder:
            print(f"\n✅ Demo completed successfully!")
            print(f"📊 Check results in: {result_folder}")
        else:
            print(f"\n❌ Demo failed to complete")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()