import os
import sys
import time
import cv2
import numpy as np

from pathlib import Path
from batchProcessor import quick_scan
from scalingTestSuite import run_scaling_test
from rotationTestSuite import run_rotation_test

IMAGE_FOLDER_PATHS = [
    'img',
    'img_jpg',
    'img_tif',
    'img_edited',
]

DOWNSCALE_SIZE = 1024  
DOWNSCALE = True
CROP_CENTER = True

RUN_BATCH_PROCESSING = True
BATCH_VISUALIZATION = True
SAVE_INTERMEDIATE_STEPS = False

RUN_SCALING_TEST = True
SCALING_VISUALIZATION = False

RUN_ROTATION_TEST = True
ROTATION_VISUALIZATION = False

USE_BATCH_MAX_GRADIENT = True

# Scaling factors configuration
SCALING_MIN = 0.5
SCALING_MAX = 2.0
SCALING_STEP = 0.1
SCALING_FACTORS = np.arange(SCALING_MIN, SCALING_MAX + SCALING_STEP/2, SCALING_STEP).tolist()

# Rotation angles configuration
ROTATION_MIN = 0.0
ROTATION_MAX = 45
ROTATION_STEP = 2.5
ROTATION_ANGLES = np.arange(ROTATION_MIN + ROTATION_STEP, ROTATION_MAX + ROTATION_STEP/2, ROTATION_STEP).tolist()

INTERPOLATION_METHODS = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}

def run_demo_single_folder(folder_path, sensitivity='medium'):
    if not os.path.exists(folder_path):
        print(f"❌ Error: Image folder '{folder_path}' not found.")
        return None
        
    print(f"\n" + "="*80)
    print(f"🔍 PROCESSING FOLDER: {folder_path}")
    print("="*80)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    folder_name = Path(folder_path).name
    root_demo_folder = f'demo/{folder_name}_{timestamp}'
    Path(root_demo_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Results will be saved to: {root_demo_folder}")
    
    max_gradient_from_batch = None
    
    if RUN_BATCH_PROCESSING:
        print("\n=== Batch Processing ===")
        if USE_BATCH_MAX_GRADIENT:
            print("🎯 Will re-evaluate with max gradient threshold after initial processing")
        else:
            print("🎯 Using default Kirchner thresholds")
            
        output_folder_batch = Path(root_demo_folder) / 'batch_results'
        try:
            results_batch = quick_scan(
                folder_path, 
                str(output_folder_batch), 
                sensitivity=sensitivity, 
                downscale_size=DOWNSCALE_SIZE, 
                downscale=DOWNSCALE, 
                crop_center=CROP_CENTER,
                save_visualizations=BATCH_VISUALIZATION,
                save_intermediate_steps=SAVE_INTERMEDIATE_STEPS,
                use_batch_max_gradient=USE_BATCH_MAX_GRADIENT
            )
            print(f"✓ Batch processing completed! Results in: {output_folder_batch}")
            print(f"✓ Analysis report: {output_folder_batch}/batch_analysis_report.png")
            if not results_batch.empty:
                detected = results_batch['detected'].sum()
                total = len(results_batch)
                print(f"✓ Detection results: {detected}/{total} images flagged")
                
                if 'max_gradient' in results_batch.columns:
                    max_gradient_from_batch = results_batch['max_gradient'].max()
                    print(f"✓ Max gradient found: {max_gradient_from_batch:.8f}")
                else:
                    print("⚠ No max_gradient column found in batch results")
            else:
                print("⚠ No results generated")
        except Exception as e:
            print(f"✗ Batch processing failed: {e}")
            return None
            
    test_threshold = None
    if USE_BATCH_MAX_GRADIENT and max_gradient_from_batch is not None:
        test_threshold = max_gradient_from_batch
        print(f"\n🎯 Using batch max gradient as threshold for scaling/rotation tests: {test_threshold:.8f}")
        if USE_BATCH_MAX_GRADIENT:
            print(f"   Note: Batch analysis also used this threshold for consistency")
    else:
        print(f"\n🎯 Using default Kirchner detector thresholds for scaling/rotation tests")
        
    if RUN_SCALING_TEST:
        print("\n=== Scaling Test ===")
        scaling_output = Path(root_demo_folder) / 'scaling_test'
        try:
            results_scaling = run_scaling_test(
                folder_path, 
                scaling_factors=SCALING_FACTORS,
                interpolation_methods=INTERPOLATION_METHODS,
                sensitivity=sensitivity,
                output_folder=str(scaling_output),
                create_visualizations=SCALING_VISUALIZATION,
                downscale_size=DOWNSCALE_SIZE,
                downscale=DOWNSCALE,
                crop_center=CROP_CENTER,
                max_gradient=test_threshold
            )
            print(f"✓ Scaling test completed! Results in: {scaling_output}")
            if results_scaling and 'overall_detection_rate' in results_scaling:
                print(f"✓ Overall detection rate: {results_scaling['overall_detection_rate']:.3f}")
        except Exception as e:
            print(f"✗ Scaling test failed: {e}")

    if RUN_ROTATION_TEST:
        print("\n=== Rotation Test ===")
        rotation_output = Path(root_demo_folder) / 'rotation_test'
        try:
            results_rotation = run_rotation_test(
                folder_path,
                rotation_angles=ROTATION_ANGLES,
                interpolation_methods=INTERPOLATION_METHODS,
                sensitivity=sensitivity,
                output_folder=str(rotation_output),
                create_visualizations=ROTATION_VISUALIZATION,
                downscale_size=DOWNSCALE_SIZE,
                downscale=DOWNSCALE,
                crop_center=CROP_CENTER,
                max_gradient=test_threshold
            )
            print(f"✓ Rotation test completed! Results in: {rotation_output}")
            if results_rotation and 'overall_detection_rate' in results_rotation:
                print(f"✓ Overall detection rate: {results_rotation['overall_detection_rate']:.3f}")
        except Exception as e:
            print(f"✗ Rotation test failed: {e}")
    
    print(f"\n✅ Folder '{folder_path}' completed! Results in: {root_demo_folder}")
    return root_demo_folder

def run_demo(sensitivity='medium'):
    print("🚀 Kirchner Fast Resampling Detector Demo")
    print("\n" + "="*80)
    print("KIRCHNER RESAMPLING DETECTOR - ANALYSIS")
    print("="*80)
    
    print(f"📂 Configured folders: {len(IMAGE_FOLDER_PATHS)}")
    for i, folder in enumerate(IMAGE_FOLDER_PATHS, 1):
        print(f"   {i}. {folder}")
    
    completed_folders = []
    failed_folders = []
    
    for folder_path in IMAGE_FOLDER_PATHS:
        try:
            result_folder = run_demo_single_folder(folder_path, sensitivity)
            if result_folder:
                completed_folders.append((folder_path, result_folder))
            else:
                failed_folders.append(folder_path)
        except Exception as e:
            print(f"❌ Unexpected error processing '{folder_path}': {e}")
            failed_folders.append(folder_path)
    
    print(f"\n" + "="*80)
    print("🎯 FINAL SUMMARY")
    print("="*80)
    
    if completed_folders:
        print(f"✅ Successfully processed {len(completed_folders)} folders:")
        for folder, result_path in completed_folders:
            print(f"   📁 {folder} → {result_path}")
    
    if failed_folders:
        print(f"\n❌ Failed to process {len(failed_folders)} folders:")
        for folder in failed_folders:
            print(f"   📁 {folder}")
    
    print(f"\n📊 Results Summary:")
    print(f"   Total folders: {len(IMAGE_FOLDER_PATHS)}")
    print(f"   Completed: {len(completed_folders)}")
    print(f"   Failed: {len(failed_folders)}")
    
    return completed_folders

def main():
    try:
        completed_folders = run_demo(sensitivity='medium')
        
        if completed_folders:
            print(f"\n🎉 Demo completed successfully!")
            print(f"📊 Processed {len(completed_folders)} folders")
        else:
            print(f"\n💥 No folders were processed successfully")
            
    except KeyboardInterrupt:
        print("\n\n⛔ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()