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
    'img_cluttered',
]

DOWNSCALE_SIZE = 512  
DOWNSCALE = True
CROP_CENTER = False

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

def save_global_run_info(run_base_path, run_number, timestamp, folders, sensitivity):
    with open(f"{run_base_path}/run_info.txt", 'w') as f:
        f.write(f"Run Number: {run_number}\n")
        f.write(f"Date/Time: {timestamp}\n")
        f.write(f"Sensitivity: {sensitivity}\n")
        
        f.write("\n=== Processed Folders ===\n")
        for idx, folder in enumerate(folders, 1):
            f.write(f"{idx}. {folder}\n")
        
        f.write("\n=== Global Settings ===\n")
        f.write(f"Downscale Size: {DOWNSCALE_SIZE}\n")
        f.write(f"Downscale Enabled: {DOWNSCALE}\n")
        f.write(f"Crop Center: {CROP_CENTER}\n")
        f.write(f"Run Batch Processing: {RUN_BATCH_PROCESSING}\n")
        f.write(f"Batch Visualization: {BATCH_VISUALIZATION}\n")
        f.write(f"Run Scaling Test: {RUN_SCALING_TEST}\n")
        f.write(f"Run Rotation Test: {RUN_ROTATION_TEST}\n")

def save_folder_run_info(root_demo_folder, run_number, timestamp, folder_path, sensitivity):
    with open(f"{root_demo_folder}/folder_info.txt", 'w') as f:
        # Basic run information
        f.write(f"Run Number: {run_number}\n")
        f.write(f"Date/Time: {timestamp}\n")
        f.write(f"Source Folder: {folder_path}\n")
        
        # Processing settings
        f.write("\n=== Processing Settings ===\n")
        f.write(f"Sensitivity: {sensitivity}\n")
        f.write(f"Downscale Size: {DOWNSCALE_SIZE}\n")
        f.write(f"Downscale Enabled: {DOWNSCALE}\n")
        f.write(f"Crop Center: {CROP_CENTER}\n")
        
        # Batch settings
        f.write("\n=== Batch Processing Settings ===\n")
        f.write(f"Run Batch Processing: {RUN_BATCH_PROCESSING}\n")
        f.write(f"Batch Visualization: {BATCH_VISUALIZATION}\n")
        f.write(f"Save Intermediate Steps: {SAVE_INTERMEDIATE_STEPS}\n")
        f.write(f"Use Batch Max Gradient: {USE_BATCH_MAX_GRADIENT}\n")
        
        # Scaling test settings
        f.write("\n=== Scaling Test Settings ===\n")
        f.write(f"Run Scaling Test: {RUN_SCALING_TEST}\n")
        f.write(f"Scaling Visualization: {SCALING_VISUALIZATION}\n")
        f.write(f"Scaling Min: {SCALING_MIN}\n")
        f.write(f"Scaling Max: {SCALING_MAX}\n")
        f.write(f"Scaling Step: {SCALING_STEP}\n")
        f.write(f"Number of Scaling Factors: {len(SCALING_FACTORS)}\n")
        f.write(f"Scaling Factors: {', '.join(f'{sf:.2f}' for sf in SCALING_FACTORS)}\n")
        
        # Rotation test settings
        f.write("\n=== Rotation Test Settings ===\n")
        f.write(f"Run Rotation Test: {RUN_ROTATION_TEST}\n")
        f.write(f"Rotation Visualization: {ROTATION_VISUALIZATION}\n")
        f.write(f"Rotation Min: {ROTATION_MIN}\n")
        f.write(f"Rotation Max: {ROTATION_MAX}\n")
        f.write(f"Rotation Step: {ROTATION_STEP}\n")
        f.write(f"Number of Rotation Angles: {len(ROTATION_ANGLES)}\n")
        f.write(f"Rotation Angles: {', '.join(f'{ra:.1f}' for ra in ROTATION_ANGLES)}\n")
        
        # Interpolation methods
        f.write("\n=== Interpolation Methods ===\n")
        for method_name in INTERPOLATION_METHODS.keys():
            f.write(f"- {method_name}\n")

def run_demo_single_folder(folder_path, sensitivity='medium', run_base_path=None, run_number=None, timestamp=None):
    if not os.path.exists(folder_path):
        print(f"❌ Error: Image folder '{folder_path}' not found.")
        return None
        
    print(f"\n" + "="*80)
    print(f"🔍 PROCESSING FOLDER: {folder_path}")
    print("="*80)
    
    folder_name = Path(folder_path).name
    
    if run_base_path is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S') if timestamp is None else timestamp
        base_demo_folder = 'demo'
        Path(base_demo_folder).mkdir(parents=True, exist_ok=True)
        
        run_number = 1 if run_number is None else run_number
        run_folder_name = f"run_{run_number}"
        run_base_path = Path(base_demo_folder) / run_folder_name
        run_base_path.mkdir(parents=True, exist_ok=True)
    
    root_demo_folder = run_base_path / folder_name
    root_demo_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Results will be saved to: {root_demo_folder} (Run #{run_number})")
    save_folder_run_info(root_demo_folder, run_number, timestamp, folder_path, sensitivity)
        
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
    
    base_demo_folder = 'demo'
    Path(base_demo_folder).mkdir(parents=True, exist_ok=True)
    
    run_number = 1
    while True:
        run_folder_name = f"run_{run_number}"
        run_base_path = Path(base_demo_folder) / run_folder_name
        if not run_base_path.exists():
            break
        run_number += 1
    
    run_base_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Creating new run folder: {run_base_path} (Run #{run_number})")
    
    print(f"📂 Configured folders: {len(IMAGE_FOLDER_PATHS)}")
    for i, folder in enumerate(IMAGE_FOLDER_PATHS, 1):
        print(f"   {i}. {folder}")
    
    completed_folders = []
    failed_folders = []
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    save_global_run_info(run_base_path, run_number, timestamp, IMAGE_FOLDER_PATHS, sensitivity)
    
    for folder_path in IMAGE_FOLDER_PATHS:
        try:
            result_folder = run_demo_single_folder(folder_path, sensitivity, run_base_path, run_number, timestamp)
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
    
    return completed_folders, run_base_path


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

