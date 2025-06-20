import os
import sys
import time
import glob
import pandas as pd
from pathlib import Path

from kirchner import KirchnerDetector
from batchProcessor import BatchProcessor, quick_scan, quick_scan_all_sensitivities


def detect_single_image(image_path, sensitivity='medium', save_plot=False):
    detector = KirchnerDetector(sensitivity)
    result = detector.detect(image_path)

    print(f"Image: {Path(image_path).name}")
    print(f"Resampling detected: {'YES' if result['detected'] else 'NO'}")

    if save_plot:
        try:
            from batchProcessor import create_single_visualization
            output_folder = Path(image_path).parent
            
            vis_result = {
                'file_name': Path(image_path).name,
                'detected': result['detected'],
                'p_map': result['p_map'],
                'spectrum': result['spectrum'],
                'prediction_error': result['prediction_error']
            }
            
            create_single_visualization(vis_result, output_folder)
            plot_path = output_folder / f'{Path(image_path).stem}_kirchner_analysis.png'
            print(f"Plot saved to: {plot_path}")
        except Exception as e:
            print(f"Warning: Could not save plot: {e}")

    return result['detected']


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


def test_single_image_debug():
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
        
        # Test metrics extraction
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
        
    print("Running Kirchner Detector Demo...")
    
    print("\n" + "="*60)
    print("KIRCHNER RESAMPLING DETECTOR DEMO")
    print("="*60)
    
    print("\n=== Debug Test: Single Image ===")
    single_result = test_single_image_debug()
    if single_result is None:
        print("Single image test failed, stopping demo")
        return None
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    print("\n=== Single Sensitivity Demo (HIGH) ===")
    output_folder_single = f'demo_single_{timestamp}'
    try:
        results_single = quick_scan('img', output_folder_single, sensitivity='high')
        print(f"✓ Single sensitivity demo completed! Results in: {output_folder_single}")
        if not results_single.empty:
            detected = results_single['detected'].sum()
            total = len(results_single)
            print(f"✓ Detected resampling in {detected}/{total} images")
        else:
            print("⚠ No results generated")
    except Exception as e:
        print(f"✗ Single sensitivity demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Multi-Sensitivity Demo (ALL LEVELS) ===")
    output_folder_multi = f'demo_multi_{timestamp}'
    try:
        results_multi = quick_scan_all_sensitivities('img', output_folder_multi)
        print(f"✓ Multi-sensitivity demo completed! Results in: {output_folder_multi}")
        if not results_multi.empty:
            detected_low = results_multi.get('detected_low', pd.Series()).sum()
            detected_medium = results_multi.get('detected_medium', pd.Series()).sum()
            detected_high = results_multi.get('detected_high', pd.Series()).sum()
            total = len(results_multi)
            print(f"✓ Detection Summary:")
            print(f"    LOW sensitivity: {detected_low}/{total} images")
            print(f"    MEDIUM sensitivity: {detected_medium}/{total} images")
            print(f"    HIGH sensitivity: {detected_high}/{total} images")
        else:
            print("⚠ No results generated")
    except Exception as e:
        print(f"✗ Multi-sensitivity demo failed: {e}")
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
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print(f"✓ Demo folders created:")
    if os.path.exists(output_folder_single):
        print(f"  - Single sensitivity: {output_folder_single}")
    if os.path.exists(output_folder_multi):
        print(f"  - Multi-sensitivity: {output_folder_multi}")
    if os.path.exists(scaling_output):
        print(f"  - Scaling test: {scaling_output}")
    
    print(f"\n📊 Check the output folders for:")
    print(f"  - CSV results files")
    print(f"  - Visualization PNG files")
    print(f"  - Detailed analysis reports")
            
    return output_folder_multi


def print_usage():
    """Print usage information."""
    print("Kirchner Resampling Detector")
    print("="*50)
    print("Usage:")
    print("  python main.py [command] [options]")
    print("")
    print("Commands:")
    print("  demo                          - Run comprehensive demonstration")
    print("  batch <folder>                - Batch process images")
    print("  scaling-test <folder>         - Run scaling factor test")
    print("  single <image>                - Analyze single image")
    print("")
    print("Options:")
    print("  --sensitivity <level>         - Set sensitivity: low, medium, high")
    print("  --all-sensitivities          - Test all sensitivity levels")
    print("  --save-plot                  - Save visualization plot")
    print("")
    print("Examples:")
    print("  python main.py demo")
    print("  python main.py batch img/")
    print("  python main.py batch img/ --all-sensitivities")
    print("  python main.py scaling-test img/ --factors 0.8,1.2,1.5")
    print("  python main.py single image.jpg --save-plot")
    print("")
    print("Input Requirements:")
    print("  - Supported formats: jpg, jpeg, png, tiff, tif, bmp, webp")
    print("  - For demo: create 'img/' folder with test images")


def main():
    if len(sys.argv) == 1:
        run_demo()
        return
    
    command = sys.argv[1]
    
    if command == "help" or command == "--help" or command == "-h":
        print_usage()
        return
    
    elif command == "demo":
        run_demo()
    
    elif command == "scaling-test":
        if len(sys.argv) < 3:
            print("Error: scaling-test requires input folder")
            print("Usage: python main.py scaling-test <input_folder> [output_folder] [options]")
            print("       python main.py scaling-test <input_folder> --factors 0.8,1.2,1.5")
            sys.exit(1)
            
        input_folder = sys.argv[2]
        output_folder = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
        
        scaling_factors = None
        if '--factors' in sys.argv:
            try:
                idx = sys.argv.index('--factors')
                if idx + 1 < len(sys.argv):
                    factors_str = sys.argv[idx + 1]
                    scaling_factors = [float(f.strip()) for f in factors_str.split(',')]
            except (ValueError, IndexError):
                print("Error: Invalid scaling factors format")
                print("Example: --factors 0.8,1.2,1.5")
                sys.exit(1)
                
        sensitivity = 'medium'
        if '--sensitivity' in sys.argv:
            try:
                idx = sys.argv.index('--sensitivity')
                if idx + 1 < len(sys.argv):
                    sensitivity = sys.argv[idx + 1]
                    if sensitivity not in ['low', 'medium', 'high']:
                        print("Error: Sensitivity must be 'low', 'medium', or 'high'")
                        sys.exit(1)
            except IndexError:
                print("Error: --sensitivity requires a value")
                sys.exit(1)
        
        print("Running SCALING TEST with detailed metrics...")
        print(f"Input folder: {input_folder}")
        print(f"Scaling factors: {scaling_factors if scaling_factors else 'default range'}")
        print(f"Sensitivity: {sensitivity}")
        
        results = run_scaling_test(input_folder, scaling_factors, sensitivity, output_folder)
        if results:
            print("✓ Scaling test completed successfully!")
        else:
            print("✗ Scaling test failed!")
    
    elif command == "batch":
        if len(sys.argv) < 3:
            print("Error: batch requires input folder")
            print("Usage: python main.py batch <input_folder> [output_folder] [options]")
            sys.exit(1)
            
        input_folder = sys.argv[2]
        output_folder = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
        test_all = "--all-sensitivities" in sys.argv
        
        if test_all:
            print("Running batch processing with ALL sensitivity levels...")
            results = quick_scan_all_sensitivities(input_folder, output_folder)
        else:
            sensitivity = 'medium'  
            if '--sensitivity' in sys.argv:
                try:
                    idx = sys.argv.index('--sensitivity')
                    if idx + 1 < len(sys.argv):
                        sensitivity = sys.argv[idx + 1]
                        if sensitivity not in ['low', 'medium', 'high']:
                            print("Error: Sensitivity must be 'low', 'medium', or 'high'")
                            sys.exit(1)
                except IndexError:
                    print("Error: --sensitivity requires a value")
                    sys.exit(1)
            
            print(f"Running batch processing with {sensitivity.upper()} sensitivity...")
            results = quick_scan(input_folder, output_folder, sensitivity)
        
        print("✓ Batch processing completed!")
    
    elif command == "single":
        if len(sys.argv) < 3:
            print("Error: single requires image path")
            print("Usage: python main.py single <image_path> [--sensitivity medium] [--save-plot]")
            sys.exit(1)
            
        image_path = sys.argv[2]
        
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            sys.exit(1)
        
        sensitivity = 'medium'
        save_plot = '--save-plot' in sys.argv
        
        if '--sensitivity' in sys.argv:
            try:
                idx = sys.argv.index('--sensitivity')
                if idx + 1 < len(sys.argv):
                    sensitivity = sys.argv[idx + 1]
                    if sensitivity not in ['low', 'medium', 'high']:
                        print("Error: Sensitivity must be 'low', 'medium', or 'high'")
                        sys.exit(1)
            except IndexError:
                print("Error: --sensitivity requires a value")
                sys.exit(1)
        
        print(f"Analyzing single image: {image_path}")
        print(f"Sensitivity: {sensitivity}")
        detected = detect_single_image(image_path, sensitivity, save_plot)
        
        print(f"\n{'='*50}")
        print(f"FINAL RESULT: {'RESAMPLING DETECTED' if detected else 'NO RESAMPLING DETECTED'}")
        print(f"{'='*50}")
    
    else:
        print(f"Unknown command: {command}")
        print("Run 'python main.py help' for usage information")
        sys.exit(1)


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