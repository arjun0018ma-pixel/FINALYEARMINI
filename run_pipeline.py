#!/usr/bin/env python3
"""
Automated pipeline runner - Execute all post-training tasks sequentially
Converts Keras model to TFLite, validates accuracy, generates headers
"""

import subprocess
import sys
import time
from pathlib import Path

# ===========================
# Configuration
# ===========================
SCRIPT_DIR = Path(__file__).parent
SCRIPTS = [
    ("convert_tflite.py", "TFLite Conversion (Float32 + INT8)"),
    ("test_tflite.py", "TFLite Accuracy Validation"),
    ("image_to_header.py", "Image to C Header (test_image.h)"),
    ("convert_model_to_header.py", "Model to C Header (model_data.h)"),
    ("generate_comparison.py", "Model Comparison Report"),
]

def run_script(script_name, description):
    """Run a single Python script and report status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {script_name}\n")
    
    script_path = SCRIPT_DIR / script_name
    
    if not script_path.exists():
        print(f"✗ ERROR: {script_path} not found")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=SCRIPT_DIR,
            capture_output=False,
            text=True,
            timeout=3600  # 1 hour timeout per script
        )
        
        if result.returncode == 0:
            print(f"\n✓ SUCCESS: {description} completed")
            return True
        else:
            print(f"\n✗ FAILED: {description} (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n✗ TIMEOUT: {description} exceeded 1 hour")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {description} - {e}")
        return False

def main():
    """Execute pipeline."""
    print("\n" + "="*60)
    print("AUTOMATED PIPELINE RUNNER")
    print("="*60)
    print("\nChecking prerequisites...")
    
    # Check for trained model
    model_path = SCRIPT_DIR / "microcrack_cnn.keras"
    if not model_path.exists():
        print(f"✗ ERROR: Trained model not found at {model_path}")
        print("Please run: python train_cnn.py")
        sys.exit(1)
    
    print(f"✓ Found trained model: {model_path}")
    print(f"✓ Ready to execute {len(SCRIPTS)} pipeline steps\n")
    
    # Execute each script
    results = {}
    start_time = time.time()
    
    for script_name, description in SCRIPTS:
        results[description] = run_script(script_name, description)
        
        # Stop on first failure
        if not results[description]:
            print(f"\n⚠ Pipeline stopped due to failure in: {description}")
            break
    
    # Summary
    elapsed = time.time() - start_time
    completed = sum(1 for v in results.values() if v)
    
    print(f"\n\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Completed: {completed}/{len(results)} steps\n")
    
    for description, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {description}")
    
    print(f"\n{'='*60}")
    
    if completed == len(results):
        print("✓ PIPELINE COMPLETE - All steps succeeded!")
        print("\nNext steps:")
        print("1. Copy embedded_ml/wokwi/* to Wokwi project")
        print("2. Copy model_data.h and test_image.h to sketch.ino directory")
        print("3. Deploy to Wokwi and run simulation")
        sys.exit(0)
    else:
        print("✗ PIPELINE INCOMPLETE - Fix errors and retry")
        sys.exit(1)

if __name__ == "__main__":
    main()
