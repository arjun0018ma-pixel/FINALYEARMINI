"""
Convert Image to C/C++ Header Array for ESP32
Usage: python image_to_header.py path/to/image.jpg output.h [--int8]
"""

import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

IMG_SIZE = 48

def image_to_header(image_path, output_path, quantize_int8=False):
    """
    Convert an image to a C/C++ header file.
    
    Args:
        image_path: Path to input image
        output_path: Path to output .h file
        quantize_int8: If True, quantize to INT8. Otherwise, use float32.
    """
    
    # Load image
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='rgb'
        )
    except Exception as e:
        print(f"Error loading image: {e}")
        return False
    
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Flatten
    img_flat = img_array.flatten()
    
    # Generate header
    header_guard = f"_{Path(output_path).stem.upper()}_H_"
    
    if quantize_int8:
        # Quantize to INT8: scale to [-128, 127]
        img_data = ((img_flat * 255) - 128).astype(np.int8)
        dtype_name = "int8_t"
        data_type = "INT8"
    else:
        # Keep as float32
        img_data = img_array.flatten().astype(np.float32)
        dtype_name = "float"
        data_type = "FLOAT32"
    
    # Create header file
    lines = []
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}")
    lines.append("")
    lines.append(f"// Test image: {Path(image_path).name}")
    lines.append(f"// Size: {IMG_SIZE}x{IMG_SIZE}x3")
    lines.append(f"// Format: {data_type}")
    lines.append(f"// Total elements: {len(img_data)}")
    lines.append("")
    
    lines.append(f"const {dtype_name} test_image_data[] = {{")
    
    # Write data in rows of 16 values for readability
    values_per_line = 16
    for i in range(0, len(img_data), values_per_line):
        chunk = img_data[i:i+values_per_line]
        line_str = "    " + ", ".join(str(int(v)) for v in chunk)
        if i + values_per_line < len(img_data):
            line_str += ","
        lines.append(line_str)
    
    lines.append("};")
    lines.append("")
    lines.append(f"const unsigned int test_image_data_len = {len(img_data)};")
    lines.append("")
    lines.append(f"#endif  // {header_guard}")
    
    # Write file
    try:
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"[OK] Header file created: {output_path}")
        print(f"  Image: {Path(image_path).name}")
        print(f"  Size: {IMG_SIZE}x{IMG_SIZE}x3 ({len(img_data)} elements)")
        print(f"  Format: {data_type}")
        return True
    except Exception as e:
        print(f"Error writing header file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python image_to_header.py <image_path> <output.h> [--int8]")
        print("Example: python image_to_header.py test_image.jpg test_image.h --int8")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    quantize_int8 = "--int8" in sys.argv
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    success = image_to_header(image_path, output_path, quantize_int8)
    sys.exit(0 if success else 1)
