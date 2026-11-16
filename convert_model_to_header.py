"""
Convert TensorFlow Lite Model to C/C++ Header Array
This creates a model_data.h file for ESP32 embedded deployment.
"""

import sys
from pathlib import Path

def tflite_to_header(tflite_path, output_path, model_name="microcrack_model"):
    """
    Convert a .tflite file to a C/C++ header file.
    
    Args:
        tflite_path: Path to .tflite model file
        output_path: Path to output .h file
        model_name: Base name for the C array variables
    """
    
    # Read binary model
    try:
        with open(tflite_path, 'rb') as f:
            model_data = f.read()
    except Exception as e:
        print(f"Error reading model file: {e}")
        return False
    
    # Generate header guard
    header_guard = f"_{Path(output_path).stem.upper()}_H_"
    
    # Create header file
    lines = []
    lines.append("#ifndef " + header_guard)
    lines.append("#define " + header_guard)
    lines.append("")
    lines.append("// TensorFlow Lite Model for Microcrack Detection")
    lines.append(f"// Model size: {len(model_data):,} bytes")
    lines.append("// Generated from INT8 quantized model")
    lines.append("")
    lines.append("// Align to 4-byte boundary for efficient access on 32-bit MCU")
    lines.append("alignas(4) const unsigned char " + model_name + "_data[] = {")
    
    # Write model bytes in rows of 16 bytes per line
    bytes_per_line = 16
    for i in range(0, len(model_data), bytes_per_line):
        chunk = model_data[i:i+bytes_per_line]
        line_str = "    " + ", ".join(f"0x{b:02x}" for b in chunk)
        if i + bytes_per_line < len(model_data):
            line_str += ","
        lines.append(line_str)
    
    lines.append("};")
    lines.append("")
    lines.append(f"const unsigned int {model_name}_data_len = {len(model_data)};")
    lines.append("")
    lines.append(f"#endif  // {header_guard}")
    
    # Write file
    try:
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        file_size = Path(output_path).stat().st_size
        
        print(f"✓ Model header file created: {output_path}")
        print(f"  Model size: {len(model_data):,} bytes ({len(model_data)/1024:.1f} KB)")
        print(f"  Header file size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        print(f"  Array variable: {model_name}_data[]")
        print(f"  Length variable: {model_name}_data_len")
        
        return True
    except Exception as e:
        print(f"Error writing header file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_model_to_header.py <model.tflite> [output.h] [model_name]")
        print("Example: python convert_model_to_header.py microcrack_int8.tflite model_data.h microcrack_model")
        sys.exit(1)
    
    tflite_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "model_data.h"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "microcrack_model"
    
    if not Path(tflite_path).exists():
        print(f"Error: Model file not found: {tflite_path}")
        sys.exit(1)
    
    success = tflite_to_header(tflite_path, output_path, model_name)
    sys.exit(0 if success else 1)
