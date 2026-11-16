import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load model for quantization params
interp = tf.lite.Interpreter('microcrack_int8.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()
inp_scale = inp[0]['quantization'][0]
inp_zp = inp[0]['quantization'][1]

# Load image same way as image_to_header.py
img = tf.keras.preprocessing.image.load_img('../dataset/Positive/00001.jpg', target_size=(48,48), color_mode='rgb')
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_flat = img_array.flatten()

# Method 1: image_to_header.py formula (current)
header_vals = ((img_flat * 255) - 128).astype(np.int8)

# Method 2: correct TFLite quantization
correct_vals = (img_flat / inp_scale + inp_zp).astype(np.int8)

# Compare
print("First 20 values comparison:")
print("  Header: ", list(header_vals[:20]))
print("  Correct:", list(correct_vals[:20]))
print("  Match:", np.array_equal(header_vals, correct_vals))

# Also verify against the actual header file
with open('wokwi/src/test_image.h', 'r') as f:
    content = f.read()

# Extract numbers from the header
import re
numbers = re.findall(r'-?\d+', content[content.index('test_image_data[] = {'):])
header_from_file = np.array([int(n) for n in numbers[:-1]], dtype=np.int8)  # skip len value
print("\nHeader file first 20:", list(header_from_file[:20]))
print("Matches Python formula:", np.array_equal(header_from_file, header_vals))
