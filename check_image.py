import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

interp = tf.lite.Interpreter('microcrack_int8.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()
inp_scale = inp[0]['quantization'][0]
inp_zp = inp[0]['quantization'][1]
out_scale = out[0]['quantization'][0]
out_zp = out[0]['quantization'][1]
print("Input: scale=%s, zp=%s" % (inp_scale, inp_zp))
print("Output: scale=%s, zp=%s" % (out_scale, out_zp))

# Test positive image
img = tf.keras.preprocessing.image.load_img('../dataset/Positive/00001.jpg', target_size=(48,48), color_mode='rgb')
img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_q = (img_arr / inp_scale + inp_zp).astype(np.int8)
interp.set_tensor(inp[0]['index'], np.expand_dims(img_q, 0))
interp.invoke()
out_raw = interp.get_tensor(out[0]['index'])[0]
out_float = (out_raw.astype(np.float32) - out_zp) * out_scale
softmax = np.exp(out_float) / np.sum(np.exp(out_float))
pred = "POSITIVE" if np.argmax(softmax) == 1 else "NEGATIVE"
print("Positive 00001.jpg -> Raw: %s, Dequantized: %s, Softmax: %s, Prediction: %s" % (out_raw, out_float, softmax, pred))

# Test negative image
img2 = tf.keras.preprocessing.image.load_img('../dataset/Negative/00001.jpg', target_size=(48,48), color_mode='rgb')
img_arr2 = tf.keras.preprocessing.image.img_to_array(img2) / 255.0
img_q2 = (img_arr2 / inp_scale + inp_zp).astype(np.int8)
interp.set_tensor(inp[0]['index'], np.expand_dims(img_q2, 0))
interp.invoke()
out_raw2 = interp.get_tensor(out[0]['index'])[0]
out_float2 = (out_raw2.astype(np.float32) - out_zp) * out_scale
softmax2 = np.exp(out_float2) / np.sum(np.exp(out_float2))
pred2 = "POSITIVE" if np.argmax(softmax2) == 1 else "NEGATIVE"
print("Negative 00001.jpg -> Raw: %s, Dequantized: %s, Softmax: %s, Prediction: %s" % (out_raw2, out_float2, softmax2, pred2))
