import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

interp = tf.lite.Interpreter('microcrack_int8.tflite')
interp.allocate_tensors()

# Get all ops in the model
details = interp._get_ops_details()
ops = set()
for d in details:
    ops.add(d['op_name'])
print("Model ops:", sorted(ops))
