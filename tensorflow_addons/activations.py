import tensorflow as tf


def gelu(x, approximate=False):
    return tf.keras.activations.gelu(x, approximate=approximate)
