import tensorflow as tf

def _get_bias_initializer():
    return tf.zeros_initializer()

def _get_weight_initializer():
    return tf.contrib.layers.xavier_initializer()