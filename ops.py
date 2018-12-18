import tensorflow as tf

def dense(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "FC"):
        w = tf.get_variable("w", [shape[1], output_size], tf.float32,
                            tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, w) + bias, w, bias
        else:
            return tf.matmul(input_, w) + bias

def conv2d(input_, output_dim,
           k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02,
           scope="conv2d"):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv

class Batch_Norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.99, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, phase):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      epsilon=self.epsilon,
                      updates_collections=None,
                      scale=True,
                      is_training=phase,
                      scope=self.name)


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D max pooling.
  
    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints
  
    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs




def dense_bn_relu(inputs, units, bn_is_training=True, activation_fn=tf.nn.relu, scope="dense"):
    with tf.variable_scope(scope) as local_scope:
        out = tf.layers.dense(inputs, units, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        batch_norm = Batch_Norm()
        out = batch_norm(out, phase=bn_is_training)
        out = activation_fn(out)
        return out