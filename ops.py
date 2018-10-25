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
           activation_fn = tf.nn.relu, bn = True,
           scope="conv2d"):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    #     if bn:
    #         batch_norm = batch_norm()
    #         outputs = batch_norm(conv, train = bn_is_train)
    #
    # if activation_fn is not None:
    #     outputs = activation_fn(outputs)

    return conv

# class batch_norm(object):
# # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
#     def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
#         with tf.variable_scope(name):
#             self.epsilon = epsilon
#             self.momentum = momentum
#             self.name = name
#
#     def __call__(self, x, train=True):
#         return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
#                                         scale=True, scope=self.name)

class Batch_Norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.99, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, phase):#直接调用实例
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


def FE_layer(inputs, cout, bn_is_training=True, scope="FE_layer"):
    """

    :param inputs: a tensor of shape (batch_size, num_pts, cin)
    :param cout: # out channels
    :return:  a tensor of shape (batch_size, num_pts, cout)
    """
    channel = cout // 2
    cin = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(scope) as local_scope:
        num_pts = inputs.get_shape().as_list()[1]
        point_wise_feature = tf.layers.dense(inputs, channel,
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        batch_norm = Batch_Norm()
        point_wise_feature = batch_norm(point_wise_feature, phase=bn_is_training)
        point_wise_feature = tf.nn.relu(point_wise_feature)  # (batch_size, num_pts, cout // 2)
        aggregated_feature = tf.reduce_max(point_wise_feature, axis=1, keepdims=True)  # batch_size, 1, cout//2
        repeated = tf.tile(aggregated_feature, [1, num_pts, 1])  # (batch_size, num_pts, cout // 2)
        point_wise_concatenated_feature = tf.concat(axis=-1, values=[point_wise_feature, repeated])
        return point_wise_concatenated_feature


def dense_bn_relu(inputs, units, bn_is_training=True, activation_fn=tf.nn.relu, scope="dense"):
    with tf.variable_scope(scope) as local_scope:
        out = tf.layers.dense(inputs, units, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        batch_norm = Batch_Norm()
        out = batch_norm(out, phase=bn_is_training)
        out = activation_fn(out)
        return out