import tensorflow as tf
from utils import *
from ops import *

BATCH_SIZE = 32





def placeholder_inputs(batch_size, num_point = 2048, num_feature = 6):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_feature))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    return pointclouds_pl, labels_pl

def get_model(point_cloud_color, bn_is_train, num_label= 16, style_transfer_test = False):
    """
    
    :param point_cloud_color: a placeholder of shape (batch_size, num_pts, 6)
    :param reuse: 
    :param bn_is_train: 
    :return: 
    """
    with tf.variable_scope("cls") as scope:
        FE_1 = FE_layer(point_cloud_color, 64 , bn_is_training=bn_is_train, scope="VFE-1")
        avg_out1 = tf.reduce_mean(FE_1, axis=1, name="avg_out1")
        FE_2 = FE_layer(FE_1, 256, bn_is_training=bn_is_train, scope="VFE-2")
        avg_out2 = tf.reduce_mean(FE_2, axis=1, name="avg_out2")
        FE_3 = FE_layer(FE_2, 1024, bn_is_training=bn_is_train, scope="VFE-3")
        avg_out3 = tf.reduce_mean(FE_3, axis=1, name="avg_out3")
        FE_4 = FE_layer(FE_3, 2048, bn_is_training=bn_is_train, scope="VFE-4") # batch_size, num_pts, 2048
        avg_out4 = tf.reduce_mean(FE_4, axis=1, name="avg_out4")
        FE_5 = FE_layer(FE_4, 4096, bn_is_training=bn_is_train, scope="VFE-5")
        avg_out5 = tf.reduce_mean(FE_5, axis=1, name="avg_out5")

        global_aggregated_feature = tf.reduce_mean(FE_5, axis= 1) # batch_size, 4096
        dense_1 = dense_bn_relu(global_aggregated_feature, 2048, scope="dense1") # batch_size, 2048
        #dense_1 = tf.nn.dropout(dense_1, keep_prob)

        dense_2 = dense_bn_relu(dense_1, 1024, bn_is_train, scope="dense2") # batch_size, 2048
        #dense_2 = tf.nn.dropout(dense_2, keep_prob)

        dense_3 = dense_bn_relu(dense_2, 256, bn_is_train, scope="dense3")
        #dense_3 = tf.nn.dropout(dense_3, keep_prob)

        dense_4 = dense_bn_relu(dense_3, 64, bn_is_train, scope="dense4")
        logits = tf.layers.dense(dense_4, num_label) # batch_size, 16
        if style_transfer_test:
            return logits, avg_out1, avg_out2, avg_out3, avg_out4, avg_out5, dense_1, dense_2, dense_3, dense_4
        else:
            return logits


def get_training_loss(logits, label, weights):
    """
    
    :param logits: 
    :param label: a placeholder of shape (batch_size,)
    :return: 
    """
    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=label, weights=weights) # batch_size, 1
    loss_sum = tf.summary.scalar("training_loss", loss)
    #acc = tf.sum(tf.argmax(logits, axis=1) == label) /
    return loss, loss_sum

def get_eval_loss(logits, label, weights):
    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=label, weights=weights) # batch_size, 1
    loss_sum = tf.summary.scalar("evaluation_loss", loss)
    #acc = tf.sum(tf.argmax(logits, axis=1) == label) /
    return loss, loss_sum



if __name__ == "__main__":
    pc_pl, labels_pl = placeholder_inputs(BATCH_SIZE)
    logits = get_model(pc_pl)
    loss = get_training_loss(logits, labels_pl)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_val = sess.run([loss], feed_dict={pc_pl:np.random.randn(BATCH_SIZE, 2048, 6),
                                    labels_pl:np.random.randint(15, size=(BATCH_SIZE,))})
        print(loss_val)