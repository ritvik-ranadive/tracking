from __future__ import print_function
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def cnn_layers(features):
    # define placeholder for inputs to network
    # xs = tf.placeholder(tf.float32, [None, 36864])
    xs = features
    # ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 64, 64, 3])
    # print(x_image.shape)  # [n_samples, 28,28,1]

    ## conv1 layer ##
    W_conv1 = weight_variable([8, 8, 3, 16])  # patch 8x8, in size 9, out size 16
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 56x56x16
    h_pool1 = max_pool_2x2(h_conv1)  # output size 28x28x16

    ## conv2 layer ##
    W_conv2 = weight_variable([5, 5, 16, 7])  # patch 5x5, in size 16, out size 7
    b_conv2 = bias_variable([7])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 24x24x7
    h_pool2 = max_pool_2x2(h_conv2)  # output size 12x12x7

    ## fc1 layer ##
    W_fc1 = weight_variable([13 * 13 * 7, 256])
    b_fc1 = bias_variable([256])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 13 * 13 * 7])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## fc2 layer ##
    W_fc2 = weight_variable([256, 16])
    b_fc2 = bias_variable([16])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    print(h_fc2)
    # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    # prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                               reduction_indices=[1]))       # loss
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)