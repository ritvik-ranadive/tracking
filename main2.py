from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
from data_utils import generateMinibatch
import tensorflow as tf
# from solver import model_fn

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.DEBUG)


# Triplet Loss Calculation
def calculate_loss(output):

    # l_triplets = tf.Variable(0.0, dtype=tf.float32)
    # l_pairs = tf.Variable(0.0, dtype=tf.float32)
    # m = tf.Variable(0.01, dtype=tf.float32)
    # triplet_term = tf.Variable(0.0, dtype=tf.float32)
    l_triplets = 0.0
    l_pairs = 0.0
    m = 0.01
    # triplet_term = 0.0
    i = 0
    while i != np.shape(output)[0]:
        print('Loss Calculation: {}'.format(i))
        fxa = tf.Variable(output[i], dtype=tf.float32)
        # print('fxa: {}'.format(np.shape(fxa)))
        fxp = tf.Variable(output[i + 1], dtype=tf.float32)
        # print('fxp: {}'.format(np.shape(fxp)))
        fxm = tf.Variable(output[i + 2], dtype=tf.float32)
        # print('fxm: {}'.format(np.shape(fxm)))
        term_p = tf.subtract(fxa, fxp)                                          #   fxa - fx+
        # print('term_p: {}'.format(np.shape(term_p)))
        term_m = tf.subtract(fxa, fxm)                                          #   fxa - fx-
        # print('term_m: {}'.format(np.shape(term_m)))

        triplet_term1 = tf.add(tf.square(tf.norm(term_p)), m)                   #   \\fxa - fx+\\22 + m
        # print('triplet_term1: {}'.format(np.shape(triplet_term1)))

        triplet_term2 = tf.square(tf.norm(term_m))                              #   \\fxa - fx-\\22
        # print('triplet_term2: {}'.format(np.shape(triplet_term2)))

        triplet_term3 = tf.divide(triplet_term1, triplet_term2)                 #   (\\fxa - fx-\\22 / \\fxa - fx+\\22 + m)
        # print('triplet_term3: {}'.format(np.shape(triplet_term3)))

        triplet_term4 = tf.maximum(0.0, tf.subtract(1.0, triplet_term3))        #   max(0, (1 - (\\fxa - fx-\\22 / \\fxa - fx+\\22 + m)))
        # print('triplet_term4: {}'.format(np.shape(triplet_term4)))

        l_triplets = tf.add(l_triplets, triplet_term4)                          #   Sigma, adding losses from previous loops
        # print('l_triplets: {}'.format(np.shape(l_triplets)))

        l_pairs = tf.add(l_pairs, tf.square(tf.norm(term_p)))
        # print('l_pairs: {}'.format(np.shape(l_pairs)))

        i = i + 3

    loss = tf.add(l_triplets, l_pairs)
    # print('loss: {}'.format(np.shape(loss)))
    return loss

# Convolutional Neural Network
def cnn_model_fn(features, mode):
    print('Entered Network')
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Triplet images are 64x64 pixels, and have 3 color channel and 3 such images are to be processed together
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])
    # print('Input Layer: {}'.format(np.shape(input_layer)))

    # Convolutional Layer #1
    # Computes 16 features using a 8x8 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 64, 64, 9]
    # Output Tensor Shape: [batch_size, 57, 57, 16]
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=[8, 8],
      padding="valid",
      activation=tf.nn.relu)
    # print('Conv1: {}'.format(np.shape(conv1)))

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 57, 57, 16]
    # Output Tensor Shape: [batch_size, 28, 28, 16]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding="valid")
    # print('Pool1: {}'.format(np.shape(pool1)))

    # Convolutional Layer #2
    # Computes 7 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 16]
    # Output Tensor Shape: [batch_size, 25, 25, 7]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=7,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)
    # print('Conv2: {}'.format(np.shape(conv2)))

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 25, 25, 7]
    # Output Tensor Shape: [batch_size, 12, 12, 7]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding="valid")
    # print('Pool2: {}'.format(np.shape(pool2)))

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 12, 12, 7]
    # Output Tensor Shape: [batch_size, 12 * 12 * 7]
    pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 7])
    # print('Pool2_flat: {}'.format(np.shape(pool2_flat)))

    # Dense Layer #1
    # Densely connected layer with 256 neurons
    # Input Tensor Shape: [batch_size, 12 * 12 * 7]
    # Output Tensor Shape: [batch_size, 256]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)
    # print('Dense1: {}'.format(np.shape(dense1)))

    # Add dropout operation; 0.5 probability that element will be kept
    dropout = tf.layers.dropout(
    inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    # print('Dropout: {}'.format(np.shape(dropout)))

    # Dense Layer #2
    # Densely connected layer with 16 neurons
    # Input Tensor Shape: [batch_size, 256]
    # Output Tensor Shape: [batch_size, 16]
    output = tf.layers.dense(inputs=dropout, units=16, activation=tf.nn.relu)

    print('Exit Network')
    # print(output)
    # print('Output: {}'.format(np.shape(output)))

    # Loss Calculation
    loss = calculate_loss(output)

    # Gradient Descent
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        # print('Loss: {}'.format(loss))
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        print('Reached Here')
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

# Main Code
script_dir = os.path.dirname(__file__)
trainSet = open(os.path.join(script_dir,"train.txt"), 'r').read().split("\n")
dbSet = open(os.path.join(script_dir,"db.txt"), 'r').read().split("\n")
testSet = open(os.path.join(script_dir,"test.txt"), 'r').read().split("\n")

trainSet = [data.split(";") for data in trainSet]
trainSet = trainSet[0:len(trainSet)-1]
dbSet = [data.split(";") for data in dbSet]
dbSet = dbSet[0:len(dbSet)-1]
testSet = [data.split(";") for data in testSet]
testSet = testSet[0:len(testSet)-1]
# print(np.shape(trainSet))

# Split the dbSet into ape, benchvise, cam, cat and duck
dbSet_ape = [data for data in dbSet if data[0].__contains__("ape")]
dbSet_benchvise = [data for data in dbSet if data[0].__contains__("benchvise")]
dbSet_cam = [data for data in dbSet if data[0].__contains__("cam")]
dbSet_cat = [data for data in dbSet if data[0].__contains__("cat")]
dbSet_duck = [data for data in dbSet if data[0].__contains__("duck")]

# Calculate mean image of train set
# images = []
# for imagePath in trainSet:
#     # print(imagePath[0])
#     images.append(cv2.imread(os.path.join(script_dir,imagePath[0])))
# for imagePath in dbSet:
#     images.append(cv2.imread(os.path.join(script_dir, imagePath[0])))
# for imagePath in testSet:
#     images.append(cv2.imread(os.path.join(script_dir, imagePath[0])))
# meanImage = np.mean(images, axis=0)
# # print(np.shape(meanImage))
# cv2.imwrite("meanImage.png", meanImage)
meanImage = cv2.imread(os.path.join(script_dir,"meanImage.png"))
meanImage = np.array(meanImage).astype('float32') / 255

# Call to generate minibatch
batchSize = 90
input_triplets = generateMinibatch(trainSet, dbSet, dbSet_ape, dbSet_benchvise, dbSet_cam, dbSet_cat, dbSet_duck, batchSize)
# print('The input shape is : {}'.format(np.shape(input_triplets)))
# for input in input_triplets:
#     print(input)

# Read the images from the paths, normalize and zero center the pixel values
# inputs_train = np.zeros((batchSize,64,64,9))
inputs_train = []
for input in input_triplets:
    # input_attach = np.zeros((64, 64, 9))
    # print('Input_Attach: {}'.format(np.shape(input_attach)))
    anchorImage = np.array(cv2.imread(os.path.join(script_dir,input[0][0]))).astype('float32') / 255
    pullerImage = np.array(cv2.imread(os.path.join(script_dir,input[1][0]))).astype('float32') / 255
    pusherImage = np.array(cv2.imread(os.path.join(script_dir,input[2][0]))).astype('float32') / 255
    inputs_train += anchorImage-meanImage, pullerImage-meanImage, pusherImage-meanImage
    # input_attach[:, :, 0:3] = anchorImage - meanImage
    # input_attach[:, :, 3:6] = pullerImage - meanImage
    # input_attach[:, :, 6:9] = pusherImage - meanImage
    # inputs_train += input_attach

inputs_train = np.asarray(inputs_train)
# inputs_train = tf.Variable(inputs_train, tf.float32)
print('Input Shape: {}'.format(np.shape(inputs_train)))
# exit(0)

# Create the Estimator
model_dir = os.path.join(script_dir, "model/ex3model")
estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir)

# Setting up the logging
tensor_to_log = {"training_loss" : "loss"}
#logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": inputs_train}, batch_size=3, num_epochs=None, shuffle=True)
estimator.train(input_fn=train_input_fn, steps=30) #, hooks=[logging_hook])
print('Reached Here')

# model = tf.estimator.Estimator(model_fn)
#
# sess = tf.Session()
# # important step
# # tf.initialize_all_variables() no long valid from
# # 2017-03-02 if using tensorflow >= 0.12
# if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#     init = tf.initialize_all_variables()
# else:
#     init = tf.global_variables_initializer()
# # sess.run(cnn_layers(inputs_train))
#
# for i in range(100):
#     sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
# #     if i % 50 == 0:
# #         print(compute_accuracy(
# # mnist.test.images[:1000], mnist.test.labels[:1000]))
