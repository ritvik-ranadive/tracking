from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from random import randint
import cv2
from data_utils import generateMinibatch
import matplotlib.pyplot as plt
import argparse
import collections
# from solver import model_fn

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
args = parser.parse_args()

# Triplet Loss Calculation
def calculate_loss(output):
    # print('Output shape: {}'.format(np.shape(output)[0]))
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
        # print('Loss Calculation: {}'.format(i))
        fxa = output[i]
        # fxa = tf.Print(fxa, [fxa], "fxa")
        # print('fxa: {}'.format(np.shape(fxa)))
        fxp = output[i + 1]
        # fxp = tf.Print(fxp, [fxp], "fxp")
        # print('fxp: {}'.format(np.shape(fxp)))
        fxm = output[i + 2]
        # fxm = tf.Print(fxm, [fxm], "fxm")
        # print('fxm: {}'.format(np.shape(fxm)))
        term_p = tf.subtract(fxa, fxp)                                          #   fxa - fx+
        # term_p = tf.Print(term_p, [term_p], "term_p")
        # print('term_p: {}'.format(np.shape(term_p)))
        term_m = tf.subtract(fxa, fxm)                                          #   fxa - fx-
        # term_m = tf.Print(term_m, [term_m], "term_m")
        # print('term_m: {}'.format(np.shape(term_m)))

        triplet_term0 = tf.square(tf.norm(term_p))
        # triplet_term0 = tf.Print(triplet_term0, [triplet_term0], "triplet_term0")

        triplet_term1 = tf.add(triplet_term0, m)                                #   \\fxa - fx+\\22 + m
        # triplet_term1 = tf.Print(triplet_term1, [triplet_term1], "triplet_term1")
        # print('triplet_term1: {}'.format(np.shape(triplet_term1)))

        triplet_term2 = tf.square(tf.norm(term_m))                              #   \\fxa - fx-\\22
        # triplet_term2 = tf.Print(triplet_term2, [triplet_term2], "triplet_term2")
        # print('triplet_term2: {}'.format(np.shape(triplet_term2)))

        triplet_term3 = tf.divide(triplet_term2, triplet_term1)                 #   (\\fxa - fx-\\22 / \\fxa - fx+\\22 + m)
        # triplet_term3 = tf.Print(triplet_term3, [triplet_term3], "triplet_term3")
        # print('triplet_term3: {}'.format(np.shape(triplet_term3)))

        triplet_term4 = tf.maximum(0.0, tf.subtract(1.0, triplet_term3))        #   max(0, (1 - (\\fxa - fx-\\22 / \\fxa - fx+\\22 + m)))
        # triplet_term4 = tf.Print(triplet_term4, [triplet_term4], "triplet_term4")
        # print('triplet_term4: {}'.format(np.shape(triplet_term4)))

        l_triplets = tf.add(l_triplets, triplet_term4)                          #   Sigma, adding losses from previous loops
        # l_triplets = tf.Print(l_triplets, [l_triplets], "l_triplets")
        # print('l_triplets: {}'.format(np.shape(l_triplets)))

        l_pairs = tf.add(l_pairs, tf.square(tf.norm(term_p)))
        # l_pairs = tf.Print(l_pairs, [l_pairs], "l_pairs")
        # print('l_pairs: {}'.format(np.shape(l_pairs)))

        i = i + 3

    loss = tf.add(l_triplets, l_pairs)
    loss = tf.Print(loss, [loss], "loss")
    # print('loss: {}'.format(np.shape(loss)))
    return loss

# Convolutional Neural Network
def cnn_model_fn(features, mode):
    # print('Entered Network')
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Triplet images are 64x64 pixels, and have 3 color channel and 3 such images are to be processed together
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])
    # input_layer = tf.Print(input_layer, [input_layer], "input_layer")
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
      activation=tf.nn.relu,
      data_format='channels_last')
    # print('Conv1: {}'.format(np.shape(conv1)))
    # conv1 = tf.Print(conv1, [conv1], "conv1")

        # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 57, 57, 16]
    # Output Tensor Shape: [batch_size, 28, 28, 16]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding="valid")
    # print('Pool1: {}'.format(np.shape(pool1)))
    # pool1 = tf.Print(pool1, [pool1], "pool1")

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
      activation=tf.nn.relu,
      data_format='channels_last')
    # print('Conv2: {}'.format(np.shape(conv2)))
    # conv2 = tf.Print(conv2, [conv2], "conv2")

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 25, 25, 7]
    # Output Tensor Shape: [batch_size, 12, 12, 7]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding="valid")
    # pool2 = tf.Print(pool2, [pool2], "pool2")
    # print('Pool2: {}'.format(np.shape(pool2)))

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 12, 12, 7]
    # Output Tensor Shape: [batch_size, 12 * 12 * 7]
    pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 7])
    # pool2_flat = tf.Print(pool2_flat, [pool2_flat], "pool2_flat")
    # print('Pool2_flat: {}'.format(np.shape(pool2_flat)))

    # Dense Layer #1
    # Densely connected layer with 256 neurons
    # Input Tensor Shape: [batch_size, 12 * 12 * 7]
    # Output Tensor Shape: [batch_size, 256]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.tanh)
    # dense1 = tf.Print(dense1, [dense1], "dense1")
    # print('Dense1: {}'.format(np.shape(dense1)))

    # Add dropout operation; 0.5 probability that element will be kept
    dropout = tf.layers.dropout(
    inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    # dropout = tf.Print(dropout, [dropout], "dropout")
    # print('Dropout: {}'.format(np.shape(dropout)))

    # Dense Layer #2
    # Densely connected layer with 16 neurons
    # Input Tensor Shape: [batch_size, 256]
    # Output Tensor Shape: [batch_size, 16]
    output = tf.layers.dense(inputs=dropout, units=16, activation=tf.nn.tanh)
    # output = tf.Print(output, [output], "output_layer")

    # print('Exit Network')
    # print(output)
    # print('Output: {}'.format(np.shape(output)))

    # Return the outputs when using the model after training
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'features': output,
        }
        print("Output Shape in PREDICT Mode: {}".format(np.shape(output)))
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # if mode == tf.estimator.ModeKeys.EVAL:
    #     print("Output Shape in EVAL Mode: {}".format(np.shape(output)))
    #     return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs)

    # Loss Calculation
    loss = calculate_loss(output)

    # Gradient Descent
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        # print('Reached Here')
        # optimizer = tf.train.AdamOptimizer(learning_rate=1e-50)
        # gvs = optimizer.compute_gradients(loss)
        # capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
        # for grad, var in capped_gvs:
        #     grad = tf.Print(grad, [grad], "grad")
        # train_op = optimizer.apply_gradients(capped_gvs)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def main(unused_argv):

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
    #     images.append(cv2.imread(os.path.join(script_dir, imagePath[0])))
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
    batchSize = 7410
    batchtype = "train"
    minibatchSize = 30
    input_triplets = generateMinibatch(trainSet, dbSet, dbSet_ape, dbSet_benchvise, dbSet_cam, dbSet_cat, dbSet_duck, batchSize, batchtype)
    # print('The input shape is : {}'.format(np.shape(input_triplets)))
    # for input in input_triplets:
    #     print(input)

    # Read the images from the paths, normalize and zero center the pixel values
    # inputs_train = np.zeros((batchSize,64,64,9))
    inputs_train = []
    i = 0
    for input in input_triplets:
        # print(input)
        # input_attach = np.zeros((64, 64, 9))
        # print('Input_Attach: {}'.format(np.shape(input_attach)))
        anchorImage = np.array(cv2.imread(os.path.join(script_dir,input[1][0]))).astype('float32') / 255
        pullerImage = np.array(cv2.imread(os.path.join(script_dir,input[2][0]))).astype('float32') / 255
        pusherImage = np.array(cv2.imread(os.path.join(script_dir,input[3][0]))).astype('float32') / 255
        inputs_train += anchorImage-meanImage, pullerImage-meanImage, pusherImage-meanImage
        # input_attach[:, :, 0:3] = anchorImage - meanImage
        # input_attach[:, :, 3:6] = pullerImage - meanImage
        # input_attach[:, :, 6:9] = pusherImage - meanImage
        # inputs_train += input_attach
    # exit(0)

    inputs_train = np.asarray(inputs_train)
    # print(inputs_train)
    for input in inputs_train:
        check = np.isnan(input)
        # print(np.shape(check))
        if check.any():
            print('Found NaN')
    # inputs_train = tf.Variable(inputs_train, tf.float32)
    print('Input Shape: {}'.format(np.shape(inputs_train)))
    # exit(0)

    # Create the Estimator
    model_dir = os.path.join(script_dir, "model/ex3model")
    estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir)

    # Train the model
    for x in range(1000):
        # print(x)
        # inputs_now = []
        indices = []
        for p in range(minibatchSize):
            y = randint(0, (batchSize-1))*3
            indices.append(y)
            indices.append(y + 1)
            indices.append(y + 2)
        # print(indices)
        inputs_now = [inputs_train[b] for b in indices]
        # print(np.shape(inputs_now))
        inputs_now = np.array(inputs_now)
        # exit(0)
        # inputs_now = inputs_train[x*30:(x*30)+30]
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": inputs_now}, batch_size=minibatchSize*3, num_epochs=None, shuffle=False)
        estimator.train(input_fn=train_input_fn, steps=1)
        # print('Reached Here')
    # exit(0)

    ############################################################
    ####################TRAINING ENDS###########################
    ####################START TESTING###########################
    ############################################################

    # We will have triplets here of anchor puller pusher all coming from the dbSet
    # These are being used to get the features for the dbSet
    batchSize_dbSet = 1335
    batchtype = "dbset"
    dbSet_triplets = generateMinibatch(dbSet, dbSet, dbSet_ape, dbSet_benchvise, dbSet_cam, dbSet_cat, dbSet_duck,
                                       batchSize_dbSet, batchtype)
    dbSet_images = []
    for input in dbSet_triplets:
        anchorImage = np.array(cv2.imread(os.path.join(script_dir,input[1][0]))).astype('float32') / 255
        pullerImage = np.array(cv2.imread(os.path.join(script_dir,input[2][0]))).astype('float32') / 255
        pusherImage = np.array(cv2.imread(os.path.join(script_dir,input[3][0]))).astype('float32') / 255
        dbSet_images += anchorImage-meanImage, pullerImage-meanImage, pusherImage-meanImage
        # print(input)
    dbSet_images = np.asarray(dbSet_images)
    for input in dbSet_images:
        check = np.isnan(input)
        # print(np.shape(check))
        if check.any():
            print('Found NaN')
    print('DB Set Shape: {}'.format(np.shape(dbSet_images)))

    # We will have triplets here of anchor coming from testSet and puller pusher coming from the dbSet
    # These are being used to get the features for the testSet
    batchSize_testSet = 3535
    batchtype = "test"
    testSet_triplets = generateMinibatch(testSet, dbSet, dbSet_ape, dbSet_benchvise, dbSet_cam, dbSet_cat, dbSet_duck,
                                       batchSize_testSet, batchtype)
    testSet_images = []
    for input in testSet_triplets:
        anchorImage = np.array(cv2.imread(os.path.join(script_dir,input[1][0]))).astype('float32') / 255
        pullerImage = np.array(cv2.imread(os.path.join(script_dir,input[2][0]))).astype('float32') / 255
        pusherImage = np.array(cv2.imread(os.path.join(script_dir,input[3][0]))).astype('float32') / 255
        testSet_images += anchorImage-meanImage, pullerImage-meanImage, pusherImage-meanImage
        # print(input)
    testSet_images = np.asarray(testSet_images)
    for input in testSet_images:
        check = np.isnan(input)
        # print(np.shape(check))
        if check.any():
            print('Found NaN')
    print('Test Set Shape: {}'.format(np.shape(testSet_images)))

    print("Reached here 2")

    # Get the features for dbSet in dbSet_features
    dbSet_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": dbSet_images}, num_epochs=1, shuffle=False)
    dbSet_results = estimator.predict(input_fn=dbSet_input_fn)
    dbSet_features = []
    for dbSet_result in dbSet_results:
        # print(dbSet_result['features'])
        dbSet_features += [dbSet_result['features']]

    # Get the features for the testSet in testSet_features
    testSet_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": testSet_images}, num_epochs=1, shuffle=False)
    testSet_results = estimator.predict(input_fn=testSet_input_fn)
    testSet_features = []
    for testSet_result in testSet_results:
        # print(testSet_result['features'])
        testSet_features += [testSet_result['features']]

    print('dbSet feature size: {}'.format(np.shape(dbSet_features)))
    print('testSet feature size: {}'.format(np.shape(testSet_features)))

    # Create the array that contains useful data for the dbSet and the testSet
    usefulData_dbSet = []
    i = 0
    for input in dbSet_triplets:
        usefulData_dbSet += [[input[0], input[1][1], input[1][2], input[1][3], input[1][4],
                            dbSet_features[i][0], dbSet_features[i][1], dbSet_features[i][2], dbSet_features[i][3],
                            dbSet_features[i][4], dbSet_features[i][5], dbSet_features[i][6], dbSet_features[i][7],
                            dbSet_features[i][8], dbSet_features[i][9], dbSet_features[i][10], dbSet_features[i][11],
                            dbSet_features[i][12], dbSet_features[i][13], dbSet_features[i][14], dbSet_features[i][15]]]
                            # [class, 4 quaternions, 16 features]
        i = i + 3
    usefulData_testSet = []
    i = 0
    for input in testSet_triplets:
        usefulData_testSet += [[input[0], input[1][1], input[1][2], input[1][3], input[1][4],
                            testSet_features[i][0], testSet_features[i][1], testSet_features[i][2], testSet_features[i][3],
                            testSet_features[i][4], testSet_features[i][5], testSet_features[i][6], testSet_features[i][7],
                            testSet_features[i][8], testSet_features[i][9], testSet_features[i][10], testSet_features[i][11],
                            testSet_features[i][12], testSet_features[i][13], testSet_features[i][14], testSet_features[i][15]]]
                            # [class, 4 quaternions, 16 features]
        i = i + 3

    writeFile = open("dbSet_output_data.txt", 'w')
    for input in usefulData_dbSet:
        for entry in input:
            writeFile.write(str(entry))
            writeFile.write(';')
        writeFile.write('\n')
        # print(input)
    # print("-------------------------------------------------------------------------------")
    writeFile2 = open("testSet_output_data.txt", 'w')
    for input in usefulData_testSet:
        for entry in input:
            writeFile2.write(str(entry))
            writeFile2.write(';')
        writeFile2.write('\n')
        # print(input)

def test(testSet, dbSet, histName = 'histogram'):

    fig, ax = plt.subplots()
    ind = np.asarray([c for c in range(4)])
    bf = cv2.BFMatcher()
    matches = bf.match(testSet[:,5:-1].astype(np.float32), dbSet[:,5:-1].astype(np.float32))
    good = 0
    # hist = {10:0, 20:0, 40:0, 120:0, 180:0 }
    hist = {10: 0, 20: 0, 40: 0, 180: 0}

    for m in matches:
        A = testSet[m.queryIdx];
        B = dbSet[m.trainIdx];
        if((int)(A[0]) == (int)(B[0])):
            good += 1
            # angularDist = np.arccos(np.abs(np.dot(A[1:5],B[1:5]))) * 360 / np.pi
            angularDist = np.arccos(np.abs(np.dot(A[1:5], B[1:5]))) * 180 / np.pi
            if(angularDist <= 180):
                hist[180] += 1
            # if (angularDist <= 180):
            #     hist[120] += 1
            if (angularDist <= 40):
                hist[40] += 1
            if (angularDist <= 20):
                hist[20] += 1
            if (angularDist <= 10):
                hist[10] += 1

    od = collections.OrderedDict(sorted(hist.items()))
    ax.bar(ind, list(od.values()), width=0.2, color='g')
    ax.xaxis.set_ticks(ind)
    ax.xaxis.set_ticklabels(('<10','<20','<40','<180'))
    plt.savefig('hist/{}.png'.format(histName))

    #return hist, good



if __name__ == "__main__":
    if(args.mode == 'train'):
        tf.app.run()
    else:
        x = np.genfromtxt(fname='dbSet_output_data.txt', delimiter=";");
        x = x[:, 0:21]
        y = np.genfromtxt(fname='testSet_output_data.txt', delimiter=";");
        y = y[:, 0:21]
        test(y, x);
