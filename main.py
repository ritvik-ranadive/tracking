import os
import numpy as np
import cv2
from data_utils import generateMinibatch
import tensorflow as tf

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
images = []
for imagePath in trainSet:
    # print(imagePath[0])
    images.append(cv2.imread(os.path.join(script_dir,imagePath[0])))
for imagePath in dbSet:
    images.append(cv2.imread(os.path.join(script_dir, imagePath[0])))
for imagePath in testSet:
    images.append(cv2.imread(os.path.join(script_dir, imagePath[0])))
meanImage = np.mean(images, axis=0)
# print(np.shape(meanImage))
cv2.imwrite("meanImage.png", meanImage)
meanImage = np.array(meanImage).astype('float32') / 255

# Call to generate minibatch
batchSize = 10
input_triplets = generateMinibatch(trainSet, dbSet, dbSet_ape, dbSet_benchvise, dbSet_cam, dbSet_cat, dbSet_duck, batchSize)
# print('The input shape is : {}'.format(np.shape(input_triplets)))
# for input in input_triplets:
#     print(input)

# Read the images from the paths, normalize and zero center the pixel values
inputs_train = []
for input in input_triplets:
    anchorImage = np.array(cv2.imread(os.path.join(script_dir,input[0][0]))).astype('float32') / 255
    pullerImage = np.array(cv2.imread(os.path.join(script_dir,input[1][0]))).astype('float32') / 255
    pusherImage = np.array(cv2.imread(os.path.join(script_dir,input[2][0]))).astype('float32') / 255
    inputs_train += anchorImage-meanImage, pullerImage-meanImage, pusherImage-meanImage

inputs_train = np.asarray(inputs_train)
inputs_train = tf.Variable(inputs_train, tf.float32)
print(np.shape(inputs_train))




