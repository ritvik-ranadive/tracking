import os
import numpy as np
import cv2
from data_utils import generateMinibatch

script_dir = os.path.dirname(__file__)
trainSet = open(os.path.join(script_dir,"train.txt"), 'r').read().split("\n")
dbSet = open(os.path.join(script_dir,"db.txt"), 'r').read().split("\n")

trainSet = [data.split(";") for data in trainSet]
trainSet = trainSet[0:len(trainSet)-1]
dbSet = [data.split(";") for data in dbSet]
dbSet = dbSet[0:len(dbSet)-1]
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
meanImage = np.mean(images, axis=0)
# print(np.shape(meanImage))
cv2.imwrite("meanImage.png", meanImage)

# Call to generate minibatch
batchSize = 100
inputs_train = generateMinibatch(trainSet, dbSet, dbSet_ape, dbSet_benchvise, dbSet_cam, dbSet_cat, dbSet_duck, meanImage, batchSize)