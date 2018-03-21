import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.interactive("True")
# Parse the testSet and dbSet
script_dir = os.path.dirname(__file__)
dbSet = open(os.path.join(script_dir,"db.txt"), 'r').read().split("\n")
testSet = open(os.path.join(script_dir,"test.txt"), 'r').read().split("\n")
dbSet = [data.split(";") for data in dbSet]
dbSet = dbSet[0:len(dbSet)-1]
testSet = [data.split(";") for data in testSet]
testSet = testSet[0:len(testSet)-1]

# Parse matches.txt
matches = open(os.path.join(script_dir,"matches.txt"), 'r').read().split("\n")
matches = [data.split(":") for data in matches]

# print(dbSet[0:10])
# print(testSet[0:10])
# print(matches)

imagePaths = []
for match in matches:
    imagePaths.append([testSet[int(match[0])][0], dbSet[int(match[1])][0]])

# print(imagePaths)

fig = plt.figure()
for imagePath in imagePaths:
    img1 = mpimg.imread(imagePath[0])
    img2 = mpimg.imread(imagePath[1])
    ax1 = fig.add_subplot(2,1,1)
    ax1.imshow(img1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(img2)
    plt.pause(2)