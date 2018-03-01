import numpy as np
import cv2
from random import randint

def generateMinibatch(trainSet, dbSet, dbSet_ape, dbSet_benchvise, dbSet_cam, dbSet_cat, dbSet_duck, meanImage, batchSize=100):

    inputImagePaths = []
    indices = [randint(0, len(trainSet)) for p in range(batchSize)]
    anchors = [trainSet[i] for i in indices]
    anchors = [[data[0], float(data[1]), float(data[2]), float(data[3]), float(data[4])] for data in anchors]
    print(np.shape(anchors))
    # exit(0)
    for anchor in anchors:
        anchor_quaternions = [anchor[1], anchor[2], anchor[3], anchor[4]]
        if anchor[0].__contains__("ape"):
            print()
            # Compare with quaternions of dbSet_ape and find proper puller
        elif anchor[0].__contains__("benchvise"):
            print()
            # Compare with quaternions of dbSet_benchvise and find proper puller
        elif anchor[0].__contains__("cam"):
            print()
            # Compare with quaternions of dbSet_cam and find proper puller
        elif anchor[0].__contains__("cat"):
            print()
            # Compare with quaternions of dbSet_cat and find proper puller
        elif anchor[0].__contains__("duck"):
            print()
            # Compare with quaternions of dbSet_duck and find proper puller

        # Choose a random pusher

        # Append the paths of anchor, puller and pusher to inputImagePaths