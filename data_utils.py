import numpy as np
import cv2
from random import randint

def generateMinibatch(trainSet, dbSet, dbSet_ape, dbSet_benchvise, dbSet_cam, dbSet_cat, dbSet_duck, batchSize=100):

    inputImagePaths = [] # The set to store the anchor, puller, pusher paths
    indices = [randint(0, batchSize) for p in range(batchSize)]
    # print('Indices: {}'.format(indices))
    anchors = [trainSet[i] for i in indices]
    anchors = [[data[0], float(data[1]), float(data[2]), float(data[3]), float(data[4])] for data in anchors]
    # print(np.shape(anchors))
    # exit(0)
    for anchor in anchors:
        theta = 50.0
        anchor_quaternions = [anchor[1], anchor[2], anchor[3], anchor[4]]
        if anchor[0].__contains__("ape"):
            # Compare with quaternions of dbSet_ape and find proper puller
            puller_set = dbSet_ape
        elif anchor[0].__contains__("benchvise"):
            # Compare with quaternions of dbSet_benchvise and find proper puller
            puller_set = dbSet_benchvise
        elif anchor[0].__contains__("cam"):
            # Compare with quaternions of dbSet_cam and find proper puller
            puller_set = dbSet_cam
        elif anchor[0].__contains__("cat"):
            # Compare with quaternions of dbSet_cat and find proper puller
            puller_set = dbSet_cat
        elif anchor[0].__contains__("duck"):
            # Compare with quaternions of dbSet_duck and find proper puller
            puller_set = dbSet_duck
        puller_final = []
        for puller in puller_set:
            puller_quaternions = [float(puller[1]), float(puller[2]), float(puller[3]), float(puller[4])]
            if np.absolute(np.dot(anchor_quaternions, puller_quaternions)) >= -1 and np.absolute(np.dot(anchor_quaternions, puller_quaternions)) <= 1:
                theta_new = 2 * np.arccos(np.absolute(np.dot(anchor_quaternions, puller_quaternions)))
                # print(theta_new)
                if theta_new < theta:
                    theta = theta_new
                    puller_final = puller
        puller_final = [puller_final[0], float(puller_final[1]), float(puller_final[2]), float(puller_final[3]), float(puller_final[4])]

        # Choose a random pusher
        pusher_index = randint(0,np.shape(dbSet)[0]-1)
        # print('Pusher_index: {}'.format(pusher_index))
        pusher = dbSet[pusher_index]
        pusher = [pusher[0], float(pusher[1]), float(pusher[2]), float(pusher[3]), float(pusher[4])]

        # Append the paths of anchor, puller and pusher to inputImagePaths
        inputImagePaths += [[anchor, puller_final, pusher]]

    return inputImagePaths