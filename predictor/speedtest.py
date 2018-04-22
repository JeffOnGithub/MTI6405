# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../segnet/')
from SegNet import CreateSegNet
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

#CONFIGURATION
TEST_FOLDER = "../test/"
RESULTS_FOLDER = "../results/"
HOW_MANY_IMAGES = 30
IMG_HEIGHT = 256
IMG_WIDTH = 256

WEIGHT = '../weights/MTI_SegNet-60.hdf5'
THRESHOLD_1 = 0.3
THRESHOLD_2 = 0.5

# Load ground truth maps
truth_maps = []
for i in range(0, HOW_MANY_IMAGES):
    truth_maps.append(cv2.imread(TEST_FOLDER + str(i) + '.png'))
np_pred_imgs = np.array(truth_maps)

# Load test images
pred_imgs = []
for i in range(0, HOW_MANY_IMAGES):
    pred_imgs.append(cv2.imread(TEST_FOLDER + str(i) + '.jpg'))
np_pred_imgs = np.array(pred_imgs)

# Generate a combined images of all test images input
imgs_comb = np.hstack( (np.asarray(i) for i in pred_imgs ) )
#plt.imsave('combined_input.png', cv2.cvtColor(imgs_comb, cv2.COLOR_BGR2RGB))

# Build a network and load weights
segnet = CreateSegNet((IMG_WIDTH, IMG_HEIGHT, 3), 2, 3, (2, 2), "softmax")
print("Segnet built")
segnet.load_weights(WEIGHT)
print("Weights loaded")

#Clear result text file
with open(RESULTS_FOLDER + "speed_results.txt", "w") as text_file:
    pass

all_times = []
for i in range(0,1000):
    print('Timer started')
    start_time = time.time()
    
    # Run images in the network
    result = segnet.predict(np_pred_imgs)
    #print("Predictions generated")
    
    # Reshape result images
    result_imgs = []
    for image in result:
        reshaped = np.reshape(image[:, 1], (IMG_WIDTH, IMG_HEIGHT))
        result_imgs.append(reshaped)
    
    # Generate a combined images of all test images output
    results_comb = np.hstack( (np.asarray(i) for i in result_imgs ) )
    #plt.imsave('combined_output.png', results_comb)
    
    # Filter/boost the result to generate a binary map
    filtered_maps = []
    binary_maps = []
    for image in result_imgs:
        #Build flat kernels
        kernel3x3 = np.ones((3,3),np.float) / 9
        kernel5x5 = np.ones((5,5),np.float) / 25
        #Apply to image
        convulted = cv2.filter2D(image, -1, kernel3x3)
        #Threshold
        filtered = cv2.threshold(convulted, THRESHOLD_1, 1, cv2.THRESH_BINARY)[1]
        filtered_maps.append(filtered)
        #Reapply kernel
        reconvulted = cv2.filter2D(filtered, -1, kernel5x5)
        #New threshold
        boosted = cv2.threshold(reconvulted, THRESHOLD_2, 1, cv2.THRESH_BINARY)[1]
        #Append result
        binary_maps.append(boosted)
    
    #print("Filtering/Boosting done")
    
    #Timer results
    print('Timer stopped run ' + str(i))
    #print('Total time: ' + str(time.time() - start_time) + ' seconds')
    #print('Time/image: ' + str((time.time() - start_time) / HOW_MANY_IMAGES) + ' seconds')
    
    with open(RESULTS_FOLDER + "speed_results.txt", "a") as text_file:
        print(str(time.time() - start_time), file=text_file)
    
    all_times.append(time.time() - start_time)

