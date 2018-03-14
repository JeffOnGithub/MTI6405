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
segnet.load_weights('../weights/MTI_SegNet.hdf5')
print("Weights loaded")

print('Timer started')
start_time = time.time()

# Run images in the network
result = segnet.predict(np_pred_imgs)
print("Predictions generated")

# Reshape result images
result_imgs = []
for image in result:
    reshaped = np.reshape(image[:, 1], (IMG_WIDTH, IMG_HEIGHT))
    result_imgs.append(reshaped)

# Generate a combined images of all test images output
results_comb = np.hstack( (np.asarray(i) for i in result_imgs ) )
#plt.imsave('combined_output.png', results_comb)

# Filter/boost the result to generate a binary map
binary_maps = []
for image in result_imgs:
    #Build 3x3 flat kernel
    kernel = np.ones((3,3),np.float) / 9
    #Apply to image
    convulted = cv2.filter2D(image, -1, kernel)
    #Threshold
    filtered = cv2.threshold(convulted, 0.9, 1, cv2.THRESH_BINARY_INV)[1]
    #Reapply kernel
    reconvulted = cv2.filter2D(filtered, -1, kernel)
    #New threshold
    boosted = cv2.threshold(reconvulted, 0.5, 1, cv2.THRESH_BINARY_INV)[1]
    #Append result
    binary_maps.append(boosted)

print("Filtering/Boosting done")

print('Timer stopped')
print('Total time: ' + str(time.time() - start_time) + ' seconds')
print('Time/image: ' + str((time.time() - start_time) / HOW_MANY_IMAGES) + ' seconds')

# Generate a combined images of all binary maps
maps_comb = np.asarray(np.hstack( (np.asarray(i) for i in binary_maps ) ), dtype=np.float32)
#plt.imsave('combined_maps', maps_comb)

# Compare predicted map to truth map
map_diffs = []
for i in range(len(binary_maps)):
    diff = np.zeros([IMG_WIDTH, IMG_HEIGHT, 3])
    performance = [0, 0, 0, 0]
    for x in range(0, IMG_WIDTH):
        for y in range(0, IMG_HEIGHT):
            if binary_maps[i][x][y] >= 0.5 and truth_maps[i][x][y][1] == 1.0:
                diff[x,y] = [0, 1, 0]
                performance[0] += 1
            elif binary_maps[i][x][y] < 0.5 and truth_maps[i][x][y][1] == 0.0:
                diff[x,y] = [0, 0, 0]
                performance[1] += 1
            elif binary_maps[i][x][y] < 0.5 and truth_maps[i][x][y][1] == 1.0:
                diff[x,y] = [1, 1, 0]
                performance[2] += 1
            elif binary_maps[i][x][y] >= 0.5 and truth_maps[i][x][y][1] == 0.0:
                diff[x,y] = [1, 0, 0]
                performance[3] += 1
    map_diffs.append(diff)

    #Save result in numerical form to text file
    with open(RESULTS_FOLDER + "combined.txt", "a") as text_file:
        print(performance, file=text_file)

print("Maps compared")

# Generate a combined images of all binary maps
diffs_comb = np.asarray(np.hstack( (np.asarray(i) for i in map_diffs ) ), dtype=np.float32)
#plt.imsave('combined_diffs.png', diffs_comb)

# Stack combined images
imgs_comb = cv2.cvtColor(imgs_comb, cv2.COLOR_BGR2RGB) / 255.0
results_comb = cv2.cvtColor(results_comb, cv2.COLOR_GRAY2RGB)
maps_comb = cv2.cvtColor(maps_comb, cv2.COLOR_GRAY2RGB)

imgs_to_stack = [imgs_comb, results_comb, maps_comb, diffs_comb]
imgs_total = np.vstack( (np.asarray(i) for i in imgs_to_stack ) )

# Save result
plt.imsave(RESULTS_FOLDER + 'combined.png', imgs_total)

print("Result compilation saved")