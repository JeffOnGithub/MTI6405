# -*- coding: utf-8 -*-
import numpy as np
import cv2
from SegNet import CreateSegNet
import matplotlib.pyplot as plt

#CONFIGURATION
TEST_FOLDER = "test/"
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
segnet.load_weights('LIP_SegNet.hdf5')
print("Weights loaded")

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
    filtered = np.zeros([IMG_WIDTH, IMG_HEIGHT])
    threshold = 3
    
    for x in range(1, IMG_WIDTH - 1):
        for y in range(1, IMG_HEIGHT - 1):
            this_cell = 0
            for delta_x in range (-1, 2):
                for delta_y in range(-1, 2):
                    this_cell += image[x + delta_x, y + delta_y]
            if this_cell > threshold:
                filtered[x, y] = 1
    
    boosted = np.zeros([IMG_WIDTH, IMG_HEIGHT])
    threshold = 20
    
    for x in range(2, IMG_WIDTH - 2):
        for y in range(2, IMG_HEIGHT - 2):
            this_cell = 0
            for delta_x in range (-2, 3):
                for delta_y in range(-2, 3):
                    this_cell += filtered[x + delta_x, y + delta_y]
            if this_cell > threshold:
                boosted[x, y] = 1
    
    binary_maps.append(boosted)

print("Filtering/Boosting done")

# Generate a combined images of all binary maps
maps_comb = np.asarray(np.hstack( (np.asarray(i) for i in binary_maps ) ), dtype=np.float32)
#plt.imsave('combined_maps', maps_comb)

# Compare predicted map to truth map
map_diffs = []
for i in range(len(binary_maps)):
    diff = np.zeros([IMG_WIDTH, IMG_HEIGHT, 3])
    for x in range(0, IMG_WIDTH):
        for y in range(0, IMG_HEIGHT):
            if binary_maps[i][x][y] == 1.0 and truth_maps[i][x][y][1] == 1.0:
                diff[x,y] = [0, 1, 0]
            elif binary_maps[i][x][y] == 0.0 and truth_maps[i][x][y][1] == 0.0:
                diff[x,y] = [0, 0, 0]    
            elif binary_maps[i][x][y] == 0.0 and truth_maps[i][x][y][1] == 1.0:
                diff[x,y] = [1, 1, 0]
            elif binary_maps[i][x][y] == 1.0 and truth_maps[i][x][y][1] == 0.0:
                diff[x,y] = [1, 0, 0]
    map_diffs.append(diff)

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
plt.imsave('combined.png', imgs_total)

print("Result compilation saved")
