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
THRESHOLD_1 = 0.7
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
    #Build flat kernels
    kernel3x3 = np.ones((3,3),np.float) / 9
    kernel5x5 = np.ones((5,5),np.float) / 25
    #Apply to image
    convulted = cv2.filter2D(image, -1, kernel3x3)
    #Threshold
    filtered = cv2.threshold(convulted, THRESHOLD_1, 1, cv2.THRESH_BINARY_INV)[1]
    #Reapply kernel
    reconvulted = cv2.filter2D(filtered, -1, kernel5x5)
    #New threshold
    boosted = cv2.threshold(reconvulted, THRESHOLD_2, 1, cv2.THRESH_BINARY_INV)[1]
    #Append result
    binary_maps.append(boosted)

print("Filtering/Boosting done")

#Timer results
print('Timer stopped')
print('Total time: ' + str(time.time() - start_time) + ' seconds')
print('Time/image: ' + str((time.time() - start_time) / HOW_MANY_IMAGES) + ' seconds')

#Clear result text file
with open(RESULTS_FOLDER + "combined.txt", "w") as text_file:
    pass

with open(RESULTS_FOLDER + "combined.txt", "a") as text_file:
    print(str(time.time() - start_time), file=text_file)
    print(str((time.time() - start_time) / HOW_MANY_IMAGES), file=text_file)

# Generate a combined images of all binary maps
maps_comb = np.asarray(np.hstack( (np.asarray(i) for i in binary_maps ) ), dtype=np.float32)
#plt.imsave('combined_maps', maps_comb)

def compare_images(compared_image, truth_image):
    # Compare predicted map to truth map
    all_diffs = []
    all_perfs = []
    for i in range(len(compared_image)):
        diff = np.zeros([IMG_WIDTH, IMG_HEIGHT, 3])
        performance = [0, 0, 0, 0, 0, 0]
        for x in range(0, IMG_WIDTH):
            for y in range(0, IMG_HEIGHT):
                if compared_image[i][x][y] >= 0.5 and truth_image[i][x][y][1] == 1.0:
                    diff[x,y] = [0, 1, 0]
                    performance[0] += 1
                elif compared_image[i][x][y] < 0.5 and truth_image[i][x][y][1] == 0.0:
                    diff[x,y] = [0, 0, 0]
                    performance[1] += 1
                elif compared_image[i][x][y] < 0.5 and truth_image[i][x][y][1] == 1.0:
                    diff[x,y] = [1, 1, 0]
                    performance[2] += 1
                elif compared_image[i][x][y] >= 0.5 and truth_image[i][x][y][1] == 0.0:
                    diff[x,y] = [1, 0, 0]
                    performance[3] += 1
        all_diffs.append(diff)
        all_perfs.append(performance)
    
        #Save result in numerical form to text file
        #Number of foreground pixels
        nb_fg_px = np.sum(truth_image[i][:,:,1])
        #% of correctly identified fg pixels
        performance[4] = performance[0] / nb_fg_px
        #Number of background pixels
        nb_bg_px = IMG_HEIGHT * IMG_WIDTH - nb_fg_px
        #% of correctly identified bg pixels
        performance[5] = performance[1] / nb_bg_px
    
        with open(RESULTS_FOLDER + "combined.txt", "a") as text_file:
            print(performance, file=text_file)
        
    return (all_diffs, np.asarray(all_perfs))


# Generate a combined images of all binary maps
(segnet_diffs, segnet_perfs) = compare_images(result_imgs, truth_maps)
segnet_diffs_comb = np.asarray(np.hstack( (np.asarray(i) for i in segnet_diffs ) ), dtype=np.float32)

# Generate a combined images of all binary maps
(map_diffs, maps_perfs) = compare_images(binary_maps, truth_maps)
final_diffs_comb = np.asarray(np.hstack( (np.asarray(i) for i in map_diffs ) ), dtype=np.float32)
#plt.imsave('combined_diffs.png', diffs_comb)

print("Maps compared")

print("Segnet results foreground " + str(1 - np.average(segnet_perfs[:,4])))
print("Segnet results background " + str(1 - np.average(segnet_perfs[:,5])))

print("Politic results foreground " + str(1 - np.average(maps_perfs[:,4])))
print("Politic results background " + str(1 - np.average(maps_perfs[:,5])))

# Stack combined images
imgs_comb = cv2.cvtColor(imgs_comb, cv2.COLOR_BGR2RGB) / 255.0
results_comb = cv2.cvtColor(results_comb, cv2.COLOR_GRAY2RGB)
maps_comb = cv2.cvtColor(maps_comb, cv2.COLOR_GRAY2RGB)

imgs_to_stack = [imgs_comb, results_comb, segnet_diffs_comb, maps_comb, final_diffs_comb]
imgs_total = np.vstack( (np.asarray(i) for i in imgs_to_stack ) )

# Save result
plt.imsave(RESULTS_FOLDER + 'combined.png', imgs_total)

print("Result compilation saved")